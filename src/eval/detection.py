"""Métricas de detecção.

Duas ferramentas, com propósitos distintos:

* ``coco_metrics`` usa a implementação de referência (pycocotools). Os números saem
  diretamente comparáveis aos baselines publicados do MinneApple (Faster R-CNN AP 0,438,
  Tiled Faster R-CNN AP 0,341).

* ``match_predictions`` é um casamento guloso próprio, que devolve o destino de CADA predição
  e de CADA anotação. O pycocotools não expõe isso, e a análise de modos de erro precisa
  saber exatamente quais frutas foram perdidas para poder cruzar com área, oclusão e
  aglomeração.

Armadilha importante do pycocotools tratada aqui: o padrão ``maxDets = [1, 10, 100]`` trunca
em 100 detecções por imagem. O MinneApple chega a 123 maçãs numa única imagem, e a
inferência em tiles produz ainda mais caixas antes da fusão, então o padrão descartaria
detecções corretas e subestimaria o recall. O limite é elevado para 300.
"""
from __future__ import annotations

import contextlib
import io
from dataclasses import dataclass

import numpy as np

# Cortes de área do COCO, usados para AP-small / medium / large.
AREA_SMALL = 32**2
AREA_MEDIUM = 96**2
MAX_DETS = 300


# ------------------------------------------------------------------ casamento guloso próprio

@dataclass(frozen=True)
class MatchResult:
    """Destino de cada predição e de cada anotação, numa imagem."""

    pred_to_gt: np.ndarray  # (P,) índice do GT casado, ou -1 se falso positivo
    gt_to_pred: np.ndarray  # (G,) índice da predição casada, ou -1 se falso negativo
    iou: np.ndarray         # (P,) IoU do casamento, 0 para falso positivo

    @property
    def tp(self) -> np.ndarray:
        return np.flatnonzero(self.pred_to_gt >= 0)

    @property
    def fp(self) -> np.ndarray:
        return np.flatnonzero(self.pred_to_gt < 0)

    @property
    def fn(self) -> np.ndarray:
        return np.flatnonzero(self.gt_to_pred < 0)

    def counts(self) -> tuple[int, int, int]:
        return len(self.tp), len(self.fp), len(self.fn)


def match_predictions(
    pred_boxes: np.ndarray,
    pred_scores: np.ndarray,
    gt_boxes: np.ndarray,
    iou_threshold: float = 0.5,
) -> MatchResult:
    """Casa predições a anotações, da maior confiança para a menor, um-para-um.

    É o mesmo critério do COCO num único limiar de IoU: cada predição fica com a anotação
    livre de maior IoU acima do limiar; anotações já tomadas não podem ser reutilizadas.
    """
    from src.tiling.merge import iou_matrix

    n_pred, n_gt = len(pred_boxes), len(gt_boxes)
    pred_to_gt = np.full(n_pred, -1, dtype=np.int64)
    gt_to_pred = np.full(n_gt, -1, dtype=np.int64)
    best_iou = np.zeros(n_pred, dtype=np.float32)
    if n_pred == 0 or n_gt == 0:
        return MatchResult(pred_to_gt, gt_to_pred, best_iou)

    ious = iou_matrix(pred_boxes, gt_boxes)
    for p in np.argsort(-np.asarray(pred_scores), kind="stable"):
        candidates = np.where(gt_to_pred < 0, ious[p], -1.0)
        g = int(np.argmax(candidates))
        if candidates[g] >= iou_threshold:
            pred_to_gt[p] = g
            gt_to_pred[g] = p
            best_iou[p] = candidates[g]
    return MatchResult(pred_to_gt, gt_to_pred, best_iou)


def operating_point_stats(
    matches: list[MatchResult],
) -> dict[str, float]:
    """Precisão, recall e F1 agregados no ponto de operação já aplicado.

    Diferente do mAP, que varre todos os limiares: isto descreve o que o sistema realmente
    faz quando roda com um limiar fixo, que é a única forma em que ele existe em produção.
    """
    tp = sum(len(m.tp) for m in matches)
    fp = sum(len(m.fp) for m in matches)
    fn = sum(len(m.fn) for m in matches)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-9)
    return {"tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall, "f1": f1}


# --------------------------------------------------------------------------- COCO oficial

def _to_coco(
    gt_per_image: dict[str, np.ndarray],
    pred_per_image: dict[str, tuple[np.ndarray, np.ndarray]],
    image_wh: tuple[int, int],
) -> tuple[dict, list[dict]]:
    """Monta as estruturas COCO em memória, sem passar por disco.

    O conjunto de imagens é a UNIÃO das chaves de anotação e de predição. Usar só as chaves do
    ground truth descartaria em silêncio todas as detecções de uma imagem sem anotação — o que
    infla o AP de graça, porque falsos positivos somem sem entrar na conta.
    """
    w, h = image_wh
    names = sorted(set(gt_per_image) | set(pred_per_image))
    image_ids = {name: i + 1 for i, name in enumerate(names)}

    images = [
        {"id": image_ids[n], "file_name": n, "width": w, "height": h} for n in names
    ]
    annotations, ann_id = [], 1
    empty = np.zeros((0, 4), np.float32)
    for name in names:
        for x0, y0, x1, y1 in gt_per_image.get(name, empty):
            annotations.append(
                {
                    "id": ann_id,
                    "image_id": image_ids[name],
                    "category_id": 1,
                    "bbox": [float(x0), float(y0), float(x1 - x0), float(y1 - y0)],
                    "area": float((x1 - x0) * (y1 - y0)),
                    "iscrowd": 0,
                }
            )
            ann_id += 1

    detections = []
    for name in names:
        boxes, scores = pred_per_image.get(
            name, (np.zeros((0, 4), np.float32), np.zeros(0, np.float32))
        )
        for (x0, y0, x1, y1), s in zip(boxes, scores):
            detections.append(
                {
                    "image_id": image_ids[name],
                    "category_id": 1,
                    "bbox": [float(x0), float(y0), float(x1 - x0), float(y1 - y0)],
                    "score": float(s),
                }
            )

    coco_gt = {
        "images": images,
        "annotations": annotations,
        "categories": [{"id": 1, "name": "apple"}],
    }
    return coco_gt, detections


def _mean_valid(block: np.ndarray) -> float:
    """Média dos valores válidos de uma fatia do COCO.

    O pycocotools marca faixa vazia com -1. Isso é uma SENTINELA, não uma métrica: propagar
    -1 para um CSV faz qualquer média ou gráfico posterior ficar errado sem avisar. Neste
    dataset a faixa "large" está vazia (0,0% das instâncias acima de 96² px), então essa
    situação acontece em toda execução.
    """
    valid = block[block > -1]
    return float(valid.mean()) if valid.size else float("nan")


def coco_metrics(
    gt_per_image: dict[str, np.ndarray],
    pred_per_image: dict[str, tuple[np.ndarray, np.ndarray]],
    image_wh: tuple[int, int] = (720, 1280),
) -> dict[str, float]:
    """AP no protocolo COCO, incluindo o corte por tamanho de objeto.

    ``AP_small`` é a métrica que mais importa aqui: o lado mediano da caixa no MinneApple é de
    ~27 px, então a maioria esmagadora do dataset cai abaixo do corte de 32x32 do COCO.

    Os números são lidos de ``ev.eval``, e não de ``ev.stats``. O motivo é uma armadilha
    concreta do pycocotools: ``_summarizeDets`` calcula ``stats[0]`` — o AP principal — sem
    repassar ``maxDets``, usando o valor 100 embutido na assinatura de ``_summarize``. Se
    ``params.maxDets`` for alterado e não contiver 100, a seleção do índice devolve fatia
    vazia e o AP sai -1,0. Como aqui o limite precisa subir para 300 (há imagens com 123
    maçãs, e o padrão de 100 descartaria detecções corretas), usar ``stats[0]`` reportaria
    -1,0 silenciosamente. Todas as outras entradas de ``stats`` repassam ``maxDets``
    corretamente; só a primeira não.

    Nota sobre ``image_wh``: para ``iouType="bbox"`` o pycocotools não usa largura e altura, e
    verificamos que trocá-las não altera nenhum número. O valor correto está aqui mesmo assim,
    porque um default transposto é exatamente o tipo de coisa que alguém copia para um lugar
    onde importa.
    """
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    keys = ["AP", "AP50", "AP75", "AP_small", "AP_medium", "AP_large", "AR_300"]
    coco_gt_dict, detections = _to_coco(gt_per_image, pred_per_image, image_wh)
    if not detections:
        return dict.fromkeys(keys, 0.0)

    # pycocotools escreve direto no stdout; silenciado para não poluir a saída dos scripts.
    with contextlib.redirect_stdout(io.StringIO()):
        coco_gt = COCO()
        coco_gt.dataset = coco_gt_dict
        coco_gt.createIndex()
        ev = COCOeval(coco_gt, coco_gt.loadRes(detections), iouType="bbox")
        ev.params.maxDets = [1, 10, MAX_DETS]
        ev.evaluate()
        ev.accumulate()

    # precision: [IoU, recall, categoria, faixa de área, maxDets]
    # recall:    [IoU,          categoria, faixa de área, maxDets]
    precision, recall = ev.eval["precision"], ev.eval["recall"]
    m = len(ev.params.maxDets) - 1                      # índice de maxDets = MAX_DETS
    iou50 = int(np.argmin(np.abs(ev.params.iouThrs - 0.50)))
    iou75 = int(np.argmin(np.abs(ev.params.iouThrs - 0.75)))
    ALL, SMALL, MEDIUM, LARGE = 0, 1, 2, 3              # ordem de params.areaRng

    return {
        "AP": _mean_valid(precision[:, :, :, ALL, m]),
        "AP50": _mean_valid(precision[iou50, :, :, ALL, m]),
        "AP75": _mean_valid(precision[iou75, :, :, ALL, m]),
        "AP_small": _mean_valid(precision[:, :, :, SMALL, m]),
        "AP_medium": _mean_valid(precision[:, :, :, MEDIUM, m]),
        "AP_large": _mean_valid(precision[:, :, :, LARGE, m]),
        "AR_300": _mean_valid(recall[:, :, ALL, m]),
    }
