"""Teste de integração do caminho completo de inferência, sem modelo treinado.

A imagem é sintética e o detector é um localizador de manchas via OpenCV. Isso permite
conhecer a contagem verdadeira exatamente e verificar a afirmação central da Tarefa 2:

    o tiling introduz duplicatas, e é a política de fusão — não o tiling em si — que
    determina se a contagem final está certa.

O detector falso reproduz o comportamento que importa: ao receber um recorte, ele enxerga
apenas o pedaço da fruta que está dentro do recorte, e portanto emite caixas truncadas nas
arestas, exatamente como um detector de verdade faria.
"""
from __future__ import annotations

import cv2
import numpy as np
import pytest

from src.inference.engine import Arm, RawDetections, run_arm
from src.inference.postprocess import MergePolicy, apply

IMAGE_WH = (720, 1280)  # retrato, como o MinneApple
RADIUS = 20

# Centros escolhidos à mão: alguns caem no meio de um tile, outros nas faixas de sobreposição.
# Na grade de 640 as faixas ficam em x 80..640 e y 512..640; na de 320, em x 256..320 e
# 400..576, y 256..320, 512..576, 768..832, 960..1024.
CENTRES = [
    (120, 120), (200, 300), (160, 560), (400, 600), (120, 700),
    (300, 900), (180, 1100), (560, 280), (620, 530), (360, 640),
    (560, 760), (600, 980), (480, 1180), (380, 180), (460, 420),
    (160, 860), (420, 1020), (660, 340), (660, 660), (660, 1150),
]


@pytest.fixture(scope="module")
def synthetic_image() -> np.ndarray:
    """Fundo preto com discos brancos. Cada disco é uma 'fruta'."""
    w, h = IMAGE_WH
    img = np.zeros((h, w, 3), dtype=np.uint8)
    for cx, cy in CENTRES:
        cv2.circle(img, (cx, cy), RADIUS, (255, 255, 255), thickness=-1)
    return img


class BlobDetector:
    """Detector falso: componentes conexas do recorte que recebe.

    Só enxerga o que está dentro do recorte, então produz caixas cortadas nas arestas —
    que é a fonte das duplicatas que a fusão precisa resolver.
    """

    def predict(self, images: list[np.ndarray], imgsz: int):
        out = []
        for img in images:
            gray = np.ascontiguousarray(img[..., 0] if img.ndim == 3 else img)
            n, _, stats, _ = cv2.connectedComponentsWithStats((gray > 0).astype(np.uint8), 8)
            boxes = [
                [x, y, x + w, y + h]
                for x, y, w, h, _ in (stats[i] for i in range(1, n))
                if w >= 2 and h >= 2
            ]
            out.append(
                (
                    np.asarray(boxes, dtype=np.float32).reshape(-1, 4),
                    np.full(len(boxes), 0.9, dtype=np.float32),
                )
            )
        return out


ARMS = {
    "A_full640": Arm("A_full640", tile=None, imgsz=640, overlap=0.0),
    "B_full1280": Arm("B_full1280", tile=None, imgsz=1280, overlap=0.0),
    "C_tile640": Arm("C_tile640", tile=640, imgsz=640, overlap=0.2),
    "D_tile320": Arm("D_tile320", tile=320, imgsz=640, overlap=0.2),
}


def test_imagem_inteira_encontra_a_contagem_exata(synthetic_image):
    raw = run_arm(BlobDetector(), synthetic_image, ARMS["A_full640"])
    assert raw.n_tiles == 1
    assert len(raw) == len(CENTRES)


@pytest.mark.parametrize("arm_name", ["C_tile640", "D_tile320"])
def test_tiling_produz_duplicatas_antes_da_fusao(synthetic_image, arm_name):
    """Premissa da tarefa: sem fusão, tiles sobrepostos contam fruta mais de uma vez."""
    raw = run_arm(BlobDetector(), synthetic_image, ARMS[arm_name])
    assert len(raw) > len(CENTRES)
    assert raw.truncated.any()


@pytest.mark.parametrize("arm_name", ["C_tile640", "D_tile320"])
def test_fusao_ios_recupera_a_contagem_exata(synthetic_image, arm_name):
    """Com IoS e descarte de borda, a contagem volta a ser exata nos dois braços de tiling."""
    raw = run_arm(BlobDetector(), synthetic_image, ARMS[arm_name])
    det = apply(raw, MergePolicy(metric="ios", policy="nmm", drop_truncated=True), conf=0.5)
    assert det.count == len(CENTRES)
    assert det.duplicates_removed > 0


def test_iou_sozinho_deixa_duplicatas_passarem(synthetic_image):
    """A falha que a escolha de métrica evita — e que o baseline tiled do artigo sofreu.

    Sem descarte de borda e casando por IoU, as meias-frutas das arestas sobrevivem e a
    contagem estoura. Trocar apenas a métrica para IoS já corrige a maior parte.
    """
    raw = run_arm(BlobDetector(), synthetic_image, ARMS["D_tile320"])
    iou_only = apply(raw, MergePolicy(metric="iou", policy="nms", drop_truncated=False), conf=0.5)
    ios_only = apply(raw, MergePolicy(metric="ios", policy="nms", drop_truncated=False), conf=0.5)

    assert iou_only.count > len(CENTRES), "IoU deveria super-contar, e super-conta"
    assert ios_only.count < iou_only.count, "IoS remove duplicatas que o IoU deixa passar"


def test_descarte_de_borda_nunca_remove_fruta_da_borda_da_imagem(synthetic_image):
    """Fruta colada na borda REAL da imagem não pode ser descartada: nenhum tile a vê inteira.

    Se o descarte confundisse aresta de tile com borda de imagem, as frutas das extremidades
    sumiriam e a contagem cairia sistematicamente.
    """
    w, h = IMAGE_WH
    img = synthetic_image.copy()
    cv2.circle(img, (8, h // 2), RADIUS, (255, 255, 255), thickness=-1)  # colada à esquerda
    cv2.circle(img, (w - 8, h // 2), RADIUS, (255, 255, 255), thickness=-1)  # à direita
    # (as duas ficam a meia altura, longe das outras, para não fundirem com nada)

    raw = run_arm(BlobDetector(), img, ARMS["C_tile640"])
    det = apply(raw, MergePolicy(drop_truncated=True), conf=0.5)
    assert det.count == len(CENTRES) + 2


def test_custo_de_cada_braco_bate_com_o_numero_de_tiles(synthetic_image):
    """Confere a coluna de custo da tabela do relatório."""
    counts = {
        name: run_arm(BlobDetector(), synthetic_image, arm).n_tiles
        for name, arm in ARMS.items()
    }
    assert counts == {"A_full640": 1, "B_full1280": 1, "C_tile640": 6, "D_tile320": 15}


def test_magnificacao_declarada_por_braco():
    """O braço C não magnifica nada; só o D leva o objeto a 2x."""
    assert ARMS["A_full640"].magnification == pytest.approx(0.5)
    assert ARMS["B_full1280"].magnification == pytest.approx(1.0)
    assert ARMS["C_tile640"].magnification == pytest.approx(1.0)
    assert ARMS["D_tile320"].magnification == pytest.approx(2.0)


def test_politica_cross_tile_nao_se_aplica_na_imagem_inteira():
    """No braço de um passe só, a supressão é NMS por IoU — nunca a política cross-tile.

    A distinção importa: o IoS casa uma caixa pequena contida numa maior e apagaria fruta
    legítima em cacho. Sem aresta de tile cortando fruta, o argumento a favor do IoS não se
    aplica; o problema ali é a caixa repetida, e o IoU resolve sem esse efeito colateral.
    """
    grande = [100.0, 100.0, 200.0, 200.0]
    pequena_dentro = [120.0, 120.0, 150.0, 150.0]   # IoS 1,0 com a grande, IoU baixo
    raw = RawDetections(
        boxes=np.array([grande, pequena_dentro], np.float32),
        scores=np.array([0.9, 0.8], np.float32),
        tile_index=np.zeros(2, np.int32), truncated=np.zeros(2, bool),
        n_tiles=1, latency_ms={},
    )
    agressiva = MergePolicy(metric="ios", policy="nmm", threshold=0.3, drop_truncated=True)
    assert apply(raw, agressiva, conf=0.25).count == 2, "as duas frutas devem sobreviver"


def test_duplicata_na_mesma_fruta_e_removida_na_imagem_inteira():
    """A cabeça NMS-free do YOLO26 emite duplicata, e a supressão dentro da imagem existe por isso.

    Medido no dataset real: 24,2% dos falsos positivos do braço de imagem inteira eram uma
    segunda caixa numa maçã que outra detecção já havia casado. Uma versão anterior pulava a
    supressão nesse braço, apostando que a cabeça um-para-um bastava. Não basta.
    """
    caixa = [100.0, 100.0, 140.0, 140.0]
    quase_igual = [102.0, 102.0, 142.0, 142.0]     # IoU ~0,82 com a primeira
    raw = RawDetections(
        boxes=np.array([caixa, quase_igual], np.float32),
        scores=np.array([0.9, 0.7], np.float32),
        tile_index=np.zeros(2, np.int32), truncated=np.zeros(2, bool),
        n_tiles=1, latency_ms={},
    )
    assert apply(raw, MergePolicy(), conf=0.25).count == 1


def test_caixa_degenerada_ainda_e_removida_na_imagem_inteira():
    """Pular a fusão não pode reabrir a porta para caixas de área zero."""
    raw = RawDetections(
        boxes=np.array([[10.0, 10.0, 50.0, 50.0], [60.0, 60.0, 60.0, 90.0]], np.float32),
        scores=np.array([0.9, 0.8], np.float32),
        tile_index=np.zeros(2, np.int32),
        truncated=np.zeros(2, bool),
        n_tiles=1,
        latency_ms={},
    )
    assert apply(raw, MergePolicy(), conf=0.25).count == 1


def test_device_cai_para_cpu_quando_nao_ha_gpu(monkeypatch):
    """Uma máquina sem GPU tem que rodar, não abortar.

    Os scripts pedem `cuda:0` porque foi onde tudo rodou, mas o repositório precisa
    funcionar num clone sem GPU — container, CI, notebook de quem revisa. Antes desta
    tradução o Ultralytics levantava `Invalid CUDA 'device=0' requested` e o pipeline
    inteiro morria numa máquina que teria feito o trabalho em CPU.
    """
    import torch

    from src.inference.engine import resolve_device

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert resolve_device("cuda:0") == "cpu"
    assert resolve_device(None) == "cpu"

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert resolve_device("cuda:0") == "cuda:0"
    assert resolve_device(None) == "cuda:0"
    # 'cpu' explícito é respeitado mesmo com GPU disponível: é como se não disputa a placa
    # com um treino em andamento.
    assert resolve_device("cpu") == "cpu"
