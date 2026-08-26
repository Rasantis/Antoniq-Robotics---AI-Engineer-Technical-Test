"""Leitura do MinneApple: máscaras de instância -> caixas + estatísticas por fruta.

As máscaras vêm como PNG de canal único em que cada maçã tem um valor de pixel próprio
(0 = fundo). Além da caixa, extraímos aqui a área da máscara, da qual sai a solidez
(área da máscara / área da caixa). A solidez é o nosso proxy de oclusão: uma maçã inteira é
quase circular e preenche ~0,79 da caixa; uma maçã atrás de uma folha vira um crescente e
preenche muito menos. A análise de modos de erro depende dessa coluna, por isso ela é
calculada aqui, uma vez, e não recomputada depois.

Só NumPy e Pillow, conforme a restrição do enunciado.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

# Largura x altura. O MinneApple e retrato; o artigo escreve "1280 x 720" (altura x
# largura), o que ja causou confusao suficiente para merecer esta nota.
NATIVE_WH = (720, 1280)


@dataclass(frozen=True)
class MaskInstances:
    """Instâncias extraídas de uma única máscara."""

    boxes: np.ndarray         # (N, 4) xyxy contínuo, borda direita/inferior exclusiva
    mask_areas: np.ndarray    # (N,) pixels efetivamente marcados
    instance_ids: np.ndarray  # (N,) valor de pixel original, para rastrear de volta

    def __len__(self) -> int:
        return len(self.boxes)


def read_mask(path: Path | str) -> np.ndarray:
    """Máscara de instâncias como array 2-D de inteiros."""
    with Image.open(path) as img:
        arr = np.array(img)
    if arr.ndim == 3:
        # Alguns PNG vêm com canais replicados; os IDs são idênticos nos três.
        arr = arr[..., 0]
    return arr


def instances_from_mask(mask: np.ndarray, min_side: int = 2) -> MaskInstances:
    """Extrai caixa e área para cada ID de instância presente na máscara.

    Agrupamento vetorizado por ordenação: um único argsort sobre os pixels marcados, depois
    reduções por segmento. Percorrer ``mask == id`` para cada uma das até 120 instâncias por
    imagem custaria ordens de magnitude mais.

    Instâncias com lado menor que ``min_side`` são descartadas: no MinneApple existem resíduos
    de máscara de 1-2 px que não correspondem a fruta anotável.
    """
    ys, xs = np.nonzero(mask)
    if ys.size == 0:
        return MaskInstances(
            boxes=np.zeros((0, 4), np.float32),
            mask_areas=np.zeros(0, np.int64),
            instance_ids=np.zeros(0, np.int64),
        )

    ids = mask[ys, xs]
    order = np.argsort(ids, kind="stable")
    ids, ys, xs = ids[order], ys[order], xs[order]
    uniq, starts, counts = np.unique(ids, return_index=True, return_counts=True)

    x0 = np.minimum.reduceat(xs, starts).astype(np.float32)
    y0 = np.minimum.reduceat(ys, starts).astype(np.float32)
    x1 = np.maximum.reduceat(xs, starts).astype(np.float32) + 1.0  # borda exclusiva
    y1 = np.maximum.reduceat(ys, starts).astype(np.float32) + 1.0

    boxes = np.stack([x0, y0, x1, y1], axis=1)
    keep = ((x1 - x0) >= min_side) & ((y1 - y0) >= min_side)
    return MaskInstances(
        boxes=boxes[keep],
        mask_areas=counts[keep].astype(np.int64),
        instance_ids=uniq[keep].astype(np.int64),
    )


def solidity(boxes: np.ndarray, mask_areas: np.ndarray) -> np.ndarray:
    """Fração da caixa que a máscara preenche. Proxy de oclusão em [0, 1].

    Referência: um disco inscrito preenche pi/4 = 0,785 da caixa. Valores bem abaixo disso
    indicam fruta parcialmente escondida por folha, galho ou outra fruta.
    """
    if len(boxes) == 0:
        return np.zeros(0, np.float32)
    wh = boxes[:, 2:] - boxes[:, :2]
    bbox_area = np.maximum(wh[:, 0] * wh[:, 1], 1.0)
    return (mask_areas / bbox_area).astype(np.float32)


def solidity_ceiling(w, h):
    """Solidez máxima ATINGÍVEL por uma fruta perfeitamente circular daquele tamanho.

    O valor de referência pi/4 = 0,785 vale para um disco contínuo. Uma máscara é discreta e
    a caixa vem de ``max - min + 1`` px, então um disco digital de lado ``s`` preenche
    ``(pi/4)·((s-1)/s)²`` — que para o lado mediano deste dataset, 27 px, dá 0,728, não
    0,785.

    A diferença não é acadêmica: com o corte em 0,80, o bin rotulado "limpa" capturava 6,1%
    das frutas e era inalcançável para fruta limpa de tamanho típico. Pior, ele ficava
    enriquecido em fruta colada na borda da imagem — cuja caixa é truncada pela moldura e
    portanto MAIS preenchida. O bin "limpa" concentrava o caso mais truncado que existe.

    Recebe largura e altura em vez da caixa porque os dois chamadores têm formatos
    diferentes — um traz um array de caixas, o outro colunas de um DataFrame — e a fórmula
    precisa morar num lugar só. Ela decide os cortes de oclusão da §5 do relatório; duas
    cópias divergindo em silêncio mudariam a tabela sem ninguém notar.
    """
    w = np.maximum(w, 1.0)
    h = np.maximum(h, 1.0)
    return (np.pi / 4) * ((w - 1) / w) * ((h - 1) / h)


def local_density(boxes: np.ndarray, radius: float = 100.0) -> np.ndarray:
    """Quantas outras frutas têm centro a menos de ``radius`` px. Proxy de aglomeração."""
    if len(boxes) == 0:
        return np.zeros(0, np.int32)
    centres = (boxes[:, :2] + boxes[:, 2:]) / 2.0
    d2 = ((centres[:, None, :] - centres[None, :, :]) ** 2).sum(-1)
    return ((d2 <= radius**2).sum(1) - 1).astype(np.int32)


def distance_to_border(boxes: np.ndarray, image_wh: tuple[int, int]) -> np.ndarray:
    """Menor distância do centro da caixa a qualquer borda da imagem, em px."""
    if len(boxes) == 0:
        return np.zeros(0, np.float32)
    w, h = image_wh
    cx = (boxes[:, 0] + boxes[:, 2]) / 2.0
    cy = (boxes[:, 1] + boxes[:, 3]) / 2.0
    return np.minimum.reduce([cx, cy, w - cx, h - cy]).astype(np.float32)
