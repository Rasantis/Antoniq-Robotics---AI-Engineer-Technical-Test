"""Geração de tiles sobrepostos.

Implementado apenas com NumPy (recorte por slicing) e OpenCV/Pillow no chamador, conforme a
restrição do enunciado. Nenhuma dependência de SAHI.
"""
from __future__ import annotations

import numpy as np


def _axis_offsets(length: int, tile: int, overlap: float) -> list[int]:
    """Deslocamentos ao longo de um eixo, com o último tile encostado na borda.

    Encostar o último tile na borda (em vez de preencher com padding) evita duas coisas:
    faixas de pixels artificiais, que produzem detecções espúrias, e um tile final com
    conteúdo útil menor que os demais.
    """
    tile = min(tile, length)
    step = max(1, int(round(tile * (1.0 - overlap))))
    offsets = list(range(0, max(length - tile, 0) + 1, step))
    last = length - tile
    if offsets[-1] != last:
        offsets.append(last)
    return offsets


def tile_grid(image_wh: tuple[int, int], tile: int, overlap: float) -> np.ndarray:
    """Grade de tiles cobrindo a imagem inteira.

    Args:
        image_wh: (largura, altura) em px.
        tile: lado do recorte quadrado em px. Se maior que a imagem, é limitado a ela.
        overlap: fração de sobreposição entre tiles vizinhos, em [0, 1).

    Returns:
        Array (K, 4) de inteiros, cada linha (x0, y0, x1, y1) em coordenadas da imagem.
    """
    if not 0.0 <= overlap < 1.0:
        raise ValueError(f"overlap deve estar em [0, 1), recebido {overlap}")
    # Sem esta guarda, tile <= 0 gera centenas de milhares de recortes de área zero ou
    # invertidos, sem exceção nenhuma — o processo apenas trava.
    if tile <= 0:
        raise ValueError(f"tile deve ser positivo, recebido {tile}")
    w, h = image_wh
    tw, th = min(tile, w), min(tile, h)
    boxes = [
        (x, y, x + tw, y + th)
        for y in _axis_offsets(h, th, overlap)
        for x in _axis_offsets(w, tw, overlap)
    ]
    return np.asarray(boxes, dtype=np.int32)


def crop_tiles(image: np.ndarray, grid: np.ndarray) -> list[np.ndarray]:
    """Recorta a imagem segundo a grade. Devolve views do array original (sem cópia)."""
    return [image[y0:y1, x0:x1] for x0, y0, x1, y1 in grid]


def stitch(tiles: list[np.ndarray], grid: np.ndarray, image_wh: tuple[int, int]) -> np.ndarray:
    """Recompõe a imagem a partir dos tiles. Existe para o teste de reconstrução.

    Regiões sobrepostas recebem o valor do último tile escrito; como os tiles vêm da mesma
    imagem, o resultado deve ser idêntico ao original pixel a pixel.
    """
    w, h = image_wh
    canvas = np.zeros((h, w, *tiles[0].shape[2:]), dtype=tiles[0].dtype)
    for tile, (x0, y0, x1, y1) in zip(tiles, grid):
        canvas[y0:y1, x0:x1] = tile
    return canvas


def coverage_map(grid: np.ndarray, image_wh: tuple[int, int]) -> np.ndarray:
    """Quantas vezes cada pixel é visto. Usado no teste de cobertura e nas figuras."""
    w, h = image_wh
    counts = np.zeros((h, w), dtype=np.int32)
    for x0, y0, x1, y1 in grid:
        counts[y0:y1, x0:x1] += 1
    return counts
