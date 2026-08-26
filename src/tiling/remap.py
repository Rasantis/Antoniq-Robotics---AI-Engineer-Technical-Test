"""Transformação de coordenadas tile <-> imagem, e detecção de caixas truncadas na borda."""
from __future__ import annotations

import numpy as np


def to_global(boxes: np.ndarray, tile: np.ndarray) -> np.ndarray:
    """Converte caixas xyxy locais ao tile para coordenadas da imagem."""
    boxes = np.asarray(boxes, dtype=np.float32)
    if len(boxes) == 0:
        return boxes.reshape(0, 4).astype(np.float32)
    shift = np.array([tile[0], tile[1], tile[0], tile[1]], dtype=np.float32)
    return np.asarray(boxes, dtype=np.float32) + shift


def to_local(boxes: np.ndarray, tile: np.ndarray) -> np.ndarray:
    """Inverso de :func:`to_global`."""
    boxes = np.asarray(boxes, dtype=np.float32)
    if len(boxes) == 0:
        return boxes.reshape(0, 4).astype(np.float32)
    shift = np.array([tile[0], tile[1], tile[0], tile[1]], dtype=np.float32)
    return np.asarray(boxes, dtype=np.float32) - shift


def is_truncated(
    boxes_global: np.ndarray,
    tile: np.ndarray,
    image_wh: tuple[int, int],
    margin: float = 2.0,
) -> np.ndarray:
    """Marca caixas coladas numa aresta interna do tile.

    Uma aresta interna é uma que não coincide com a borda da imagem. Uma caixa colada nela
    quase certamente é uma fruta cortada pelo recorte: o tile vizinho enxerga a fruta inteira,
    então esta versão truncada só pode virar duplicata ou caixa com extensão errada.

    Caixas coladas na borda real da imagem NÃO são marcadas — ali a fruta está genuinamente
    cortada e não existe tile vizinho para vê-la melhor.
    """
    if len(boxes_global) == 0:
        return np.zeros(0, dtype=bool)
    boxes = np.asarray(boxes_global, dtype=np.float32)
    x0, y0, x1, y1 = (float(v) for v in tile)
    w, h = image_wh

    touches = np.zeros(len(boxes), dtype=bool)
    if x0 > 0:
        touches |= boxes[:, 0] <= x0 + margin
    if y0 > 0:
        touches |= boxes[:, 1] <= y0 + margin
    if x1 < w:
        touches |= boxes[:, 2] >= x1 - margin
    if y1 < h:
        touches |= boxes[:, 3] >= y1 - margin
    return touches
