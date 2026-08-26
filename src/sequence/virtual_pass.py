"""Passada virtual do robô: transforma uma imagem rotulada numa sequência com GT exato.

Por que isto existe. O bônus escolhido é a supressão de duplicatas entre quadros — o problema
real da Antoniq, em que a mesma framboesa reaparece em quadros consecutivos de um robô que
anda pelo corredor. Para medir isso é preciso saber, para cada detecção, se ela é uma fruta
nova ou uma já contada. Nenhum dataset público entrega essa correspondência.

A solução: varrer uma janela sobre uma imagem já rotulada, com deslocamento conhecido. Cada
posição da janela vira um "quadro"; como as anotações da imagem original são as mesmas em
todas as posições, sabemos exatamente quais frutas se repetem entre quadros. O ground truth
da contagem única é, por construção, exato.

Isto não substitui uma sequência real — não há paralaxe, nem mudança de iluminação, nem motion
blur. O que ele mede é precisamente a parte que interessa: se o rastreamento e a fusão entre
quadros convertem N observações repetidas em uma contagem única correta. As limitações estão
declaradas no relatório, e a demonstração qualitativa roda também nos vídeos reais do split de
teste oficial.

OpenCV e NumPy apenas.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Fração mínima da caixa que precisa estar dentro da janela para a fruta contar como visível.
# Espelha o comportamento de um detector: fruta cortada pela metade ainda é detectável;
# uma lasca de 10% não é.
MIN_VISIBLE_FRACTION = 0.5


@dataclass(frozen=True)
class Frame:
    """Um quadro da passada virtual."""

    index: int
    origin: tuple[int, int]   # (x, y) do canto da janela na imagem original
    image: np.ndarray         # recorte
    boxes: np.ndarray         # (N, 4) xyxy em coordenadas do QUADRO
    fruit_ids: np.ndarray     # (N,) identidade estável da fruta na imagem original


@dataclass(frozen=True)
class VirtualPass:
    frames: list[Frame]
    true_unique_count: int    # frutas distintas vistas em algum quadro
    naive_sum: int            # soma das contagens por quadro, se ninguém deduplicar
    step_px: int

    @property
    def overcount_factor(self) -> float:
        """Quanto a contagem ingênua infla. É o que a deduplicação precisa desfazer."""
        return self.naive_sum / max(self.true_unique_count, 1)


def visible_fraction(boxes: np.ndarray, window: tuple[int, int, int, int]) -> np.ndarray:
    """Fração da área de cada caixa que cai dentro da janela."""
    if len(boxes) == 0:
        return np.zeros(0, dtype=np.float32)
    boxes = np.asarray(boxes, dtype=np.float64)
    x0, y0, x1, y1 = window

    inter_w = np.clip(np.minimum(boxes[:, 2], x1) - np.maximum(boxes[:, 0], x0), 0, None)
    inter_h = np.clip(np.minimum(boxes[:, 3], y1) - np.maximum(boxes[:, 1], y0), 0, None)
    area = np.maximum((boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]), 1e-9)
    return ((inter_w * inter_h) / area).astype(np.float32)


def build_pass(
    image: np.ndarray,
    boxes: np.ndarray,
    window: tuple[int, int] = (480, 1280),
    n_frames: int = 5,
    step_px: int = 60,
    axis: str = "x",
    min_visible: float = MIN_VISIBLE_FRACTION,
) -> VirtualPass:
    """Gera a sequência varrendo uma janela pela imagem.

    Args:
        image: imagem original (H, W, 3).
        boxes: anotações da imagem original, xyxy.
        window: (largura, altura) da janela — o "sensor" do robô virtual.
        n_frames: quantas posições.
        step_px: deslocamento entre posições consecutivas. Com 60 px e janela de 480, dois
            quadros vizinhos compartilham 87% do campo de visão, que é a ordem de grandeza de
            um robô a 1 m/s amostrando alguns quadros por segundo.
        axis: "x" para varredura horizontal — a direção real de deslocamento do robô, já que
            a câmera aponta de lado para a fileira enquanto ele anda ao longo dela. Na imagem
            retrato do MinneApple isso corresponde ao eixo de 720 px.

    A varredura é truncada se a imagem acabar antes de ``n_frames`` posições.
    """
    h, w = image.shape[:2]
    win_w, win_h = window
    if win_w > w or win_h > h:
        raise ValueError(f"janela {window} nao cabe na imagem {(w, h)}")

    span = (w - win_w) if axis == "x" else (h - win_h)
    max_frames = span // step_px + 1
    n = int(min(n_frames, max_frames))
    if n < 2:
        raise ValueError(
            f"passada precisa de ao menos 2 quadros; a imagem {(w, h)} com janela {window} "
            f"e passo {step_px} comporta {max_frames}"
        )

    boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
    frames: list[Frame] = []
    seen: set[int] = set()
    naive_sum = 0

    for i in range(n):
        ox, oy = (i * step_px, 0) if axis == "x" else (0, i * step_px)
        win = (ox, oy, ox + win_w, oy + win_h)

        keep = np.flatnonzero(visible_fraction(boxes, win) >= min_visible)
        local = boxes[keep] - np.array([ox, oy, ox, oy], dtype=np.float32)
        # Recorta às bordas do quadro: é o que o detector enxergaria.
        local[:, [0, 2]] = np.clip(local[:, [0, 2]], 0, win_w)
        local[:, [1, 3]] = np.clip(local[:, [1, 3]], 0, win_h)

        frames.append(
            Frame(
                index=i,
                origin=(ox, oy),
                # Contíguo de propósito: um recorte com deslocamento em x é uma view não
                # contígua, e cv2.cvtColor / cv2.phaseCorrelate recusam esse layout.
                image=np.ascontiguousarray(image[oy : oy + win_h, ox : ox + win_w]),
                boxes=local,
                fruit_ids=keep.astype(np.int64),
            )
        )
        seen.update(keep.tolist())
        naive_sum += len(keep)

    return VirtualPass(
        frames=frames,
        true_unique_count=len(seen),
        naive_sum=naive_sum,
        step_px=step_px,
    )
