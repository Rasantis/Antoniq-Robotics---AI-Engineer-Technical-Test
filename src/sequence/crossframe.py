"""Supressão de duplicatas entre quadros consecutivos (bônus).

O problema: um robô andando pelo corredor fotografa a mesma fruta em vários quadros. Somar as
contagens por quadro superestima a colheita por um fator que depende só da velocidade e da
taxa de amostragem — nada a ver com a qualidade do detector. É um erro puramente sistemático,
exatamente o tipo que o critério de aceitação do relatório trava.

A abordagem aqui é de baixo custo, apropriada para um Jetson: estimar o deslocamento entre
quadros consecutivos diretamente das imagens, transportar as caixas do quadro anterior para o
atual, e casar por sobreposição. Cada fruta vira uma trilha, e a contagem é o número de
trilhas — não o número de detecções.

Duas formas de estimar o movimento, ambas em OpenCV:

    phase_correlate   translação pura, via correlação de fase no domínio da frequência.
                      Barato (uma FFT por quadro) e robusto a mudança de brilho. É o certo
                      quando a câmera anda paralela à fileira, que é o caso do robô.

    orb_homography    ORB + RANSAC, para quando há paralaxe ou rotação. Mais caro e mais
                      frágil em cena repetitiva (folhagem gera muitos casamentos ruins).

Convenção de sinal, fixada em teste: ``estimate_shift(prev, curr)`` devolve ``(dx, dy)`` tal
que um ponto em ``p`` no quadro anterior aparece em ``p + (dx, dy)`` no quadro atual.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from src.tiling.merge import iou_matrix


def _gray32(image: np.ndarray) -> np.ndarray:
    """Cinza contíguo em float32, que é o que phaseCorrelate e ORB aceitam.

    RGB2GRAY, não BGR2GRAY: as imagens chegam em RGB (via PIL). A troca não afetaria o
    deslocamento estimado, porque o erro seria idêntico nos dois quadros, mas usar a
    conversão errada de propósito é o tipo de coisa que morde quando alguém reaproveitar
    a função para outra coisa.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if image.ndim == 3 else image
    return np.ascontiguousarray(gray, dtype=np.float32)


def estimate_shift(prev: np.ndarray, curr: np.ndarray) -> tuple[float, float]:
    """Translação entre dois quadros, por correlação de fase.

    Uma janela de Hanning suprime o artefato de borda que a FFT introduz ao tratar a imagem
    como periódica — sem ela, o pico da correlação fica enviesado em cena com estrutura forte
    nas bordas, que é o caso de uma fileira de plantas.
    """
    a, b = _gray32(prev), _gray32(curr)
    if a.shape != b.shape:
        raise ValueError(f"quadros de tamanhos diferentes: {a.shape} vs {b.shape}")
    window = cv2.createHanningWindow((a.shape[1], a.shape[0]), cv2.CV_32F)
    (dx, dy), _response = cv2.phaseCorrelate(a, b, window)
    return float(dx), float(dy)


def estimate_shift_orb(
    prev: np.ndarray, curr: np.ndarray, max_features: int = 1500
) -> tuple[float, float]:
    """Translação pela mediana do deslocamento de pontos ORB casados.

    Alternativa para cenas com paralaxe. A mediana é usada no lugar de uma homografia completa
    porque, num corredor, o movimento dominante é translação; estimar 8 parâmetros a partir de
    folhagem repetitiva costuma piorar o resultado em vez de melhorar.
    """
    a, b = _gray32(prev).astype(np.uint8), _gray32(curr).astype(np.uint8)
    orb = cv2.ORB_create(nfeatures=max_features)
    kp_a, des_a = orb.detectAndCompute(a, None)
    kp_b, des_b = orb.detectAndCompute(b, None)
    if des_a is None or des_b is None or len(kp_a) < 4 or len(kp_b) < 4:
        return 0.0, 0.0

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = matcher.match(des_a, des_b)
    if len(matches) < 4:
        return 0.0, 0.0

    deltas = np.array(
        [np.subtract(kp_b[m.trainIdx].pt, kp_a[m.queryIdx].pt) for m in matches]
    )
    return float(np.median(deltas[:, 0])), float(np.median(deltas[:, 1]))


_ESTIMATORS = {"phase_correlate": estimate_shift, "orb_homography": estimate_shift_orb}


def warp_boxes(boxes: np.ndarray, shift: tuple[float, float]) -> np.ndarray:
    """Transporta caixas do quadro anterior para o sistema de coordenadas do atual."""
    if len(boxes) == 0:
        return np.zeros((0, 4), np.float32)
    dx, dy = shift
    return np.asarray(boxes, np.float32) + np.array([dx, dy, dx, dy], np.float32)


@dataclass
class TrackingResult:
    """Saída do rastreamento sobre uma sequência."""

    track_ids: list[np.ndarray]     # id de trilha por detecção, quadro a quadro
    unique_count: int               # contagem deduplicada -> a resposta do sistema
    naive_sum: int                  # soma por quadro, sem deduplicar
    shifts: list[tuple[float, float]] = field(default_factory=list)

    @property
    def suppression_rate(self) -> float:
        """Fração das detecções que era repetição. Zero significa que nada foi deduplicado."""
        return 1.0 - self.unique_count / max(self.naive_sum, 1)


def track_sequence(
    images: list[np.ndarray],
    detections: list[np.ndarray],
    method: str = "phase_correlate",
    match_iou: float = 0.3,
    max_age: int = 2,
) -> TrackingResult:
    """Rastreia detecções ao longo de uma sequência e devolve a contagem única.

    Args:
        images: quadros na ordem temporal.
        detections: caixas xyxy por quadro, em coordenadas do quadro.
        match_iou: sobreposição mínima, após o transporte, para considerar a mesma fruta.
            O limiar é mais frouxo que o de avaliação (0,5) de propósito: o erro de estimativa
            do deslocamento se soma ao erro de localização do detector.
        max_age: por quantos quadros uma trilha sobrevive sem ser vista. Tolera oclusão
            momentânea por folha, que é comum e não deveria criar uma fruta nova.

    Returns:
        ``TrackingResult`` com ``unique_count`` = número de trilhas distintas.
    """
    if method not in _ESTIMATORS:
        raise ValueError(f"method deve ser um de {sorted(_ESTIMATORS)}, recebido {method!r}")
    if len(images) != len(detections):
        raise ValueError(f"{len(images)} quadros e {len(detections)} listas de deteccoes")

    estimator = _ESTIMATORS[method]
    track_boxes: list[np.ndarray] = []   # última caixa conhecida de cada trilha
    track_age: list[int] = []            # quadros desde a última vez que foi vista
    assignments: list[np.ndarray] = []
    shifts: list[tuple[float, float]] = []
    naive_sum = 0

    for i, boxes in enumerate(detections):
        boxes = np.asarray(boxes, np.float32).reshape(-1, 4)
        naive_sum += len(boxes)

        if i == 0:
            shift = (0.0, 0.0)
        else:
            shift = estimator(images[i - 1], images[i])
            shifts.append(shift)
            # Todas as trilhas vivas se movem junto com a cena.
            track_boxes = [warp_boxes(b.reshape(1, 4), shift)[0] for b in track_boxes]
            track_age = [a + 1 for a in track_age]

        ids = np.full(len(boxes), -1, dtype=np.int64)
        if track_boxes and len(boxes):
            alive = [t for t, age in enumerate(track_age) if age <= max_age]
            if alive:
                ious = iou_matrix(boxes, np.stack([track_boxes[t] for t in alive]))
                taken: set[int] = set()
                # Casamento guloso pelo melhor IoU. Com fruta pequena e bem separada, a
                # diferença para o ótimo húngaro é desprezível e o custo é muito menor.
                for det_i, trk_j in sorted(
                    ((d, t) for d in range(len(boxes)) for t in range(len(alive))),
                    key=lambda p: -ious[p],
                ):
                    if ious[det_i, trk_j] < match_iou:
                        break
                    if ids[det_i] >= 0 or trk_j in taken:
                        continue
                    ids[det_i] = alive[trk_j]
                    taken.add(trk_j)

        for det_i in range(len(boxes)):
            if ids[det_i] < 0:
                track_boxes.append(boxes[det_i])
                track_age.append(0)
                ids[det_i] = len(track_boxes) - 1
            else:
                track_boxes[ids[det_i]] = boxes[det_i]
                track_age[ids[det_i]] = 0

        assignments.append(ids)

    return TrackingResult(
        track_ids=assignments,
        unique_count=len(track_boxes),
        naive_sum=naive_sum,
        shifts=shifts,
    )
