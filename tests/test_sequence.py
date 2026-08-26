"""Testes da passada virtual e da deduplicação entre quadros (bônus).

Tudo sintético e com ground truth exato, que é a razão de a passada virtual existir: sem ela
não há como saber, numa sequência real sem rótulo de identidade, se uma detecção é fruta nova
ou repetição.
"""
from __future__ import annotations

import cv2
import numpy as np
import pytest

from src.sequence.crossframe import (
    estimate_shift,
    estimate_shift_orb,
    track_sequence,
    warp_boxes,
)
from src.sequence.virtual_pass import build_pass, visible_fraction

IMAGE_WH = (720, 1280)   # retrato, como o MinneApple
# Janela de altura cheia varrida na horizontal: é a direção real de deslocamento do robô.
# Curso = 720 - 480 = 240 px, que com passo 60 comporta exatamente 5 posições.
WINDOW = (480, 1280)
STEP = 60
RADIUS = 22

CENTRES = [
    (180, 120), (420, 300), (200, 480), (560, 640), (300, 760),
    (150, 900), (480, 1020), (260, 1160), (620, 200), (100, 560),
]


@pytest.fixture(scope="module")
def scene() -> np.ndarray:
    """Fundo texturizado (folhagem) com discos claros (frutas).

    A textura importa: correlação de fase sobre fundo liso não tem pico definido, e o teste
    passaria por acidente.
    """
    rng = np.random.default_rng(0)
    img = rng.integers(0, 90, (IMAGE_WH[1], IMAGE_WH[0], 3), dtype=np.uint8)
    for cx, cy in CENTRES:
        cv2.circle(img, (cx, cy), RADIUS, (240, 240, 240), thickness=-1)
    return img


@pytest.fixture(scope="module")
def gt_boxes() -> np.ndarray:
    return np.array(
        [[cx - RADIUS, cy - RADIUS, cx + RADIUS, cy + RADIUS] for cx, cy in CENTRES],
        dtype=np.float32,
    )


# ------------------------------------------------------------------------ passada virtual

def test_fracao_visivel_de_caixa_inteiramente_dentro_e_um():
    boxes = np.array([[100.0, 100.0, 140.0, 140.0]])
    assert visible_fraction(boxes, (0, 0, 480, 1280))[0] == pytest.approx(1.0)


def test_fracao_visivel_de_caixa_cortada_ao_meio():
    boxes = np.array([[460.0, 100.0, 500.0, 140.0]])  # metade passa de x=480
    assert visible_fraction(boxes, (0, 0, 480, 1280))[0] == pytest.approx(0.5)


def test_fracao_visivel_de_caixa_fora_e_zero():
    boxes = np.array([[600.0, 100.0, 640.0, 140.0]])
    assert visible_fraction(boxes, (0, 0, 480, 1280))[0] == pytest.approx(0.0)


def test_passada_gera_os_quadros_pedidos(scene, gt_boxes):
    vp = build_pass(scene, gt_boxes, window=WINDOW, n_frames=5, step_px=STEP)
    assert len(vp.frames) == 5
    assert [f.origin for f in vp.frames] == [(i * STEP, 0) for i in range(5)]
    assert all(f.image.shape == (1280, 480, 3) for f in vp.frames)


def test_passada_e_truncada_quando_a_imagem_acaba(scene, gt_boxes):
    """720 - 480 = 240 px de curso; com passo 60 cabem 5 posições, não 20."""
    vp = build_pass(scene, gt_boxes, window=WINDOW, n_frames=20, step_px=STEP)
    assert len(vp.frames) == 5


def test_janela_maior_que_a_imagem_e_rejeitada(scene, gt_boxes):
    with pytest.raises(ValueError, match="nao cabe"):
        build_pass(scene, gt_boxes, window=(2000, 1280))


def test_caixas_do_quadro_ficam_dentro_dos_limites(scene, gt_boxes):
    vp = build_pass(scene, gt_boxes, window=WINDOW, n_frames=5, step_px=STEP)
    for frame in vp.frames:
        assert (frame.boxes[:, [0, 2]] >= 0).all()
        assert (frame.boxes[:, [0, 2]] <= WINDOW[0]).all()
        assert (frame.boxes[:, [1, 3]] >= 0).all()
        assert (frame.boxes[:, [1, 3]] <= WINDOW[1]).all()


def test_caixa_nao_recortada_volta_exatamente_para_a_anotacao_original(scene, gt_boxes):
    """A transformação quadro -> imagem original tem de ser reversível.

    Só vale para caixas que não encostam na borda do quadro: as que encostam foram recortadas
    de propósito, porque é isso que o detector enxergaria.
    """
    vp = build_pass(scene, gt_boxes, window=WINDOW, n_frames=5, step_px=STEP)
    checked = 0
    for frame in vp.frames:
        ox, oy = frame.origin
        offset = np.array([ox, oy, ox, oy], dtype=np.float32)
        for local, fruit in zip(frame.boxes, frame.fruit_ids):
            touches_edge = (
                local[0] <= 0 or local[1] <= 0
                or local[2] >= WINDOW[0] or local[3] >= WINDOW[1]
            )
            if touches_edge:
                continue
            assert (local + offset).tolist() == pytest.approx(gt_boxes[fruit].tolist())
            checked += 1
    assert checked > 20, "o teste precisa cobrir varias caixas para valer alguma coisa"


def test_fruta_na_borda_do_quadro_e_recortada(scene, gt_boxes):
    """Fruta parcialmente fora da janela aparece cortada, como um detector a veria."""
    vp = build_pass(scene, gt_boxes, window=WINDOW, n_frames=5, step_px=STEP)
    clipped = [
        local
        for frame in vp.frames
        for local in frame.boxes
        if local[0] <= 0 or local[2] >= WINDOW[0]
    ]
    assert clipped, "a geometria escolhida deveria produzir ao menos uma caixa cortada"
    for box in clipped:
        assert (box[2] - box[0]) < 2 * RADIUS  # mais estreita que a fruta inteira


def test_contagem_ingenua_supera_a_contagem_unica(scene, gt_boxes):
    """A premissa do bônus: sem deduplicar, a mesma fruta é contada em vários quadros."""
    vp = build_pass(scene, gt_boxes, window=WINDOW, n_frames=5, step_px=STEP)
    assert vp.naive_sum > vp.true_unique_count
    assert vp.overcount_factor > 1.5


# ----------------------------------------------------------------- estimativa de movimento

@pytest.mark.parametrize("estimator", [estimate_shift, estimate_shift_orb])
def test_convencao_de_sinal_do_deslocamento(scene, estimator):
    """Fixa a convenção: p no quadro anterior aparece em p + (dx, dy) no atual.

    Com o robô avançando +STEP px na cena, o conteúdo recua -STEP px dentro do quadro.
    Um erro de sinal aqui transportaria as caixas para o lado oposto, nenhuma trilha casaria,
    e a deduplicação silenciosamente não faria nada — sem levantar exceção.
    """
    prev = np.ascontiguousarray(scene[:, 0 : WINDOW[0]])
    curr = np.ascontiguousarray(scene[:, STEP : STEP + WINDOW[0]])
    dx, dy = estimator(prev, curr)
    assert dx == pytest.approx(-STEP, abs=1.0)
    assert dy == pytest.approx(0.0, abs=1.0)


def test_quadros_de_tamanhos_diferentes_sao_rejeitados(scene):
    with pytest.raises(ValueError, match="tamanhos diferentes"):
        estimate_shift(scene[:, :480], scene[:, :400])


def test_warp_desloca_as_caixas():
    boxes = np.array([[100.0, 100.0, 140.0, 140.0]])
    assert warp_boxes(boxes, (-60.0, 0.0))[0].tolist() == [40.0, 100.0, 80.0, 140.0]
    assert warp_boxes(np.zeros((0, 4)), (10.0, 5.0)).shape == (0, 4)


# ------------------------------------------------------------------------ rastreamento

def test_deduplicacao_recupera_a_contagem_unica_exata(scene, gt_boxes):
    """O teste central do bônus.

    Alimentando o rastreador com detecções perfeitas (as próprias anotações), a contagem de
    trilhas tem de bater exatamente com o número de frutas distintas vistas na passada.
    """
    vp = build_pass(scene, gt_boxes, window=WINDOW, n_frames=5, step_px=STEP)
    result = track_sequence(
        [f.image for f in vp.frames], [f.boxes for f in vp.frames], method="phase_correlate"
    )
    assert result.naive_sum == vp.naive_sum
    assert result.unique_count == vp.true_unique_count
    assert result.suppression_rate > 0.4


def test_sem_deduplicacao_a_contagem_estoura(scene, gt_boxes):
    """Contraprova: casar com IoU inatingível degenera para 'cada detecção é uma fruta'."""
    vp = build_pass(scene, gt_boxes, window=WINDOW, n_frames=5, step_px=STEP)
    # 1,5 e inatingivel: o IoU satura em 1,0. Um limiar de 0,999 NAO serviria -- a estimativa
    # de deslocamento acerta com erro de 0,006 px, o que ainda deixa o IoU em ~0,9997.
    result = track_sequence(
        [f.image for f in vp.frames], [f.boxes for f in vp.frames], match_iou=1.5
    )
    assert result.unique_count == vp.naive_sum
    assert result.suppression_rate == pytest.approx(0.0)


def test_camera_parada_nao_cria_frutas_novas(scene, gt_boxes):
    """Robô parado: cinco quadros idênticos devem dar a contagem de um quadro só."""
    frame = np.ascontiguousarray(scene[:, 0 : WINDOW[0]])
    boxes = build_pass(scene, gt_boxes, window=WINDOW, n_frames=2, step_px=STEP).frames[0].boxes
    result = track_sequence([frame] * 5, [boxes] * 5)
    assert result.unique_count == len(boxes)


def test_oclusao_momentanea_nao_cria_fruta_nova(scene, gt_boxes):
    """Uma fruta some por um quadro (folha na frente) e volta: continua sendo a mesma trilha."""
    vp = build_pass(scene, gt_boxes, window=WINDOW, n_frames=5, step_px=STEP)
    images = [f.image for f in vp.frames]
    detections = [f.boxes.copy() for f in vp.frames]
    detections[2] = detections[2][1:]  # a primeira fruta desaparece no quadro do meio

    with_memory = track_sequence(images, detections, max_age=2)
    without_memory = track_sequence(images, detections, max_age=0)
    assert with_memory.unique_count == vp.true_unique_count
    assert without_memory.unique_count > with_memory.unique_count


def test_metodo_desconhecido_e_rejeitado(scene, gt_boxes):
    with pytest.raises(ValueError, match="method deve ser"):
        track_sequence([scene], [np.zeros((0, 4))], method="kalman")


def test_numero_de_quadros_e_de_deteccoes_precisa_bater(scene):
    with pytest.raises(ValueError, match="listas de deteccoes"):
        track_sequence([scene, scene], [np.zeros((0, 4))])
