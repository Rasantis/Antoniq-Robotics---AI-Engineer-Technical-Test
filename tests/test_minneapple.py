"""Testes da leitura de máscaras de instância.

Máscaras sintéticas com geometria conhecida, para que o resultado esperado seja calculável à
mão. O parsing roda uma vez sobre todo o dataset e alimenta a análise de erro inteira: um bug
aqui contamina tudo o que vem depois sem levantar exceção.
"""
from __future__ import annotations

import numpy as np
import pytest

from src.data.minneapple import (
    distance_to_border,
    instances_from_mask,
    local_density,
    solidity,
)

IMAGE_WH = (1280, 720)


def _disc(mask: np.ndarray, cx: int, cy: int, r: int, value: int) -> None:
    """Desenha um disco preenchido com o ID `value`. Aproxima o formato de uma maçã."""
    ys, xs = np.ogrid[: mask.shape[0], : mask.shape[1]]
    mask[(xs - cx) ** 2 + (ys - cy) ** 2 <= r**2] = value


def test_extrai_uma_instancia_por_id():
    mask = np.zeros((200, 200), dtype=np.uint16)
    _disc(mask, 50, 50, 10, value=7)
    _disc(mask, 150, 120, 15, value=3)

    inst = instances_from_mask(mask)
    assert len(inst) == 2
    # Ordenadas por ID crescente, resultado de np.unique.
    assert inst.instance_ids.tolist() == [3, 7]


def test_caixa_tem_borda_exclusiva():
    """Um quadrado de 10x10 px deve produzir largura e altura exatamente 10."""
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[20:30, 40:50] = 5

    inst = instances_from_mask(mask)
    assert inst.boxes[0].tolist() == [40.0, 20.0, 50.0, 30.0]
    assert inst.mask_areas.tolist() == [100]


def test_area_da_mascara_conta_pixels_marcados_nao_a_caixa():
    mask = np.zeros((100, 100), dtype=np.uint8)
    _disc(mask, 50, 50, 20, value=1)

    inst = instances_from_mask(mask)
    box_area = float((inst.boxes[0, 2] - inst.boxes[0, 0]) * (inst.boxes[0, 3] - inst.boxes[0, 1]))
    assert inst.mask_areas[0] < box_area  # disco não preenche o retângulo


def test_solidez_de_um_disco_fica_perto_de_pi_sobre_4():
    """Um disco inscrito preenche pi/4 = 0,785 da caixa. É a referência do proxy de oclusão."""
    mask = np.zeros((200, 200), dtype=np.uint8)
    _disc(mask, 100, 100, 40, value=1)

    inst = instances_from_mask(mask)
    assert solidity(inst.boxes, inst.mask_areas)[0] == pytest.approx(np.pi / 4, abs=0.02)


def test_solidez_cai_quando_a_fruta_e_ocluida():
    """Metade do disco escondida: mesma caixa, metade da área, solidez pela metade."""
    full = np.zeros((200, 200), dtype=np.uint8)
    _disc(full, 100, 100, 40, value=1)
    half = full.copy()
    half[:, 100:] = 0  # uma folha cobre o lado direito
    half[95:105, 100:141] = 1  # mantém a extensão da caixa

    s_full = solidity(*_boxes_and_areas(full))[0]
    s_half = solidity(*_boxes_and_areas(half))[0]
    assert s_half < 0.6 < s_full


def _boxes_and_areas(mask):
    inst = instances_from_mask(mask)
    return inst.boxes, inst.mask_areas


def test_instancias_minusculas_sao_descartadas():
    """Resíduos de 1 px de máscara não são fruta anotável."""
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[10, 10] = 1          # 1x1 -> descartado
    mask[50:60, 50:60] = 2    # 10x10 -> mantido

    inst = instances_from_mask(mask, min_side=2)
    assert inst.instance_ids.tolist() == [2]


def test_mascara_vazia_devolve_zero_instancias():
    inst = instances_from_mask(np.zeros((100, 100), dtype=np.uint8))
    assert len(inst) == 0
    assert inst.boxes.shape == (0, 4)
    assert solidity(inst.boxes, inst.mask_areas).shape == (0,)


def test_mascara_rgb_usa_o_primeiro_canal():
    """Alguns PNG do conjunto vêm com os IDs replicados nos três canais."""
    gray = np.zeros((100, 100), dtype=np.uint8)
    gray[10:20, 10:20] = 4
    rgb = np.stack([gray] * 3, axis=-1)
    assert np.array_equal(
        instances_from_mask(rgb[..., 0]).boxes, instances_from_mask(gray).boxes
    )


def test_densidade_local_conta_vizinhas_dentro_do_raio():
    #  duas frutas próximas (centros a 50 px) e uma isolada bem longe
    boxes = np.array(
        [
            [100.0, 100.0, 140.0, 140.0],
            [150.0, 100.0, 190.0, 140.0],
            [900.0, 600.0, 940.0, 640.0],
        ]
    )
    assert local_density(boxes, radius=100.0).tolist() == [1, 1, 0]


def test_distancia_a_borda_usa_a_borda_mais_proxima():
    boxes = np.array([[0.0, 300.0, 40.0, 340.0], [620.0, 340.0, 660.0, 380.0]])
    d = distance_to_border(boxes, IMAGE_WH)
    assert d[0] == pytest.approx(20.0)   # colada na borda esquerda
    assert d[1] == pytest.approx(360.0)  # centro da imagem
