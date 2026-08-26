"""Testes do cálculo de AP e das métricas de contagem degeneradas.

Estes testes existem por um motivo específico e documentável: a suíte tinha 79 testes verdes
enquanto ``coco_metrics`` devolvia AP = −1,0 — a métrica principal do projeto — em toda
execução. Nenhum teste tocava em ``coco_metrics`` nem em ``_to_coco``, e o valor errado ia
direto para o CSV e para o relatório.

A lição não é "faltava um teste". É que a cobertura estava concentrada onde escrever teste é
fácil (geometria pura, funções sem dependência) e ausente exatamente na fronteira com a
biblioteca de terceiros, que é onde moram as armadilhas.
"""
from __future__ import annotations

import numpy as np
import pytest

from src.eval.counting import count_metrics
from src.eval.detection import _to_coco, coco_metrics

IMAGE_WH = (720, 1280)


def _grid_boxes(n: int, side: float = 28.0, seed: int = 0) -> np.ndarray:
    """Caixas do tamanho típico do dataset, espalhadas sem se tocar."""
    rng = np.random.default_rng(seed)
    cx = rng.uniform(side, IMAGE_WH[0] - side, n)
    cy = rng.uniform(side, IMAGE_WH[1] - side, n)
    return np.stack([cx - side / 2, cy - side / 2, cx + side / 2, cy + side / 2], 1).astype(
        np.float32
    )


@pytest.fixture(scope="module")
def dense_scene():
    """Cinco imagens com 120 maçãs cada — acima do padrão de 100 detecções do pycocotools."""
    gt = {f"img{i}.png": _grid_boxes(120) for i in range(5)}
    preds = {
        name: (boxes.copy(), np.linspace(0.99, 0.10, len(boxes)).astype(np.float32))
        for name, boxes in gt.items()
    }
    return gt, preds


# ------------------------------------------------------------------------------- AP

def test_ap_nao_e_menos_um(dense_scene):
    """A regressão que motivou este arquivo.

    Elevar ``params.maxDets`` de 100 para 300 faz ``pycocotools`` devolver fatia vazia em
    ``stats[0]``, porque essa é a única entrada de ``stats`` que não repassa ``maxDets`` — ela
    usa o 100 embutido na assinatura de ``_summarize``. O resultado é AP = −1,0, silencioso.
    """
    gt, preds = dense_scene
    metrics = coco_metrics(gt, preds, image_wh=IMAGE_WH)
    assert metrics["AP"] != -1.0
    assert 0.0 <= metrics["AP"] <= 1.0


def test_predicao_perfeita_tem_ap_quase_um(dense_scene):
    gt, preds = dense_scene
    metrics = coco_metrics(gt, preds, image_wh=IMAGE_WH)
    assert metrics["AP"] > 0.95
    assert metrics["AP50"] > 0.99


def test_faixa_de_area_vazia_devolve_nan_e_nao_menos_um(dense_scene):
    """0,0% do MinneApple é large. O −1 do pycocotools é sentinela, não métrica.

    Deixá-lo passar como número contaminava qualquer média, gráfico ou "AP por tamanho" no
    relatório, sem nenhum aviso.
    """
    gt, preds = dense_scene
    metrics = coco_metrics(gt, preds, image_wh=IMAGE_WH)
    assert np.isnan(metrics["AP_large"]), "caixas de 28 px nunca são 'large'"
    assert not np.isnan(metrics["AP_small"]), "e são todas 'small'"


def test_deteccoes_acima_de_cem_por_imagem_contam(dense_scene):
    """Com 120 maçãs por imagem, o padrão de 100 detecções descartaria as corretas.

    O recall com o limite elevado tem de ser maior — se não for, é porque o `maxDets` não
    está sendo respeitado no ponto em que a métrica é lida.
    """
    gt, preds = dense_scene
    assert coco_metrics(gt, preds, image_wh=IMAGE_WH)["AR_300"] > 0.9


def test_sem_predicoes_devolve_zeros(dense_scene):
    gt, _ = dense_scene
    metrics = coco_metrics(gt, {}, image_wh=IMAGE_WH)
    assert metrics["AP"] == 0.0 and metrics["AP50"] == 0.0


def test_imagem_so_com_predicao_entra_na_conta():
    """Falso positivo numa imagem sem anotação não pode sumir em silêncio.

    Tomar as imagens só das chaves do ground truth descartava todas as detecções de imagens
    ausentes ali — o que infla o AP de graça, porque os falsos positivos desaparecem.
    """
    gt = {"a.png": _grid_boxes(3)}
    acertos = (gt["a.png"].copy(), np.full(3, 0.90, np.float32))
    # Scores dos FP ACIMA dos acertos, de propósito: com scores empatados a ordenação estável
    # do pycocotools coloca a imagem 'a' primeiro e o AP não se move, o que tornaria o teste
    # verde por acidente em vez de por correção.
    preds = {"a.png": acertos, "b.png": (_grid_boxes(20, seed=1), np.full(20, 0.95, np.float32))}

    coco_gt, detections = _to_coco(gt, preds, IMAGE_WH)
    assert len(coco_gt["images"]) == 2
    assert len(detections) == 23, "as 20 detecções da imagem sem anotação têm de sobreviver"

    com_fp = coco_metrics(gt, preds, image_wh=IMAGE_WH)["AP"]
    sem_fp = coco_metrics(gt, {"a.png": acertos}, image_wh=IMAGE_WH)["AP"]
    assert sem_fp == pytest.approx(1.0)
    assert com_fp < 0.5, "20 falsos positivos com score alto têm de derrubar o AP"


def test_imagem_sem_predicao_ainda_conta_no_recall():
    """Anotação numa imagem que o detector ignorou continua no denominador."""
    gt = {"a.png": _grid_boxes(3), "b.png": _grid_boxes(3)}
    preds = {"a.png": (gt["a.png"].copy(), np.full(3, 0.9, np.float32))}
    assert coco_metrics(gt, preds, image_wh=IMAGE_WH)["AR_300"] < 0.55  # ~metade do recall


# --------------------------------------------------------------------------- contagem

def test_r2_pode_ser_negativo():
    """R² tem de conseguir dizer que o sistema é pior que prever a média.

    A versão anterior calculava o r² da regressão ``pred ~ gt``, que por construção de mínimos
    quadrados fica preso em [0, 1] — e portanto era incapaz de revelar o defeito.
    """
    gt = np.array([10, 30, 60, 100, 42, 55])
    assert count_metrics(2.0 * gt, gt)["R2"] < 0, "contar cada maçã duas vezes é pior que a média"
    assert count_metrics(gt, gt)["R2"] == pytest.approx(1.0)


def test_predicao_constante_nao_ganha_r2_perfeito():
    """0/0 no ajuste virava 1,0 — o valor mais bonito da tabela aparecia onde o sistema falhou.

    Na ponta alta da varredura de confiança o detector para de detectar e a predição vira
    constante zero. Ali, o R² antigo reportava 1,0000.
    """
    gt = np.array([10, 30, 60, 100, 42, 55])
    zeros = count_metrics(np.zeros_like(gt), gt)
    assert zeros["R2"] < 0
    assert np.isnan(zeros["fit_r2"]), "o ajuste é indefinido para predição constante"


def test_saturacao_aparece_na_inclinacao_e_no_r2():
    """Detector que satura em cena densa: ajuste perfeito, concordância ruim."""
    gt = np.array([10, 30, 60, 100, 42, 55])
    m = count_metrics(0.75 * gt, gt)
    assert m["fit_r2"] == pytest.approx(1.0), "a reta pred=0,75*gt ajusta perfeitamente"
    assert m["R2"] < 0.9, "mas a concordância com a verdade é ruim"
    assert m["slope"] == pytest.approx(0.75)
