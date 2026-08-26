"""Testes das métricas de detecção e de contagem.

O casamento predição<->anotação alimenta a análise de modos de erro inteira: se ele atribuir
errado, o ranking de modos de erro fica errado e ninguém percebe. Daí a cobertura detalhada.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.eval.counting import (
    acceptance_check,
    count_metrics,
    per_group_metrics,
    row_level_metrics,
)
from src.eval.detection import match_predictions, operating_point_stats


def _box(x, y, s=40.0):
    return [x, y, x + s, y + s]


# ------------------------------------------------------------------------ casamento

def test_predicao_perfeita_e_toda_tp():
    gt = np.array([_box(0, 0), _box(200, 200), _box(400, 400)])
    m = match_predictions(gt.copy(), np.array([0.9, 0.8, 0.7]), gt)
    assert m.counts() == (3, 0, 0)
    assert m.pred_to_gt.tolist() == [0, 1, 2]


def test_sem_sobreposicao_gera_fp_e_fn():
    gt = np.array([_box(0, 0)])
    pred = np.array([_box(500, 500)])
    m = match_predictions(pred, np.array([0.9]), gt)
    assert m.counts() == (0, 1, 1)


def test_uma_anotacao_so_pode_ser_casada_uma_vez():
    """Duas detecções na mesma fruta: uma vira TP, a outra vira FP.

    É o que acontece quando a fusão entre tiles falha — e é por isso que uma duplicata não
    resolvida custa duas vezes: infla a contagem e derruba a precisão.
    """
    gt = np.array([_box(100, 100)])
    pred = np.array([_box(100, 100), _box(103, 103)])
    m = match_predictions(pred, np.array([0.9, 0.85]), gt)
    assert m.counts() == (1, 1, 0)
    assert m.pred_to_gt.tolist() == [0, -1]


def test_casamento_e_guloso_por_confianca_nao_por_iou():
    """Quem tem mais confiança escolhe primeiro, mesmo que outra predição encaixe melhor.

    É o critério do COCO, e importa aqui: a caixa de maior score fica com a anotação, e a
    que encaixava perfeitamente vira falso positivo. Um matcher que ordenasse por IoU daria
    um recall mais bonito e não corresponderia ao que o sistema entrega.
    """
    gt = np.array([_box(100, 100)])
    # A 2a encaixa exato (IoU 1,00); a 1a fica deslocada mas ainda acima do limiar (IoU 0,57).
    pred = np.array([_box(106, 106), _box(100, 100)])
    m = match_predictions(pred, np.array([0.90, 0.50]), gt)

    assert m.pred_to_gt.tolist() == [0, -1]  # a de maior score levou, apesar do encaixe pior
    assert m.counts() == (1, 1, 0)


def test_limiar_de_iou_e_respeitado():
    gt = np.array([_box(100, 100)])
    pred = np.array([_box(120, 100)])  # deslocamento de 20 px numa caixa de 40 -> IoU ~0,33
    assert match_predictions(pred, np.array([0.9]), gt, iou_threshold=0.5).counts() == (0, 1, 1)
    assert match_predictions(pred, np.array([0.9]), gt, iou_threshold=0.3).counts() == (1, 0, 0)


def test_entradas_vazias():
    gt = np.array([_box(0, 0)])
    assert match_predictions(np.zeros((0, 4)), np.zeros(0), gt).counts() == (0, 0, 1)
    assert match_predictions(gt, np.array([0.9]), np.zeros((0, 4))).counts() == (0, 1, 0)


def test_ponto_de_operacao_agrega_varias_imagens():
    gt = np.array([_box(0, 0), _box(200, 200)])
    m1 = match_predictions(gt.copy(), np.array([0.9, 0.8]), gt)          # 2 TP
    m2 = match_predictions(np.array([_box(900, 600)]), np.array([0.9]), gt)  # 1 FP, 2 FN
    stats = operating_point_stats([m1, m2])
    assert (stats["tp"], stats["fp"], stats["fn"]) == (2, 1, 2)
    assert stats["precision"] == pytest.approx(2 / 3)
    assert stats["recall"] == pytest.approx(0.5)


# ------------------------------------------------------------------------- contagem

def test_contagem_perfeita():
    counts = np.array([10, 20, 30, 40])
    m = count_metrics(counts, counts)
    assert m["MAE"] == 0 and m["bias"] == 0 and m["MAPE"] == 0
    assert m["slope"] == pytest.approx(1.0) and m["R2"] == pytest.approx(1.0)


def test_vies_constante_aparece_com_sinal():
    gt = np.array([10, 20, 30, 40])
    m = count_metrics(gt + 3, gt)
    assert m["bias"] == pytest.approx(3.0)
    assert m["MAE"] == pytest.approx(3.0)  # sem erro aleatório, MAE == |viés|
    assert m["bias_rel"] == pytest.approx(12 / 100)


def test_erro_aleatorio_tem_mae_alta_e_vies_nulo():
    """A distinção que sustenta o critério de aceitação do relatório.

    Dois sistemas com a MESMA MAE: um sem viés, outro totalmente enviesado. Ao somar a
    fileira, o primeiro acerta e o segundo erra em 10%. MAE sozinha não separa os dois.
    """
    gt = np.array([50, 50, 50, 50])
    unbiased = count_metrics(np.array([55, 45, 55, 45]), gt)
    biased = count_metrics(np.array([55, 55, 55, 55]), gt)

    assert unbiased["MAE"] == biased["MAE"] == 5.0
    assert unbiased["bias_rel"] == pytest.approx(0.0)
    assert biased["bias_rel"] == pytest.approx(0.10)


def test_contagem_certa_com_deteccao_errada():
    """Um falso positivo e um falso negativo na mesma imagem dão erro de contagem zero.

    É a razão de o enunciado pedir métricas de contagem além do mAP: aqui a contagem está
    perfeita e a detecção está errada nas duas pontas.
    """
    gt_boxes = np.array([_box(100, 100), _box(300, 300)])
    pred_boxes = np.array([_box(100, 100), _box(900, 600)])  # acerta uma, inventa outra
    m = match_predictions(pred_boxes, np.array([0.9, 0.9]), gt_boxes)

    assert m.counts() == (1, 1, 1)                     # detecção: errada dos dois lados
    assert count_metrics([len(pred_boxes)], [len(gt_boxes)])["MAE"] == 0.0  # contagem: exata


def test_inclinacao_detecta_saturacao_em_cenas_cheias():
    """Detector que satura em imagens densas tem inclinação bem abaixo de 1."""
    gt = np.array([10, 30, 60, 100])
    saturated = np.array([10, 27, 48, 70])
    assert count_metrics(saturated, gt)["slope"] < 0.8


def test_metricas_por_fileira():
    df = pd.DataFrame(
        {
            "session": ["a", "a", "b", "b"],
            "pred": [11, 9, 24, 26],  # fileira a: 20 vs 20 | fileira b: 50 vs 40
            "gt": [10, 10, 20, 20],
        }
    )
    m = row_level_metrics(df)
    assert m["n_rows"] == 2
    assert m["row_worst_abs_rel"] == pytest.approx(0.25)  # fileira b erra 25%
    assert m["row_MAPE"] == pytest.approx(0.125)         # media de 0% e 25%


def test_criterio_de_aceitacao_reprova_por_pior_fileira():
    """Média boa não salva: o critério trava também o pior caso."""
    metrics = {"bias_rel": 0.01}
    row = {"row_MAPE": 0.06, "row_worst_abs_rel": 0.22}
    result = acceptance_check(metrics, row)
    assert result["row_MAPE<=10%"] is True
    assert result["worst_row<=15%"] is False
    assert result["GO"] is False


def test_criterio_de_aceitacao_aprova_quando_tudo_passa():
    result = acceptance_check(
        {"bias_rel": -0.02}, {"row_MAPE": 0.07, "row_worst_abs_rel": 0.11}
    )
    assert result["GO"] is True


def test_metricas_por_grupo_devolvem_uma_linha_por_sessao():
    df = pd.DataFrame(
        {"session": ["a", "a", "b"], "pred": [10, 12, 30], "gt": [10, 10, 25]}
    )
    out = per_group_metrics(df)
    assert len(out) == 2
    assert set(out.session) == {"a", "b"}


def test_paired_wins_conta_direcao_certa_por_metrica():
    """A MAE é a única em que MENOR vence, e tratá-la junto com as outras já inverteu número
    neste projeto. Aqui o IoS é melhor em precisão e PIOR em MAE, de propósito."""
    import pandas as pd

    from src.eval.ablation import paired_wins

    def linha(metrica, conf, precision, mae):
        return {"arm": "C", "policy": "nmm", "threshold": 0.5, "drop_truncated": True,
                "conf": conf, "fold": 0, "metric": metrica,
                "precision": precision, "recall": 0.5, "f1": 0.5, "MAE": mae}

    tabela = pd.DataFrame([
        linha("ios", 0.1, 0.90, 30.0), linha("iou", 0.1, 0.70, 20.0),
        linha("ios", 0.2, 0.80, 25.0), linha("iou", 0.2, 0.60, 15.0),
    ])
    r = paired_wins(tabela).set_index("metrica")
    assert r.loc["precision", "pairs"] == 2
    assert r.loc["precision", "wins"] == 2          # IoS mais preciso nas duas
    assert r.loc["MAE", "wins"] == 0                # e pior na contagem nas duas
    assert r.loc["recall", "wins"] == 0             # empate nao conta como vitoria


def test_paired_wins_ignora_configuracao_sem_par():
    """Uma linha de IoS sem o IoU correspondente não pode entrar na conta."""
    import pandas as pd

    from src.eval.ablation import paired_wins

    base = {"arm": "C", "policy": "nmm", "threshold": 0.5, "drop_truncated": True,
            "fold": 0, "precision": 0.9, "recall": 0.5, "f1": 0.5, "MAE": 10.0}
    tabela = pd.DataFrame([
        {**base, "conf": 0.1, "metric": "ios"},
        {**base, "conf": 0.1, "metric": "iou", "precision": 0.5},
        {**base, "conf": 0.9, "metric": "ios"},          # sem par
    ])
    r = paired_wins(tabela).set_index("metrica")
    assert r.loc["precision", "pairs"] == 1


def test_checkpoint_retomavel_recusa_run_ja_finalizado():
    """`patience` faz um run parar em 80 de 120 épocas — e ele está COMPLETO, não interrompido.

    Testar "épocas feitas < épocas pedidas" mandava esse caso para o `resume`, e o Ultralytics,
    sem estado de otimizador, começava um treino novo com os padrões dele (dataset coco8, saída
    fora do projeto). O que decide é o conteúdo do checkpoint, não a contagem de épocas.
    """
    import importlib.util
    import sys
    from pathlib import Path

    caminho = Path(__file__).resolve().parents[1] / "scripts" / "02_train_cv.py"
    spec = importlib.util.spec_from_file_location("treino_cv", caminho)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["treino_cv"] = mod
    spec.loader.exec_module(mod)

    # como o Ultralytics grava um run terminado: sem otimizador, epoch = -1
    assert not mod.checkpoint_retomavel({"epoch": -1, "optimizer": None})
    assert not mod.checkpoint_retomavel({"epoch": -1, "optimizer": {"state": {}}})
    assert not mod.checkpoint_retomavel({"epoch": 40, "optimizer": None})
    assert not mod.checkpoint_retomavel({})
    # e como fica um run de fato interrompido no meio
    assert mod.checkpoint_retomavel({"epoch": 40, "optimizer": {"state": {}}})
