"""Métricas de contagem — as que o produto realmente entrega.

A Tarefa 3 pede explicitamente para caracterizar a contagem, "não só mAP de detecção". O
motivo é que as duas coisas se descolam: numa contagem, um falso positivo e um falso negativo
na mesma imagem se cancelam e o erro final é zero, enquanto o mAP pune ambos.

Daí a separação entre dois números que costumam ser confundidos:

    viés   = média(pred - gt)          erro sistemático, com sinal
    MAE    = média(|pred - gt|)        erro por imagem, sem sinal

Ao agregar ao longo de uma fileira, o erro aleatório se cancela e o viés se acumula. Um
sistema com MAE alto e viés zero pode entregar um total de fileira excelente; um sistema com
MAE baixo e viés consistente de +5% erra a colheita inteira em 5%. Por isso o critério de
aceitação proposto no relatório trava as duas coisas, e não só a MAE.

Nota sobre a agregação por fileira neste dataset: cada sessão de captura é um lado de uma
fileira, e seus quadros se sobrepõem. Somar as contagens por quadro portanto NÃO produz o
número real de maçãs da fileira — a mesma fruta entra várias vezes. Como o mesmo excesso está
na predição e no ground truth, o ERRO RELATIVO por fileira continua sendo uma medida válida
de qualidade agregada; o total absoluto é que não deve ser lido como colheita. O bônus de
deduplicação cross-frame ataca exatamente essa contagem repetida.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def count_metrics(pred: np.ndarray, gt: np.ndarray) -> dict[str, float]:
    """Métricas de contagem por imagem.

    Args:
        pred: contagens previstas, uma por imagem.
        gt: contagens verdadeiras, uma por imagem.
    """
    pred = np.asarray(pred, dtype=np.float64)
    gt = np.asarray(gt, dtype=np.float64)
    if len(pred) != len(gt):
        raise ValueError(f"pred ({len(pred)}) e gt ({len(gt)}) tem tamanhos diferentes")
    if len(pred) == 0:
        return {}

    error = pred - gt
    nonzero = gt > 0

    # Duas coisas diferentes, que é fácil confundir num número só.
    #
    # R2 mede CONCORDÂNCIA com a verdade: 1 - SQ(pred-gt) / SQ(gt-média(gt)). Pode ser
    # negativo, e é isso que o torna útil — um valor abaixo de zero diz que prever a média
    # seria melhor que o sistema. É o r2_score do sklearn.
    #
    # fit_r2 mede a QUALIDADE DO AJUSTE da reta pred ~ gt, e é o quadrado da correlação de
    # Pearson. Por construção de mínimos quadrados fica preso em [0, 1], logo NÃO consegue
    # revelar que o sistema é ruim: um detector que prevê 0,75*gt em toda imagem tem fit_r2
    # igual a 1,0 e R2 igual a 0,81; um que conta cada maçã duas vezes tem fit_r2 1,0 e R2
    # -2,1. Reportar só o fit_r2 esconderia exatamente o defeito que a inclinação denuncia.
    ss_tot_gt = float(((gt - gt.mean()) ** 2).sum())
    r2 = 1.0 - float(((pred - gt) ** 2).sum()) / ss_tot_gt if ss_tot_gt > 0 else np.nan

    if len(gt) > 1 and gt.std() > 0:
        slope, intercept = np.polyfit(gt, pred, 1)
        ss_res = float(((pred - (slope * gt + intercept)) ** 2).sum())
        ss_tot = float(((pred - pred.mean()) ** 2).sum())
        # Predição constante dá 0/0. Sem esta guarda o resultado sairia 1,0 — o valor mais
        # bonito da tabela apareceria justamente onde o detector parou de detectar.
        fit_r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
    else:
        slope, intercept, fit_r2 = np.nan, np.nan, np.nan

    return {
        "n_images": len(pred),
        "gt_total": float(gt.sum()),
        "pred_total": float(pred.sum()),
        "MAE": float(np.abs(error).mean()),
        "RMSE": float(np.sqrt((error**2).mean())),
        "MAPE": float((np.abs(error[nonzero]) / gt[nonzero]).mean()) if nonzero.any() else np.nan,
        "bias": float(error.mean()),
        "bias_rel": float(error.sum() / max(gt.sum(), 1e-9)),
        # R2 = concordância com a verdade (pode ser negativo, é o r2_score do sklearn).
        # fit_r2 = qualidade do ajuste da reta pred~gt (preso em [0,1], não revela defeito).
        "R2": float(r2),
        "fit_r2": float(fit_r2),
        "slope": float(slope),
        "intercept": float(intercept),
    }


def row_level_metrics(df: pd.DataFrame, group: str = "session") -> dict[str, float]:
    """Erro relativo no nível de agregação do negócio (aqui, a fileira).

    É a métrica de aceitação defendida no relatório. Compara o total previsto e o total real
    por fileira, e reporta o pior caso — porque um contrato de colheita quebra pela pior
    fileira, não pela média.

    ``df`` precisa das colunas ``pred``, ``gt`` e a coluna de agrupamento.
    """
    # Acesso por colchetes de proposito: `totals.gt` devolveria o metodo DataFrame.gt(),
    # nao a coluna. E um erro silencioso -- o codigo roda e compara contra uma funcao.
    totals = df.groupby(group)[["pred", "gt"]].sum()
    rel = (totals["pred"] - totals["gt"]) / totals["gt"].clip(lower=1)
    return {
        "n_rows": len(totals),
        "row_MAPE": float(rel.abs().mean()),
        "row_worst_abs_rel": float(rel.abs().max()),
        "row_bias_rel": float(rel.mean()),
        "row_spread": float(rel.max() - rel.min()),
    }


def per_group_metrics(df: pd.DataFrame, group: str = "session") -> pd.DataFrame:
    """Métricas de contagem quebradas por grupo — expõe a variação entre condições."""
    rows = []
    for name, grp in df.groupby(group):
        rows.append(
            {group: name, **count_metrics(grp["pred"].to_numpy(), grp["gt"].to_numpy())}
        )
    return pd.DataFrame(rows)


def acceptance_check(
    metrics: dict[str, float],
    row_metrics: dict[str, float],
    max_row_mape: float = 0.10,
    max_abs_bias: float = 0.03,
    max_worst_row: float = 0.15,
) -> dict[str, object]:
    """Aplica o critério de go/no-go proposto para uma V1.

        erro relativo medio por fileira <= 10%
        |vies relativo|                 <=  3%
        pior fileira                    <= 15%

    Os números são uma proposta defensável, não um padrão da indústria: o relatório explicita
    de onde saem e o que mudaria se o cliente tolerasse mais erro em troca de mais velocidade.
    """
    checks = {
        "row_MAPE<=10%": row_metrics["row_MAPE"] <= max_row_mape,
        "|bias_rel|<=3%": abs(metrics["bias_rel"]) <= max_abs_bias,
        "worst_row<=15%": row_metrics["row_worst_abs_rel"] <= max_worst_row,
    }
    return {**checks, "GO": all(checks.values())}
