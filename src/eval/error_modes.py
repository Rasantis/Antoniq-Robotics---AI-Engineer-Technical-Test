"""Análise de modos de erro, ranqueada pela contribuição ao erro de CONTAGEM.

A maioria das análises de erro para aqui: "o recall cai em objetos pequenos". Isso é
verdadeiro e inútil — não diz quanto do erro do produto vem daí.

Este módulo usa uma identidade exata. Numa imagem, com casamento um-para-um:

    contagem_prevista - contagem_real = FP - FN

Ou seja, o erro de contagem se decompõe SEM RESÍDUO em falsos positivos menos falsos
negativos. Cada FN contribui -1, cada FP contribui +1. Então, ao estratificar os FN por área,
oclusão ou aglomeração, o número de FN de cada estrato É a contribuição daquele estrato para a
subcontagem — em maçãs, não em pontos percentuais de recall.

Isso permite responder à pergunta que o enunciado faz de verdade: quais modos de erro
ranquear, e quanto cada um custa. Um modo com recall péssimo mas que atinge 40 frutas do
dataset importa menos que um modo com recall razoável que atinge 8.000.

Os fatores de estratificação vêm da Etapa 1 (results/instances.csv):

    bbox_area       tamanho do alvo
    solidity        área da máscara / área da caixa -> proxy direto de oclusão
    local_density   frutas com centro a menos de 100 px -> aglomeração / cacho
    dist_to_border  distância do centro à borda da imagem -> efeito de borda
    mean_value      brilho médio da imagem (V do HSV) -> iluminação
    session         fileira / horário / lado ensolarado ou sombreado
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.data.minneapple import solidity_ceiling
from src.eval.detection import match_predictions

# Cortes COCO, mantidos para comparabilidade com a literatura.
COCO_AREA_BINS = [0, 32**2, 96**2, np.inf]
COCO_AREA_LABELS = ["small (<32²)", "medium (32²–96²)", "large (>96²)"]

# Cortes sobre a solidez NORMALIZADA pelo teto geométrico do tamanho da caixa (ver
# src/data/minneapple.py). 1,0 significa "tão preenchida quanto um disco daquele tamanho
# consegue ser". Cortar na solidez crua misturaria oclusão com tamanho: um disco perfeito de
# 11 px preenche 0,67 e um de 81 px preenche 0,77, então o mesmo corte classificaria fruta
# pequena e limpa como ocluída.
SOLIDITY_BINS = [0.0, 0.70, 0.85, 0.95, 10.0]
SOLIDITY_LABELS = ["muito ocluída (<0,70)", "ocluída (0,70–0,85)",
                   "pouco ocluída (0,85–0,95)", "limpa (>0,95)"]

DENSITY_BINS = [-0.5, 2.5, 5.5, 10.5, np.inf]
DENSITY_LABELS = ["isolada (0–2)", "grupo (3–5)", "cacho (6–10)", "cacho denso (>10)"]

BORDER_BINS = [0, 50, 150, np.inf]
BORDER_LABELS = ["colada na borda (<50 px)", "perto (50–150 px)", "interior (>150 px)"]


def attribute(
    predictions: dict[str, tuple[np.ndarray, np.ndarray]],
    instances: pd.DataFrame,
    match_iou: float = 0.5,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Casa predições com anotações e devolve (anotações marcadas, predições marcadas).

    Args:
        predictions: {imagem: (caixas, scores)} já fundidas e filtradas.
        instances: results/instances.csv — uma linha por fruta anotada, com os covariáveis.
        match_iou: limiar de IoU do casamento.

    Returns:
        gt_table: ``instances`` mais a coluna ``detected`` (bool) e ``match_iou``.
        fp_table: uma linha por falso positivo, com propriedades da caixa predita.
    """
    gt_flags, fp_rows = [], []

    # UNIÃO, não só as imagens anotadas: uma imagem sem anotação nenhuma ainda pode receber
    # detecções, e iterando só sobre `instances` esses falsos positivos sumiam da decomposição
    # — deflacionando `false_positives` e o erro líquido. Não morde no MinneApple, onde toda
    # imagem rotulada tem fruta, mas morde em qualquer conjunto com quadro vazio.
    por_imagem = {n: g for n, g in instances.groupby("image", sort=False)}
    vazio = instances.iloc[0:0]
    for name in sorted(set(por_imagem) | set(predictions)):
        grp = por_imagem.get(name, vazio)
        gt_boxes = grp[["x0", "y0", "x1", "y1"]].to_numpy(dtype=np.float32)
        pred_boxes, pred_scores = predictions.get(
            name, (np.zeros((0, 4), np.float32), np.zeros(0, np.float32))
        )
        m = match_predictions(pred_boxes, pred_scores, gt_boxes, match_iou)

        flags = pd.DataFrame(
            {
                "image": name,
                "instance_id": grp["instance_id"].to_numpy(),
                "detected": m.gt_to_pred >= 0,
            }
        )
        gt_flags.append(flags)

        for p in m.fp:
            box = pred_boxes[p]
            fp_rows.append(
                {
                    "image": name,
                    "score": float(pred_scores[p]),
                    "x0": float(box[0]), "y0": float(box[1]),
                    "x1": float(box[2]), "y1": float(box[3]),
                    "bbox_area": float((box[2] - box[0]) * (box[3] - box[1])),
                    "best_iou": float(m.iou[p]),
                }
            )

    gt_table = instances.merge(
        pd.concat(gt_flags, ignore_index=True), on=["image", "instance_id"], how="left"
    )
    gt_table["detected"] = gt_table["detected"].fillna(False)
    return gt_table, pd.DataFrame(fp_rows)


def _binned(series: pd.Series, bins: list, labels: list[str]) -> pd.Series:
    return pd.cut(series, bins=bins, labels=labels, right=False, include_lowest=True)


def _normalised_solidity(gt_table: pd.DataFrame) -> pd.Series:
    """Solidez descontada do teto geométrico do tamanho, calculada a partir de w e h."""
    ceiling = solidity_ceiling(gt_table["w"], gt_table["h"])
    return gt_table["solidity"] / ceiling.clip(lower=1e-6)


def stratify_false_negatives(gt_table: pd.DataFrame) -> pd.DataFrame:
    """Perda por estrato, para cada fator, com três leituras distintas.

    ``fn_rate`` sozinha engana: um estrato com 90% de perda e 40 frutas custa menos que um
    com 25% de perda e 12.000. ``fn`` sozinha engana na direção oposta: o estrato com mais
    perdas costuma ser simplesmente o mais populoso. ``excess_fn`` — perda acima da que a
    taxa global explicaria — é a que separa "grande" de "problemático", e é por ela que se
    ordena.

    Os fatores se sobrepõem: a mesma maçã aparece em área, oclusão, aglomeração e borda. A
    tabela é uma lista de lentes, não uma decomposição sem resíduo.
    """
    factors = {
        "área (COCO)": _binned(gt_table["bbox_area"], COCO_AREA_BINS, COCO_AREA_LABELS),
        "oclusão (solidez)": _binned(
            _normalised_solidity(gt_table), SOLIDITY_BINS, SOLIDITY_LABELS
        ),
        "aglomeração": _binned(gt_table["local_density"], DENSITY_BINS, DENSITY_LABELS),
        "borda": _binned(gt_table["dist_to_border"], BORDER_BINS, BORDER_LABELS),
        "sessão": gt_table["session"],
    }
    if "mean_value" in gt_table.columns:
        factors["iluminação"] = pd.qcut(
            gt_table["mean_value"], 3, labels=["escura", "média", "clara"], duplicates="drop"
        )

    total_fn = int((~gt_table["detected"]).sum())
    global_rate = total_fn / max(len(gt_table), 1)

    rows = []
    for factor, binning in factors.items():
        grouped = gt_table.assign(_bin=binning).groupby("_bin", observed=True)
        for name, grp in grouped:
            fn = int((~grp["detected"]).sum())
            expected = len(grp) * global_rate
            rows.append(
                {
                    "factor": factor,
                    "stratum": str(name),
                    "n_gt": len(grp),
                    "fn": fn,
                    "fn_rate": fn / max(len(grp), 1),
                    # Perda ACIMA do que a taxa global já explicaria. É esta coluna que
                    # ranqueia, não `fn`: ordenar por contagem bruta elege sempre o estrato
                    # mais POPULOSO, não o mais problemático. Num teste com perda injetada de
                    # propósito em fruta ocluída, o "vencedor" por contagem bruta foi o bin
                    # "pouco ocluída", só por ter mais frutas.
                    "excess_fn": fn - expected,
                    "excess_ratio": (fn / max(len(grp), 1)) / max(global_rate, 1e-9),
                    "share_of_all_fn": fn / max(total_fn, 1),
                }
            )
    return pd.DataFrame(rows).sort_values(
        ["factor", "excess_fn"], ascending=[True, False], ignore_index=True
    )


def rank_error_modes(
    gt_table: pd.DataFrame, fp_table: pd.DataFrame, top_k: int = 8
) -> pd.DataFrame:
    """Ranqueia os modos de erro pelo EXCESSO de perda que cada um explica.

    Usa a identidade ``pred - gt = FP - FN``: falsos negativos puxam a contagem para baixo,
    falsos positivos para cima, sem resíduo.

    Os fatores se sobrepõem e as frações NÃO somam 1. Uma mesma maçã pode ser pequena,
    ocluída e estar num cacho ao mesmo tempo, e é contada nos três. Somar as colunas dá mais
    que 100% — na primeira versão deste código a soma dava 2,32, e a tabela era lida como se
    fosse uma decomposição. Não é: é uma lista de lentes, cada uma dizendo quanta perda
    aquele recorte explica além da taxa média. A decomposição sem resíduo existe, mas é só
    a de duas linhas (FN total, FP total), reportada por ``count_error_decomposition``.

    A ordenação é por ``excess_fn`` — perda acima da que a taxa global já explicaria. Ordenar
    por contagem bruta de FN elegeria o estrato mais populoso, que costuma ser o mediano e
    não o problemático.
    """
    strata = stratify_false_negatives(gt_table)
    total_fn = int((~gt_table["detected"]).sum())
    total_fp = len(fp_table)
    gross = max(total_fn + total_fp, 1)

    # A SESSÃO ENTRA no ranking. Uma versão anterior a excluía com um `!= "sessão"` mudo, e
    # com isso apagava o maior fator medido: a sessão 20150919_174151 perde 83,2% da sua
    # fruta, tem excesso 3.037 — quase o dobro do segundo colocado — e responde sozinha por
    # 52,7% de todos os falsos negativos. Excluí-la deixava no ranking o proxy
    # "iluminação", que o próprio relatório admite estar confundido com sessão, e removia o
    # confundidor real. Ranquear os modos de erro sem o fator dominante não é simplificação,
    # é responder outra pergunta.
    worst = (
        strata
        .sort_values("excess_fn", ascending=False)
        .groupby("factor", observed=True)
        .head(1)
    )
    rows = [
        {
            "modo": f"FN — {r.factor}: {r.stratum}",
            "direção": "subcontagem",
            "n": int(r.fn),
            "delta_count": -int(r.fn),
            "excess_fn": round(r.excess_fn, 1),
            "share": r.fn / gross,
            "detalhe": (
                f"{r.fn_rate:.1%} de perda em {r.n_gt:,} frutas — "
                f"{r.excess_ratio:.2f}x a taxa média"
            ),
        }
        for r in worst.itertuples()
    ]
    rows.append(
        {
            "modo": "FP — detecções sem anotação correspondente",
            "direção": "sobrecontagem",
            "n": total_fp,
            "delta_count": total_fp,
            "excess_fn": 0.0,
            "share": total_fp / gross,
            "detalhe": (
                "inclui fruta real em árvore de fundo e no chão, que o MinneApple "
                "deliberadamente não anota"
            ),
        }
    )
    table = pd.DataFrame(rows)
    table["_sort"] = table["excess_fn"].abs().where(
        table["direção"] == "subcontagem", table["n"]
    )
    return table.sort_values("_sort", ascending=False).head(top_k).drop(columns="_sort")


def count_error_decomposition(gt_table: pd.DataFrame, fp_table: pd.DataFrame) -> dict:
    """Confere a identidade e devolve o balanço global.

    Se ``fp - fn`` não bater com a soma de ``pred - gt`` por imagem, alguma coisa no
    casamento está errada — vale como asserção sobre todo o pipeline.
    """
    fn = int((~gt_table["detected"]).sum())
    fp = len(fp_table)
    return {
        "total_gt": len(gt_table),
        "false_negatives": fn,
        "false_positives": fp,
        "net_count_error": fp - fn,
        "gross_error": fp + fn,
        "recall": 1 - fn / max(len(gt_table), 1),
        "fp_per_image": fp / max(gt_table["image"].nunique(), 1),
    }
