"""Etapa 5: análise de modos de erro, ranqueada pela contribuição ao erro de contagem.

Consome as detecções brutas gravadas pela Etapa 3, aplica a política de fusão e o limiar
escolhidos, e cruza cada anotação perdida com os covariáveis extraídos na Etapa 1 — área,
solidez (oclusão), densidade local, distância à borda, brilho da imagem e sessão.

O ranking não é por taxa de erro, e sim pelo número absoluto de maçãs que cada modo custa.
Vale a identidade exata ``previsto - real = FP - FN``, que o script confere e reporta.

Saídas em results/:
    error_gt.csv          uma linha por anotação, com a coluna `detected`
    error_fp.csv          uma linha por falso positivo
    error_strata.csv      taxa de perda e frutas perdidas por estrato, para cada fator
    error_ranking.csv     os modos ranqueados pelo custo em contagem
    figures/06_modos_de_erro.png

Uso:
    python scripts/04_error_modes.py
    python scripts/04_error_modes.py --arm C_tile640 --conf 0.30
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.eval.error_modes import (  # noqa: E402
    attribute,
    count_error_decomposition,
    rank_error_modes,
    stratify_false_negatives,
)
from src.inference import store  # noqa: E402
from src.inference.engine import arms_from_config  # noqa: E402
from src.inference.postprocess import MergePolicy, apply  # noqa: E402
from src.utils.config import experiment, paths  # noqa: E402
from src.viz import figures  # noqa: E402


def frozen_config(results: Path) -> dict:
    """Lê o ponto de operação congelado pela Etapa 3.

    A análise de erro roda no MESMO ponto em que o sistema rodaria, e esse ponto foi
    escolhido na validação. Re-selecionar aqui — por menor MAE no teste, como uma versão
    anterior deste script fazia — reintroduziria a seleção sobre o conjunto de avaliação
    pela porta dos fundos.
    """
    path = results / "operating_point.json"
    if not path.exists():
        raise SystemExit(f"{path} nao existe. Rode 03_eval_arms.py antes.")
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", default=None, help="braço a analisar (padrão: menor MAE)")
    parser.add_argument("--conf", type=float, default=None, help="limiar (padrão: menor MAE)")
    args = parser.parse_args()

    cfg, p = experiment(), paths()
    results = p.resolve(p.results_root)
    frozen = frozen_config(results)
    arm = args.arm or frozen["arm"]
    spec = frozen["per_arm"].get(arm, frozen)
    conf = args.conf if args.conf is not None else spec["conf"]
    policy = MergePolicy(
        metric=spec["metric"], policy=spec["merge_policy"],
        threshold=spec["threshold"], drop_truncated=spec["drop_truncated"],
    )
    print(f"analisando: braço={arm}  conf={conf:.2f}  fusão={policy.label}")
    print(f"(congelado na validação: {frozen['criterion']})\n")

    # ---- reconstrói as predições finais a partir das detecções brutas -------------
    predictions = {}
    for fold in range(cfg["data"]["n_folds"]):
        path = store.path_for(results, fold, arm, "test")
        if not path.exists():
            raise SystemExit(f"{path} nao existe. Rode 03_eval_arms.py --infer.")
        for name, raw in store.load(path).items():
            det = apply(raw, policy, conf)
            predictions[name] = (det.boxes, det.scores)
    print(f"[1/4] {len(predictions)} imagens com predição (união dos folds de teste)")

    # ---- casa com as anotações e junta os covariáveis ----------------------------
    instances = pd.read_csv(results / "instances.csv")
    images = pd.read_csv(results / "images.csv")[["image", "mean_value", "n_apples"]]
    instances = instances.merge(images, on="image")
    instances = instances[instances["image"].isin(predictions)]

    gt_table, fp_table = attribute(
        predictions, instances, match_iou=cfg["counting"]["match_iou"]
    )
    print(f"[2/4] {len(gt_table)} anotações, {len(fp_table)} falsos positivos")

    balance = count_error_decomposition(gt_table, fp_table)
    print("\n  --- balanço do erro de contagem ---")
    print(f"  anotações .............. {balance['total_gt']:,}")
    print(f"  falsos negativos ....... {balance['false_negatives']:,}  (recall "
          f"{balance['recall']:.1%})")
    print(f"  falsos positivos ....... {balance['false_positives']:,}  "
          f"({balance['fp_per_image']:.1f} por imagem)")
    print(f"  erro LÍQUIDO (FP - FN) . {balance['net_count_error']:+,}")
    print(f"  erro BRUTO (FP + FN) ... {balance['gross_error']:,}")
    print("  O erro líquido é o que aparece na contagem; o bruto é o que o detector")
    print("  realmente erra. A diferença entre os dois é cancelamento, não acerto.")

    # ---- estratificação e ranking ------------------------------------------------
    strata = stratify_false_negatives(gt_table)
    ranking = rank_error_modes(gt_table, fp_table)

    print("\n[3/4] perda por estrato (piores 3 de cada fator)")
    for factor, grp in strata.groupby("factor", observed=True):
        print(f"\n  {factor}")
        for r in grp.head(3).itertuples():
            print(f"    {r.stratum:<28} {r.fn:>6,} perdidas de {r.n_gt:>6,} "
                  f"({r.fn_rate:>5.1%})  {r.share_of_all_fn:>5.1%} de todos os FN")

    print("\n  --- modos ranqueados pelo custo em contagem ---")
    for r in ranking.itertuples():
        print(f"  {r.delta_count:>+7,}  ({r.share:>5.1%})  {r.modo}")
        print(f"           {r.detalhe}")

    # ---- persistência ------------------------------------------------------------
    print("\n[4/4] gravando")
    gt_table.to_csv(results / "error_gt.csv", index=False)
    fp_table.to_csv(results / "error_fp.csv", index=False)
    strata.to_csv(results / "error_strata.csv", index=False)
    ranking.to_csv(results / "error_ranking.csv", index=False)
    (results / "error_balance.json").write_text(
        json.dumps({"arm": arm, "conf": conf, "policy": policy.label, **balance}, indent=2),
        encoding="utf-8",
    )
    fig = figures.error_mode_ranking(ranking, p.resolve(p.figures_root))
    print(f"  -> {results}\n  -> {fig}")


if __name__ == "__main__":
    main()
