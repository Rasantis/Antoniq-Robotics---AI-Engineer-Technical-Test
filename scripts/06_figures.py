"""Etapa 7: gera as figuras do relatório a partir dos CSVs já produzidos.

Separado das etapas de cálculo de propósito: iterar no visual de uma figura não deve exigir
reprocessar nada. Todas as entradas são CSVs em results/.

O painel qualitativo atende ao pedido explícito da Tarefa 2 — "2-3 example images" mostrando
onde a estratégia de alta resolução ajuda e onde atrapalha. Os três casos não são escolhidos a
dedo: são o de maior ganho, o de maior perda e a mediana, selecionados pelo delta de erro
absoluto de contagem entre o braço de imagem inteira e o de tiles.

Uso:
    python scripts/06_figures.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.eval.detection import match_predictions  # noqa: E402
from src.inference import store  # noqa: E402
from src.inference.postprocess import MergePolicy, apply  # noqa: E402
from src.utils.config import experiment, paths  # noqa: E402
from src.viz import figures  # noqa: E402

BASELINE_ARM = "A_full640"  # o que se faria sem pensar: imagem inteira, imgsz padrão
TILED_ARM = "C_tile640"     # o braço de tiles do painel qualitativo, fixo aqui


def _classified_boxes(
    pred_boxes: np.ndarray, pred_scores: np.ndarray, gt_boxes: np.ndarray, match_iou: float
) -> dict[str, list]:
    """Separa as caixas em TP / FP (preditas) e FN (anotações perdidas)."""
    m = match_predictions(pred_boxes, pred_scores, gt_boxes, match_iou)
    return {
        "TP": [pred_boxes[i].tolist() for i in m.tp],
        "FP": [pred_boxes[i].tolist() for i in m.fp],
        "FN": [gt_boxes[i].tolist() for i in m.fn],
    }


def build_cases(
    p, cfg: dict, results: Path, tiled_arm: str, pontos: dict[str, tuple[MergePolicy, float]]
) -> list[dict]:
    """Escolhe três imagens: maior ganho do tiling, maior perda, e o caso mediano.

    ``pontos`` traz o par (política, limiar) de CADA braço. Antes esta função recebia um limiar
    e uma política só, aplicados aos dois: o limiar global (0,13, do A_full640) e a política do
    experiment.yaml. O painel de tiles então rodava 18 pontos abaixo do limiar dele e com outro
    limiar de casamento — um híbrido que nunca foi medido, e que inflava os falsos positivos do
    lado do tiling justamente na figura que existe para julgar o tiling.
    """
    match_iou = cfg["counting"]["match_iou"]
    gt_by_image = {
        name: grp[["x0", "y0", "x1", "y1"]].to_numpy(dtype=np.float32)
        for name, grp in pd.read_csv(results / "instances.csv").groupby("image")
    }

    per_arm: dict[str, dict] = {}
    for arm in (BASELINE_ARM, tiled_arm):
        merged = {}
        for fold in range(cfg["data"]["n_folds"]):
            path = store.path_for(results, fold, arm, "test")
            if path.exists():
                merged.update(store.load(path))
        per_arm[arm] = merged

    shared = sorted(set(per_arm[BASELINE_ARM]) & set(per_arm[tiled_arm]))
    rows = []
    for name in shared:
        truth = gt_by_image.get(name, np.zeros((0, 4), np.float32))
        counts = {}
        for arm in (BASELINE_ARM, tiled_arm):
            politica, conf = pontos[arm]
            det = apply(per_arm[arm][name], politica, conf)
            counts[arm] = det.count
        rows.append({
            "image": name,
            "gt": len(truth),
            "err_full": abs(counts[BASELINE_ARM] - len(truth)),
            "err_tiled": abs(counts[tiled_arm] - len(truth)),
        })

    table = pd.DataFrame(rows)
    table["gain"] = table["err_full"] - table["err_tiled"]  # positivo = tiling ajudou
    table = table.sort_values("gain")
    picks = [
        (table.iloc[-1]["image"], "tiling ajuda mais"),
        (table.iloc[len(table) // 2]["image"], "caso mediano"),
        (table.iloc[0]["image"], "tiling atrapalha mais"),
    ]

    cases = []
    for name, title in picks:
        truth = gt_by_image.get(name, np.zeros((0, 4), np.float32))
        image = np.asarray(Image.open(p.train_images / name).convert("RGB"))
        case = {"image": image, "title": f"{title}\n{name[:15]}  (GT {len(truth)})"}
        for key, arm in (("full", BASELINE_ARM), ("tiled", tiled_arm)):
            politica, conf = pontos[arm]
            det = apply(per_arm[arm][name], politica, conf)
            case[key] = _classified_boxes(det.boxes, det.scores, truth, match_iou)
        cases.append(case)
    return cases


def _pontos_por_braco(results: Path, padrao: MergePolicy) -> dict:
    """(política, limiar) de cada braço, do ponto que a Etapa 3 congelou.

    Cai no padrão da configuração só se `operating_point.json` não existir — e aí a figura
    sai de um ponto não calibrado, o que o print acima deixa visível.
    """
    caminho = results / "operating_point.json"
    if not caminho.exists():
        return {}
    import json

    dados = json.loads(caminho.read_text(encoding="utf-8"))
    saida = {}
    for nome, spec in dados.get("per_arm", {}).items():
        saida[nome] = (
            MergePolicy(metric=spec["metric"], policy=spec["merge_policy"],
                        threshold=spec["threshold"], drop_truncated=spec["drop_truncated"]),
            float(spec["conf"]),
        )
    return saida


def main() -> None:
    argparse.ArgumentParser(description=__doc__).parse_args()

    cfg, p = experiment(), paths()
    results = p.resolve(p.results_root)
    out = p.resolve(p.figures_root)
    policy = MergePolicy.from_config(cfg)

    # `instances.csv` tem 28 mil linhas e fica fora do repositorio, entao num clone limpo ele
    # nao existe. Sem esta checagem o script morre com um FileNotFoundError cru, que nao diz
    # ao leitor qual etapa faltou rodar.
    faltando = [n for n in ("instances.csv", "images.csv", "folds.csv")
                if not (results / n).exists()]
    if faltando:
        raise SystemExit(
            f"Faltam {', '.join(faltando)} em {results}. "
            f"Rode scripts/01_prepare_data.py antes — as figuras saem das tabelas dele."
        )

    made = []

    # --- 1 e 2: dataset e splits (só dependem da Etapa 1) --------------------------
    instances = pd.read_csv(results / "instances.csv")
    images = pd.read_csv(results / "images.csv")
    made.append(figures.dataset_overview(instances, images, out))
    made.append(figures.split_composition(pd.read_csv(results / "folds.csv"), images, out))

    # --- 3, 4, 5: dependem da Etapa 3 ---------------------------------------------
    arms_csv = results / "arms.csv"
    if arms_csv.exists():
        arms = pd.read_csv(arms_csv)
        made.append(figures.arms_comparison(arms, out))

        pareado_csv = results / "merge_paired.csv"
        if pareado_csv.exists():
            made.append(figures.merge_paired_wins(pd.read_csv(pareado_csv), out))
        else:
            print("  results/merge_paired.csv ausente — figura 4 pulada (rode 03_eval_arms.py)")

        sweep_csv = results / "conf_sweep.csv"
        if sweep_csv.exists():
            made.append(figures.conf_calibration(pd.read_csv(sweep_csv), out))

        # --- 7: painel qualitativo -------------------------------------------------
        pontos = _pontos_por_braco(results, policy)
        # O braço de tiles do painel é FIXO na configuração, não escolhido pela menor MAE de
        # `arms.csv` — aquilo é o conjunto de TESTE, e escolher por ele seria seleção sobre a
        # avaliação, ainda que só para decidir qual figura ilustrar.
        tiled = TILED_ARM if TILED_ARM in arms["arm"].unique() else None
        if tiled:
            pol_t, conf_t = pontos[tiled]
            pol_f, conf_f = pontos[BASELINE_ARM]
            print(f"  painel qualitativo: {BASELINE_ARM} @ conf {conf_f:.2f} ({pol_f.label}) "
                  f"vs {tiled} @ conf {conf_t:.2f} ({pol_t.label})")
            cases = build_cases(p, cfg, results, tiled, pontos)
            made.append(figures.qualitative_panel(cases, out))
    else:
        print("  results/arms.csv ausente — figuras 3-5 e 7 puladas (rode 03_eval_arms.py)")

    # --- 6: modos de erro (Etapa 5) -----------------------------------------------
    ranking_csv = results / "error_ranking.csv"
    if ranking_csv.exists():
        made.append(figures.error_mode_ranking(pd.read_csv(ranking_csv), out))
    else:
        print("  results/error_ranking.csv ausente — figura 6 pulada (rode 04_error_modes.py)")

    # --- 8: cross-frame (Etapa 6) -------------------------------------------------
    for suffix, note in (
        ("", "Anotacoes usadas como deteccoes: isola o mecanismo de deduplicacao."),
        ("_detector", "Deteccoes do modelo treinado; o residuo inclui os erros do detector."),
    ):
        csv = results / f"crossframe_virtual{suffix}.csv"
        if csv.exists():
            made.append(figures.crossframe_counts(
                pd.read_csv(csv), out, note=note, name=f"08_crossframe{suffix}"))

    print(f"\n{len(made)} figuras em {out}")
    for path in made:
        print(f"  {path.name}")


if __name__ == "__main__":
    main()
