"""Etapa 3: compara as estratégias de inferência, escolhe o ponto de operação e avalia.

A separação entre ESCOLHER e MEDIR é o ponto central deste script.

    SELEÇÃO   roda sobre as sessões de validação (val.txt). Escolhe o braço, a política de
              fusão e o limiar de confiança.
    AVALIAÇÃO roda sobre as sessões de teste (test.txt), com tudo já congelado.

Sem essa separação o número reportado seria um mínimo sobre 4 braços × 24 políticas × 13
limiares calculado no próprio conjunto de teste, apresentado como desempenho de teste. O caso
mais indefensável seria o critério de aceitação: ele exige |viés| ≤ 3% e a varredura escolheria
justamente o limiar de menor viés no teste — o critério passaria por construção, sem conteúdo.

As sessões de validação já existiam desde a Etapa 1, separadas e limpas, usadas só para early
stopping. Calibrar o ponto de operação é exatamente para o que elas servem.

Regra de seleção, escolhida para espelhar o critério de aceitação do relatório: entre as
combinações que respeitam |viés relativo| ≤ 3% na validação, fica a de menor MAE. Selecionar
só por MAE premiaria cancelamento entre falso positivo e falso negativo — em regime de recall
baixo, deixar duplicatas passar melhora a MAE porque a sobrecontagem compensa a subdetecção.

Duas fases de execução, separadas porque uma é cara e a outra não:

    --infer     roda o detector uma vez por (fold, braço, split) e grava as detecções BRUTAS
    --analyze   faz seleção e avaliação sobre o que está gravado, sem tocar no detector

Saídas em results/:
    selection.csv         a varredura completa na VALIDAÇÃO (braço × política × limiar)
    operating_point.json  o que ficou congelado, e por quê
    arms.csv              avaliação no TESTE, cada braço no seu próprio ponto de operação
    merge_ablation.csv    ablação de fusão na validação
    merge_paired.csv      IoS vs IoU pareado, por métrica — o achado da Tarefa 2
    conf_sweep.csv        curvas de calibração (validação)
    per_image.csv         contagem prevista e real por imagem, no teste
    grid_stats.csv        a geometria REAL de cada braço: sobreposição e redundância medidas

Uso:
    python scripts/03_eval_arms.py
    python scripts/03_eval_arms.py --analyze
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# O torch PRECISA ser importado antes do pandas neste ambiente Windows, senao falha com
# WinError 1114 na inicializacao da DLL. Determinístico, medido — ver src/utils/torch_first.py.
from src.utils import torch_first  # noqa: F401  isort:skip

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

from src.eval.ablation import paired_wins  # noqa: E402
from src.eval.counting import acceptance_check, count_metrics, row_level_metrics  # noqa: E402
from src.eval.detection import coco_metrics, match_predictions, operating_point_stats  # noqa: E402
from src.inference import store  # noqa: E402
from src.inference.engine import (RAW_CONF, Arm, arms_from_config,  # noqa: E402
                                  resolve_device, run_arm)
from src.inference.postprocess import MergePolicy, ablation_grid, apply  # noqa: E402
from src.tiling.slicer import coverage_map, tile_grid  # noqa: E402
from src.utils.config import experiment, paths  # noqa: E402
from src.utils.seed import set_seed  # noqa: E402

MAX_ABS_BIAS = 0.03  # o mesmo teto do critério de aceitação, em src/eval/counting.py


# ------------------------------------------------------------------------------ entradas

def load_ground_truth(results_root: Path) -> dict[str, np.ndarray]:
    inst = pd.read_csv(results_root / "instances.csv")
    return {
        name: grp[["x0", "y0", "x1", "y1"]].to_numpy(dtype=np.float32)
        for name, grp in inst.groupby("image")
    }


def fold_images(runs_root: Path, fold: int, split: str, tag: str = "grouped") -> list[str]:
    path = runs_root / "folds" / f"{tag}_fold{fold}" / f"{split}.txt"
    return [Path(l).name for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def grid_statistics(arms: list[Arm], image_wh: tuple[int, int]) -> pd.DataFrame:
    """Geometria REAL de cada braço, medida — não a que a configuração declara.

    Necessário porque a regra "último tile encostado na borda" faz a sobreposição efetiva
    divergir da configurada quando o tile não cabe um número inteiro de vezes no eixo. No
    braço C, tile 640 num eixo de 720 colapsa a grade em dois offsets [0, 80]: a sobreposição
    real em x é de 87,5%, não os 0,2 declarados. Chamar isso de "sobreposição 0,2" no
    relatório seria errado, e compararia braços de redundâncias muito diferentes como se
    fossem equivalentes.
    """
    rows = []
    for arm in arms:
        if arm.tile is None:
            rows.append({"arm": arm.name, "n_tiles": 1, "overlap_cfg": 0.0,
                         "overlap_real_x": 0.0, "overlap_real_y": 0.0, "redundancy": 1.0})
            continue
        grid = tile_grid(image_wh, arm.tile, arm.overlap)
        xs = sorted({int(t[0]) for t in grid})
        ys = sorted({int(t[1]) for t in grid})

        def real_overlap(offsets: list[int], tile: int) -> float:
            if len(offsets) < 2:
                return 0.0
            steps = np.diff(offsets)
            return float(np.mean(np.clip(tile - steps, 0, tile) / tile))

        rows.append({
            "arm": arm.name,
            "n_tiles": len(grid),
            "overlap_cfg": arm.overlap,
            "overlap_real_x": real_overlap(xs, min(arm.tile, image_wh[0])),
            "overlap_real_y": real_overlap(ys, min(arm.tile, image_wh[1])),
            "redundancy": float(coverage_map(grid, image_wh).mean()),
        })
    return pd.DataFrame(rows)


# ------------------------------------------------------------------------------ inferência

def run_inference(p, folds, arms, splits, force, device, limit, tag="grouped") -> None:
    from src.inference.engine import UltralyticsDetector

    results = p.resolve(p.results_root)
    for fold in folds:
        weights = p.runs_root / "train" / f"{tag}_fold{fold}" / "weights" / "best.pt"
        if not weights.exists():
            raise SystemExit(f"Pesos nao encontrados: {weights}. Rode 02_train_cv.py.")

        todo = [
            (arm, split) for split in splits for arm in arms
            if force or not store.path_for(results, fold, arm.name, split, tag).exists()
        ]
        if not todo:
            print(f"  {tag} fold {fold}: detecções brutas já existem, pulando")
            continue

        detector = UltralyticsDetector(str(weights), device=device)
        for arm, split in todo:
            images = fold_images(p.runs_root, fold, split, tag)
            if limit:
                images = images[:limit]
            per_image = {}
            for name in tqdm(images, desc=f"  {tag}{fold} {split} {arm.name}", ncols=78):
                img = np.asarray(Image.open(p.train_images / name).convert("RGB"))
                per_image[name] = run_arm(detector, img, arm)
            store.save(store.path_for(results, fold, arm.name, split, tag), per_image)


# -------------------------------------------------------------------------------- análise

def evaluate(raw_by_image, gt, policy, conf, match_iou, with_coco, segments=None):
    """Métricas de um (braço, política, limiar).

    Dois pontos de operação, de propósito. Contagem, F1 e viés saem do limiar ``conf``, que é
    onde o sistema rodaria. O AP do COCO sai da lista COMPLETA: ele é a integral da curva
    precisão-recall sobre todas as detecções ordenadas por score, e cortar em ``conf`` antes
    trunca a cauda — medido aqui, calcular o AP a 0,25 em vez de 0,01 custa ~0,30 de AP50.

    A latência inclui a FUSÃO. Ela é custo que só os braços com tile pagam, é um laço Python
    de uma thread só, e é justamente o que fica caro num núcleo ARM de Jetson. Deixá-la de
    fora tornaria o tiling artificialmente barato na comparação.

    ``segments`` é uma lista de ``(raw_by_image, policy, conf)``. Quando presente, substitui
    os três primeiros argumentos e avalia cada segmento no seu próprio ponto de operação,
    agregando as métricas uma única vez sobre a união. É o que o protocolo aninhado exige: o
    teste do fold *k* é lido no ponto que a validação do fold *k* escolheu, e o agregado sai
    das predições concatenadas — não de uma média de médias, que daria outro número.
    """
    preds, preds_full, matches, rows = {}, {}, [], []
    for seg_raw, seg_policy, seg_conf in (segments or [(raw_by_image, policy, conf)]):
        for name, raw in seg_raw.items():
            t0 = time.perf_counter()
            det = apply(raw, seg_policy, seg_conf)
            merge_ms = (time.perf_counter() - t0) * 1e3

            # A fusão da lista completa só é necessária para o AP, e é MUITO mais cara: em
            # RAW_CONF há uma ordem de magnitude mais caixas, e a fusão é O(n²). Fazê-la em
            # cada combinação da seleção — onde o AP nem é pedido — dominava o tempo total.
            preds[name] = (det.boxes, det.scores)
            if with_coco:
                det_full = apply(raw, seg_policy, RAW_CONF)
                preds_full[name] = (det_full.boxes, det_full.scores)
            truth = gt.get(name, np.zeros((0, 4), np.float32))
            matches.append(match_predictions(det.boxes, det.scores, truth, match_iou))
            rows.append({
                "image": name, "pred": det.count, "gt": len(truth),
                "n_raw": det.n_raw, "dropped_border": det.n_dropped_border,
                "duplicates_removed": det.duplicates_removed, "n_tiles": raw.n_tiles,
                "latency_infer_ms": raw.latency_ms.get("slice", 0.0)
                + raw.latency_ms.get("infer", 0.0),
                "latency_merge_ms": merge_ms,
            })

    table = pd.DataFrame(rows)
    metrics = {
        **operating_point_stats(matches),
        **count_metrics(table["pred"].to_numpy(), table["gt"].to_numpy()),
        "latency_total_ms": float(
            (table["latency_infer_ms"] + table["latency_merge_ms"]).median()
        ),
        "latency_merge_ms": float(table["latency_merge_ms"].median()),
        "n_tiles": int(table["n_tiles"].iloc[0]) if len(table) else 0,
    }
    if with_coco:
        # As chaves vêm de `preds_full`, não de `raw_by_image`: com `segments` este último é
        # None e o AP sairia de um conjunto vazio, devolvendo -1 em silêncio.
        metrics.update(coco_metrics(
            {k: gt.get(k, np.zeros((0, 4), np.float32)) for k in preds_full},
            preds_full,
            image_wh=tuple(experiment()["data"]["image_size"]),
        ))
    return metrics, table


def load_split(results: Path, folds, arm_name, split, tag="grouped") -> dict:
    merged = {}
    for fold in folds:
        path = store.path_for(results, fold, arm_name, split, tag)
        if path.exists():
            merged.update(store.load(path))
    return merged


def conf_grid(cfg) -> list[float]:
    """Grade de limiares da SELEÇÃO — fina, e por um motivo medido.

    A grade de relatório (`counting.conf_sweep`, passo 0,05) serve para desenhar curvas e é
    inadequada para impor uma restrição. O viés do braço A é monótono em ``conf`` e cruza
    zero entre 0,12 e 0,14; com passo 0,05 a varredura salta de +10,3% (0,10) para −4,1%
    (0,15) e o braço aparece como inviável sob |viés| ≤ 3% — quando existem quatro limiares
    viáveis, o melhor deles com MAE de validação 8,99 contra 17,84 do braço que era
    congelado no lugar dele.

    Uma restrição só é aplicável se a grade a amostrar. Passo 0,01.
    """
    # Piso em 0,01, nao em 0,05. Medido no E1: o limiar otimo de duas das tres sessoes de
    # teste fica ABAIXO de 0,05 (0,01 e 0,04) — uma grade que comeca em 0,05 nao consegue
    # sequer expressar a resposta certa para elas.
    return [round(float(c), 2) for c in np.arange(0.01, 0.91, 0.01)]


def select_on_validation(results, folds, arms, gt, cfg) -> pd.DataFrame:
    """Varre braço × política × limiar na VALIDAÇÃO. Nada aqui toca no teste."""
    match_iou = cfg["counting"]["match_iou"]
    grid = conf_grid(cfg)
    rows = []
    for arm in arms:
        raw = load_split(results, folds, arm.name, "val")
        if not raw:
            continue
        # Braço de um tile só: `postprocess.apply` pula a fusão entre tiles quando
        # ``n_tiles <= 1``, então as 24 políticas dão resultado IDÊNTICO. Varrê-las produzia
        # 24 cópias da mesma linha e fazia o espaço de busca parecer 24x maior do que é —
        # foi essa contagem inflada que sustentou a afirmação de que "nenhuma das 24x13
        # combinações" viabilizava o braço A. O espaço real dele é a grade de limiares.
        policies = ablation_grid(cfg) if arm.tile else [MergePolicy.from_config(cfg)]
        for policy in tqdm(policies, desc=f"  val {arm.name}", ncols=78):
            for conf in grid:
                m, _ = evaluate(raw, gt, policy, conf, match_iou, with_coco=False)
                rows.append({
                    "arm": arm.name, "policy_label": policy.label, "metric": policy.metric,
                    "policy": policy.policy, "threshold": policy.threshold,
                    "drop_truncated": policy.drop_truncated, "conf": conf,
                    "n_policies_reais": len(policies), **m,
                })
    return pd.DataFrame(rows)


GROUP_KEYS = ["arm", "policy_label", "metric", "policy", "threshold", "drop_truncated", "conf"]


def pool_folds(selection: pd.DataFrame) -> pd.DataFrame:
    """Junta as linhas por fold numa tabela de validação AGREGADA.

    Serve só para escolher o ponto de operação que vai para `operating_point.json` — o que
    um sistema implantado usaria, calibrado em toda a validação disponível. Isso é legítimo:
    o vazamento que a seleção por fold conserta estava na avaliação, não em existir um
    ponto único de deployment.

    Congelar direto sobre a tabela por fold pegaria o mínimo entre folds, que é escolher
    a dedo o subconjunto mais favorável — e os MAE despencavam de 9,5 para 4,4 sem que nada
    tivesse melhorado.

    A agregação é exata, não média de médias: MAE é média por imagem, então soma-se
    ``MAE * n_images`` e divide-se pelo total; viés sai dos totais de contagem.
    """
    if "fold" not in selection.columns:
        return selection
    g = selection.copy()
    g["_mae_sum"] = g["MAE"] * g["n_images"]
    g["_f1_sum"] = g["f1"] * g["n_images"]
    agg = g.groupby(GROUP_KEYS, as_index=False).agg(
        n_images=("n_images", "sum"), gt_total=("gt_total", "sum"),
        pred_total=("pred_total", "sum"), _mae_sum=("_mae_sum", "sum"),
        _f1_sum=("_f1_sum", "sum"), n_folds=("fold", "nunique"),
    )
    agg["MAE"] = agg["_mae_sum"] / agg["n_images"]
    agg["f1"] = agg["_f1_sum"] / agg["n_images"]
    agg["bias_rel"] = (agg["pred_total"] - agg["gt_total"]) / agg["gt_total"].clip(lower=1)
    return agg.drop(columns=["_mae_sum", "_f1_sum"])


def calibration_optima(selection: pd.DataFrame, arm: str) -> dict:
    """Onde cada métrica é ótima ao longo do limiar de confiança, para um braço.

    Sai da tabela JÁ AGREGADA por `pool_folds`, e a agregação importa: fazendo média simples
    das métricas entre folds os ótimos saem em 0,22 / 0,13 / 0,15; com o pool exato saem em
    0,21 / 0,13 / 0,13. Uma versão anterior do relatório afirmava 0,21 / 0,17 / 0,10, que não
    corresponde a nenhuma das duas — era número escrito à mão, sem artefato que o sustentasse.

    O achado é que o ótimo de F1 e o de contagem NÃO coincidem: calibrar por F1 custa viés.
    """
    bloco = selection[selection["arm"] == arm]
    if bloco.empty:
        return {}
    linha = lambda i: {"conf": float(bloco.loc[i, "conf"]), "MAE": float(bloco.loc[i, "MAE"]),
                       "f1": float(bloco.loc[i, "f1"]),
                       "bias_rel": float(bloco.loc[i, "bias_rel"])}
    return {
        "arm": arm,
        "melhor_f1": linha(bloco["f1"].idxmax()),
        "menor_mae": linha(bloco["MAE"].idxmin()),
        "vies_zero": linha(bloco["bias_rel"].abs().idxmin()),
    }


def freeze_operating_point(selection: pd.DataFrame) -> dict:
    """Aplica a regra de seleção e devolve a configuração congelada.

    Cada braço recebe o SEU melhor ponto de operação, e não um limiar comum. Braços que rodam
    em escalas de objeto diferentes produzem distribuições de confiança deslocadas: comparar
    todos num 0,25 fixo mediria qual deles por acaso está bem calibrado ali, não qual detecta
    melhor.

    Um braço que não consiga respeitar |viés| ≤ 3% em nenhuma combinação continua na
    tabela, com o melhor ponto que ele tem e a marca ``bias_ok = false``. Deixá-lo de fora
    faria o braço sumir do comparativo justamente por ser ruim — que é o oposto de reportar.
    """
    ok = selection[selection["bias_rel"].abs() <= MAX_ABS_BIAS]
    constrained = not ok.empty
    pool = ok if constrained else selection
    best = pool.loc[pool["MAE"].idxmin()]

    per_arm = {}
    for arm, grp in selection.groupby("arm"):
        feasible = grp[grp["bias_rel"].abs() <= MAX_ABS_BIAS]
        bias_ok = not feasible.empty
        candidates = feasible if bias_ok else grp
        row = candidates.loc[candidates["MAE"].idxmin()]
        per_arm[arm] = {
            "conf": float(row["conf"]), "policy_label": row["policy_label"],
            "metric": row["metric"], "merge_policy": row["policy"],
            "threshold": float(row["threshold"]),
            "drop_truncated": bool(row["drop_truncated"]),
            "val_MAE": float(row["MAE"]), "val_f1": float(row["f1"]),
            "val_bias_rel": float(row["bias_rel"]),
            "bias_ok": bias_ok,
        }
    return {
        "arm": best["arm"],
        "conf": float(best["conf"]),
        "metric": best["metric"],
        "merge_policy": best["policy"],
        "threshold": float(best["threshold"]),
        "drop_truncated": bool(best["drop_truncated"]),
        "policy_label": best["policy_label"],
        "criterion": (
            "menor MAE na VALIDACAO entre as combinacoes com |bias_rel| <= 3%"
            if constrained else
            "menor MAE na VALIDACAO — NENHUMA combinacao respeitou |bias_rel| <= 3%"
        ),
        "bias_constraint_satisfied": constrained,
        "selected_on": "validation",
        "per_arm": per_arm,
    }


def policy_from(spec: dict) -> MergePolicy:
    return MergePolicy(
        metric=spec["metric"], policy=spec["merge_policy"],
        threshold=spec["threshold"], drop_truncated=spec["drop_truncated"],
    )


# ------------------------------------------------------------------------------- main

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--infer", action="store_true")
    parser.add_argument("--analyze", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--folds", default=None, help="ex.: '0' ou '0,2'")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--device", default=None, help="padrão: GPU se houver, senão CPU")
    parser.add_argument(
        "--reuse-selection", action="store_true",
        help="reaproveita results/selection.csv em vez de varrer a validação de novo",
    )
    args = parser.parse_args()
    do_infer = args.infer or not args.analyze
    do_analyze = args.analyze or not args.infer

    cfg, p = experiment(), paths()
    set_seed(cfg["seed"])
    results = p.resolve(p.results_root)
    arms = arms_from_config(cfg)
    folds = ([int(f) for f in args.folds.split(",")] if args.folds
             else list(range(cfg["data"]["n_folds"])))
    match_iou = cfg["counting"]["match_iou"]
    image_wh = tuple(cfg["data"]["image_size"])

    if do_infer:
        print(f"[1/5] inferência | folds={folds} device={resolve_device(args.device)}")
        run_inference(p, folds, arms, ["val", "test"], args.force, args.device, args.limit)
    if not do_analyze:
        return

    gt = load_ground_truth(results)

    print("\n[2/5] geometria real das grades")
    grids = grid_statistics(arms, image_wh)
    grids.to_csv(results / "grid_stats.csv", index=False)
    print(grids.round(3).to_string(index=False))
    print("  'overlap_real' diverge do configurado quando o tile não cabe um número inteiro")
    print("  de vezes no eixo: o último tile é encostado na borda, e a sobreposição sobe.")

    print("\n[3/5] SELEÇÃO na validação (o teste não é tocado aqui)")
    cached = results / "selection.csv"
    if args.reuse_selection and cached.exists():
        selection = pd.read_csv(cached)
        print(f"  reaproveitando {cached.name} ({len(selection)} combinações já varridas)")
    else:
        # SELEÇÃO POR FOLD, e não sobre a união das validações.
        #
        # A união vazava. As três sessões de validação são sessões de TESTE de algum fold:
        # a val do fold 0 (20150921_131453) é teste do fold 1; a do fold 1 (20150919_174151)
        # e a do fold 2 (20150921_131234) são teste do fold 0. Fundir as validações e depois
        # fundir os testes colocava 206 das 670 imagens de teste (30,7%) dentro do pool que
        # escolheu braço, limiar e política. O `assert_no_session_leak` não pegava porque
        # ele verifica vazamento DENTRO de um fold; este nasce na agregação.
        #
        # Pior: a sessão de 98,1 maçãs/img que responde por 50,9% de todos os falsos
        # negativos calibrava o limiar e depois era reportada como resultado de teste.
        parts = []
        for fold in folds:
            part = select_on_validation(results, [fold], arms, gt, cfg)
            parts.append(part.assign(fold=fold))
        selection = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
        if selection.empty:
            raise SystemExit("Sem detecções de validação. Rode com --infer.")
        selection.to_csv(cached, index=False)

    # Um ponto de operação POR FOLD. O fold k é testado no ponto que a validação do fold k
    # escolheu — nenhuma imagem de teste participa da escolha aplicada a ela.
    frozen_by_fold = {
        fold: freeze_operating_point(selection[selection["fold"] == fold])
        for fold in folds if (selection["fold"] == fold).any()
    }
    for fold, fz in frozen_by_fold.items():
        print(f"  fold {fold}: {fz['arm']} @ conf {fz['conf']:.2f}, fusão {fz['policy_label']}")

    # O ponto "global" só existe para as tabelas de apoio e para os consumidores que precisam
    # de UM ponto (run_inference.py, auditoria visual). Ele sai da validação AGREGADA — e não
    # da tabela por fold, que daria o mínimo entre folds — e a avaliação de teste NÃO o usa.
    agregada = pool_folds(selection)
    frozen = freeze_operating_point(agregada)
    (results / "operating_point.json").write_text(json.dumps(frozen, indent=2), encoding="utf-8")

    # Onde cada metrica e otima ao longo do limiar. Gravado porque a §5 cita esses numeros, e
    # um numero citado sem artefato e um numero que ninguem confere.
    so_do_congelado = agregada[agregada["policy_label"] == frozen["policy_label"]]
    otimos = calibration_optima(so_do_congelado, frozen["arm"])
    (results / "calibration_optima.json").write_text(
        json.dumps(otimos, indent=2, ensure_ascii=False), encoding="utf-8")
    for nome in ("melhor_f1", "menor_mae", "vies_zero"):
        o = otimos.get(nome, {})
        if o:
            print(f"  {nome:<11} conf {o['conf']:.2f}  (MAE {o['MAE']:.1f}, "
                  f"F1 {o['f1']:.3f}, vies {o['bias_rel']:+.1%})")
    print(f"  congelado: {frozen['arm']} @ conf {frozen['conf']:.2f}, "
          f"fusão {frozen['policy_label']}")
    print(f"  critério: {frozen['criterion']}")
    for arm_name, spec in frozen["per_arm"].items():
        flag = "" if spec["bias_ok"] else "  <- nao respeita |vies| <= 3% em nenhuma combinacao"
        print(f"    {arm_name:<12} conf={spec['conf']:.2f}  {spec['policy_label']:<20} "
              f"val_MAE={spec['val_MAE']:.2f}  val_F1={spec['val_f1']:.3f}"
              f"  val_bias={spec['val_bias_rel']:+.3f}{flag}")

    # Tabelas de apoio para o relatório, todas da validação.
    #
    # A ablação de fusão sai do melhor braço COM TILES, e não do braço congelado. Se o
    # congelado for de um tile só — que foi o que aconteceu — a fusão entre tiles é pulada
    # por construção (ver postprocess.apply) e as 24 políticas dão resultado idêntico. Uma
    # tabela de ablação com 24 linhas iguais não é um resultado nulo: é uma tabela sem
    # sentido, porque o eixo não se aplica àquele braço.
    tiled = [a.name for a in arms if a.tile]
    ablation_arm = (
        selection[selection["arm"].isin(tiled)].groupby("arm")["MAE"].min().idxmin()
        if tiled else frozen["arm"]
    )
    if ablation_arm != frozen["arm"]:
        print(f"  ablação de fusão medida em {ablation_arm}: o braço congelado tem um tile "
              f"só, onde a fusão entre tiles não se aplica")
    ablacao = selection[selection["arm"] == ablation_arm]
    ablacao.to_csv(results / "merge_ablation.csv", index=False)

    # Comparação pareada IoS vs IoU. Fica gravada porque é o achado central da Tarefa 2 e o
    # relatório o cita: um número afirmado no texto sem artefato que o sustente é um número
    # que ninguém consegue conferir — inclusive eu, meses depois.
    pareado = paired_wins(ablacao)
    pareado.to_csv(results / "merge_paired.csv", index=False)
    for _, r in pareado.iterrows():
        print(f"  IoS vence IoU em {r['metrica']:<10} {r['win_rate']:6.1%} "
              f"({int(r['wins'])} de {int(r['pairs'])} configurações pareadas)")
    selection[selection["policy_label"] == frozen["policy_label"]].to_csv(
        results / "conf_sweep.csv", index=False)

    print("\n[4/5] AVALIAÇÃO no teste — fold k lido no ponto que a validação do fold k escolheu")
    sessions = pd.read_csv(results / "images.csv")[["image", "session"]]
    arm_rows, per_image = [], []
    for arm in arms:
        # Um segmento por fold, cada um com o SEU ponto de operação. É isto que fecha o
        # vazamento: a imagem de teste do fold k é lida no limiar que a validação do fold k
        # escolheu, e essa validação não contém imagem alguma do teste do fold k.
        segments, per_fold, confs = [], [], []
        for fold in folds:
            spec = frozen_by_fold.get(fold, {}).get("per_arm", {}).get(arm.name)
            raw_f = load_split(results, [fold], arm.name, "test")
            if spec is None or not raw_f:
                continue
            pol = policy_from(spec)
            segments.append((raw_f, pol, spec["conf"]))
            confs.append(spec["conf"])
            mf, _ = evaluate(None, gt, None, None, match_iou, with_coco=True,
                             segments=[(raw_f, pol, spec["conf"])])
            per_fold.append(mf)
        if not segments:
            continue

        m, table = evaluate(None, gt, None, None, match_iou, with_coco=True, segments=segments)
        table = table.merge(sessions, on="image")

        # O desvio mede variacao ENTRE SESSOES de captura, nao incerteza do estimador: os
        # tres folds testam conjuntos de sessoes disjuntos que cobrem as 670 imagens.
        spread = {
            f"{k}_sd": float(np.std([pf[k] for pf in per_fold]))
            for k in ("AP", "AP50", "AP_small", "f1", "MAE", "bias_rel")
        } if len(per_fold) > 1 else {}

        specs = [frozen_by_fold[f]["per_arm"][arm.name] for f in folds
                 if arm.name in frozen_by_fold.get(f, {}).get("per_arm", {})]
        arm_rows.append({
            "arm": arm.name,
            "conf": float(np.mean(confs)),                      # media dos pontos por fold
            "conf_por_fold": "|".join(f"{c:.2f}" for c in confs),
            "policy_label": specs[0]["policy_label"] if specs else "",
            "val_bias_ok": all(s["bias_ok"] for s in specs) if specs else False,
            "n_folds": len(per_fold),
            **m, **spread, **row_level_metrics(table),
            **grids[grids["arm"] == arm.name].iloc[0][
                ["overlap_real_x", "overlap_real_y", "redundancy"]].to_dict(),
        })
        per_image.append(table.assign(arm=arm.name))

    arms_table = pd.DataFrame(arm_rows)
    arms_table.to_csv(results / "arms.csv", index=False)
    pd.concat(per_image, ignore_index=True).to_csv(results / "per_image.csv", index=False)

    cols = ["arm", "conf", "n_tiles", "redundancy", "AP", "AP50", "AP50_sd", "AP_small",
            "AR_300", "f1", "MAE", "MAE_sd", "bias_rel", "row_worst_abs_rel",
            "latency_total_ms"]
    print(arms_table[[c for c in cols if c in arms_table]].round(3).to_string(index=False))
    print("  *_sd = desvio ENTRE OS TRES FOLDS. Mede variacao entre sessoes de captura,")
    print("  nao incerteza do estimador — os folds testam conjuntos de sessoes diferentes.")

    # O critério roda no braço que a VALIDAÇÃO agregada elegeu. Rodá-lo no de menor MAE de
    # teste seria escolher pelo teste — exatamente o que a §2 do relatório rejeita.
    print(f"\n[5/5] critério de aceitação, no braço congelado ({frozen['arm']})")
    winner = arms_table[arms_table["arm"] == frozen["arm"]].iloc[0]
    verdict = acceptance_check(winner.to_dict(), winner.to_dict())
    for check, ok in verdict.items():
        print(f"  {check:<18} {'PASSA' if ok else 'REPROVA'}")
    print(f"\n  -> {results}")


if __name__ == "__main__":
    main()
