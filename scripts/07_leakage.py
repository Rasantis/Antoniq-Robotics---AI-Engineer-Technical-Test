"""Etapa 2b: quanto o split aleatório infla a métrica.

O experimento existe para dar um número à afirmação central da §2 do relatório. Dois modelos,
o mesmo detector, o mesmo treino, as MESMAS imagens de avaliação — só muda como o conjunto de
teste foi construído:

    agrupado    teste = sessões inteiras que o modelo nunca viu
    aleatório   teste = imagens sorteadas; quadros vizinhos das mesmas sessões estão no treino

Como cada imagem cai no teste de exatamente um fold agrupado, dá para pegar as predições do
modelo agrupado para as MESMAS 224 imagens do teste aleatório. A comparação fica pareada por
imagem, o que remove a diferença de composição do conjunto.

O que este número é, e o que não é. Ele é um LIMITE SUPERIOR do efeito de vazamento, não
uma medida isolada dele. Três coisas diferem entre os dois modelos, e o experimento não as
separa:

  1. vazamento de quadro — quadros quase idênticos em treino e teste (o efeito de interesse);
  2. cobertura de domínio — o modelo aleatório treina com as 10 sessões, o agrupado com 5 ou 6,
     e as sessões variam de 21,9 a 98,1 maçãs por imagem;
  3. early stopping — a validação do modelo aleatório também é sorteada, portanto também
     vazada, então ele para num checkpoint mais sobreajustado.

Separar os três exigiria um terceiro braço de controle (10 sessões no treino, mas sem os
quadros vizinhos do teste), que o dataset permite construir e fica declarado como próximo
passo. O relatório reporta o número com esta ressalva, e não como "o efeito do vazamento".

Uso:
    python scripts/07_leakage.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# O torch PRECISA ser importado antes do pandas neste ambiente Windows, senao falha com
# WinError 1114 na inicializacao da DLL. Determinístico, medido — ver src/utils/torch_first.py.
from src.utils import torch_first  # noqa: F401  isort:skip

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

from src.eval.counting import count_metrics  # noqa: E402
from src.eval.detection import coco_metrics, match_predictions, operating_point_stats  # noqa: E402
from src.inference import store  # noqa: E402
from src.inference.engine import Arm, arms_from_config, run_arm  # noqa: E402
from src.inference.postprocess import MergePolicy, apply  # noqa: E402
from src.utils.config import experiment, paths  # noqa: E402
from src.utils.seed import set_seed  # noqa: E402

# A pergunta aqui é sobre o SPLIT, não sobre a estratégia de inferência. Usar o braço de
# imagem inteira mantém o experimento barato e a comparação limpa de um confundidor a menos.
ARM_NAME = "A_full640"


def metrics_for(preds, gt, images, match_iou, image_wh) -> dict:
    matches, rows = [], []
    for name in images:
        boxes, scores = preds.get(name, (np.zeros((0, 4), np.float32), np.zeros(0, np.float32)))
        truth = gt.get(name, np.zeros((0, 4), np.float32))
        matches.append(match_predictions(boxes, scores, truth, match_iou))
        rows.append({"pred": len(boxes), "gt": len(truth)})
    table = pd.DataFrame(rows)
    return {
        **operating_point_stats(matches),
        **count_metrics(table["pred"].to_numpy(), table["gt"].to_numpy()),
        **coco_metrics({n: gt.get(n, np.zeros((0, 4), np.float32)) for n in images},
                       {n: preds.get(n, (np.zeros((0, 4), np.float32),
                                         np.zeros(0, np.float32))) for n in images},
                       image_wh=image_wh),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default=None, help="padrão: GPU se houver, senão CPU")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    cfg, p = experiment(), paths()
    set_seed(cfg["seed"])
    results = p.resolve(p.results_root)
    match_iou = cfg["counting"]["match_iou"]
    image_wh = tuple(cfg["data"]["image_size"])
    arm: Arm = next(a for a in arms_from_config(cfg) if a.name == ARM_NAME)

    frozen_path = results / "operating_point.json"
    if not frozen_path.exists():
        raise SystemExit("operating_point.json ausente. Rode 03_eval_arms.py antes.")
    frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
    spec = frozen["per_arm"].get(ARM_NAME, frozen)
    policy = MergePolicy(metric=spec["metric"], policy=spec["merge_policy"],
                         threshold=spec["threshold"], drop_truncated=spec["drop_truncated"])
    conf = spec["conf"]
    print(f"braço {ARM_NAME} | conf {conf:.2f} | fusão {policy.label}")
    print("(ponto de operação congelado na validação AGRUPADA, aplicado aos dois modelos)\n")

    # ---- inferência do modelo aleatório sobre o teste aleatório ---------------------
    weights = p.runs_root / "train" / "random_fold0" / "weights" / "best.pt"
    if not weights.exists():
        raise SystemExit(f"{weights} nao existe. Rode 02_train_cv.py --only random_fold0.")

    listing = (p.runs_root / "folds" / "random_fold0" / "test.txt").read_text(encoding="utf-8")
    images = [Path(l).name for l in listing.splitlines() if l.strip()]
    print(f"[1/3] teste aleatório: {len(images)} imagens")

    raw_path = store.path_for(results, 0, ARM_NAME, "test", "random")
    if args.force or not raw_path.exists():
        from src.inference.engine import UltralyticsDetector

        detector = UltralyticsDetector(str(weights), device=args.device)
        per_image = {}
        for name in tqdm(images, desc="  random", ncols=78):
            img = np.asarray(Image.open(p.train_images / name).convert("RGB"))
            per_image[name] = run_arm(detector, img, arm)
        store.save(raw_path, per_image)
    random_raw = store.load(raw_path)

    # ---- predições do modelo AGRUPADO para as mesmas imagens ------------------------
    print("[2/3] predições agrupadas para as mesmas imagens")
    grouped_raw = {}
    for fold in range(cfg["data"]["n_folds"]):
        path = store.path_for(results, fold, ARM_NAME, "test", "grouped")
        if path.exists():
            grouped_raw.update(store.load(path))
    shared = [n for n in images if n in grouped_raw]
    missing = len(images) - len(shared)
    if missing:
        print(f"  AVISO: {missing} imagens sem predição agrupada — "
              f"rode 03_eval_arms.py --infer para todos os folds")
    if not shared:
        raise SystemExit("Nenhuma imagem em comum. Rode 03_eval_arms.py --infer.")

    # ---- comparação pareada --------------------------------------------------------
    print(f"[3/3] comparando nas {len(shared)} imagens em comum\n")
    gt = {
        name: grp[["x0", "y0", "x1", "y1"]].to_numpy(dtype=np.float32)
        for name, grp in pd.read_csv(results / "instances.csv").groupby("image")
    }

    rows = []
    for label, raw in (("aleatório (vazado)", random_raw), ("agrupado (honesto)", grouped_raw)):
        preds = {}
        for name in shared:
            det = apply(raw[name], policy, conf)
            preds[name] = (det.boxes, det.scores)
        rows.append({"split": label, **metrics_for(preds, gt, shared, match_iou, image_wh)})

    table = pd.DataFrame(rows)
    cols = ["split", "AP", "AP50", "AP_small", "AR_300", "precision", "recall", "f1",
            "MAE", "bias_rel"]
    print(table[cols].round(4).to_string(index=False))

    leaked, honest = table.iloc[0], table.iloc[1]
    print("\n  --- inflação atribuível ao split aleatório ---")
    for metric in ("AP", "AP50", "AP_small", "f1"):
        delta = leaked[metric] - honest[metric]
        rel = delta / max(abs(honest[metric]), 1e-9)
        print(f"  {metric:<10} {honest[metric]:.4f} -> {leaked[metric]:.4f}   "
              f"{delta:+.4f}  ({rel:+.1%})")
    print(f"  {'MAE':<10} {honest['MAE']:.2f} -> {leaked['MAE']:.2f} maçãs/imagem")

    table["n_images"] = len(shared)
    table.to_csv(results / "leakage.csv", index=False)
    print(f"\n  -> {results / 'leakage.csv'}")
    print("\n  LEMBRETE: este delta é um LIMITE SUPERIOR do vazamento. Ele soma três efeitos")
    print("  — quadros vizinhos, cobertura de domínio (10 sessões contra 5-6) e early stopping")
    print("  sobre uma validação também sorteada. O relatório declara isso.")


if __name__ == "__main__":
    main()
