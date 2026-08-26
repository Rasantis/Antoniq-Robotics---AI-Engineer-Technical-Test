"""Etapa 10: auditoria visual — anotação, predição em imagem inteira e predição com tiles.

Serve para inspecionar o que o modelo está fazendo, não para o relatório. Cada imagem vira uma
figura de três painéis lado a lado:

    ANOTADO           todas as caixas do ground truth
    IMAGEM INTEIRA    predição do braço A, com TP / FP / FN
    TILES + FUSÃO     predição do braço com tiles, com TP / FP / FN

Os painéis de predição usam cor por ESTADO e não por caixa: verde é acerto, vermelho é
detecção sem anotação correspondente, laranja é anotação que o modelo perdeu. Ver o que está
laranja e o que está vermelho é o que diz onde atacar — uma imagem cheia de laranja pede
recall, uma cheia de vermelho pede precisão ou revisão do escopo de anotação.

Saída em alta resolução, para dar zoom: fruta de 27 px num quadro de 1280 px só é julgável
ampliada.

Uso:
    python scripts/10_visual_audit.py                      # 6 imagens representativas
    python scripts/10_visual_audit.py --images a.png b.png # imagens específicas
    python scripts/10_visual_audit.py --n 12               # mais casos
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

import matplotlib
import numpy as np
import pandas as pd
from PIL import Image

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch, Rectangle  # noqa: E402

from src.eval.detection import match_predictions  # noqa: E402
from src.inference import store  # noqa: E402
from src.inference.postprocess import MergePolicy, apply  # noqa: E402
from src.utils.config import experiment, paths  # noqa: E402
from src.viz.figures import INK, INK_SOFT, STATUS, SURFACE  # noqa: E402

GT_COLOUR = "#2a78d6"
FULL_ARM = "A_full640"
TILED_ARM = "C_tile640"
DPI = 220


def load_frozen(results: Path, arm: str) -> tuple[MergePolicy, float]:
    data = json.loads((results / "operating_point.json").read_text(encoding="utf-8"))
    spec = data["per_arm"].get(arm, data)
    policy = MergePolicy(
        metric=spec["metric"], policy=spec["merge_policy"],
        threshold=spec["threshold"], drop_truncated=spec["drop_truncated"],
    )
    return policy, float(spec["conf"])


def load_frozen_model(results: Path, model: str, arm: str) -> tuple[MergePolicy, float]:
    """Ponto de operação de um dos pesos da Etapa 12, lido de ``model_comparison.csv``.

    A política de fusão continua vindo do ``operating_point.json`` — ela é propriedade do
    pipeline e não dos pesos, e é assim que a Etapa 12 a trata. O que muda por modelo é só o
    limiar de confiança, porque a distribuição de scores se desloca de um treino para outro.
    """
    policy, _ = load_frozen(results, arm)
    table = pd.read_csv(results / "model_comparison.csv")
    row = table[(table["modelo"] == model) & (table["arm"] == arm)]
    if row.empty:
        raise SystemExit(f"Sem linha para ({model}, {arm}) em model_comparison.csv")
    return policy, float(row["conf"].iloc[0])


def load_predictions(results: Path, arm: str, n_folds: int, tag: str = "grouped") -> dict:
    merged = {}
    for fold in range(n_folds):
        path = store.path_for(results, fold, arm, "test", tag)
        if path.exists():
            merged.update(store.load(path))
    return merged


def pick_images(per_image: pd.DataFrame, n: int, arm: str = FULL_ARM,
                available: set | None = None) -> list[str]:
    """Casos espalhados pela faixa de erro, com fruta suficiente para a imagem dizer algo.

    Escolher só os melhores seria propaganda; escolher só os piores seria autoflagelação. O
    útil é cobrir a faixa, porque é a variação entre sessões que define o problema aqui.

    ``available`` restringe aos nomes que têm detecção gravada: os pesos da Etapa 12 foram
    treinados só no fold 0, e o teste deles é o teste daquele fold — pedir uma imagem dos
    outros dois folds daria um KeyError silencioso disfarçado de "sem predição".
    """
    a = per_image.copy()
    if "arm" in a.columns:
        a = a[a["arm"] == arm]
    a = a[a["gt"] >= 25]
    if available is not None:
        a = a[a["image"].isin(available)]
    a["err"] = (a["pred"] - a["gt"]) / a["gt"]
    a = a.sort_values("err").reset_index(drop=True)
    if len(a) <= n:
        return a["image"].tolist()
    idx = np.linspace(0, len(a) - 1, n).round().astype(int)
    return a.loc[idx, "image"].tolist()


def draw(ax, image: np.ndarray, boxes_by_state: dict, title: str, subtitle: str) -> None:
    ax.imshow(image)
    for state, colour in boxes_by_state.items():
        for x0, y0, x1, y1 in colour["boxes"]:
            ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False,
                                   edgecolor=colour["colour"], linewidth=1.6))
    # Título e subtítulo como dois textos ancorados em alturas distintas acima do eixo. Usar
    # set_title junto com um text em y=1.005 fazia os dois se sobreporem.
    ax.text(0.0, 1.035, title, transform=ax.transAxes, fontsize=13,
            color=INK, fontweight="bold", va="bottom")
    ax.text(0.0, 1.005, subtitle, transform=ax.transAxes, fontsize=10.5,
            color=INK_SOFT, va="bottom")
    ax.set_xticks([]), ax.set_yticks([])
    for side in ax.spines.values():
        side.set_visible(False)


def audit_one(
    name: str, image: np.ndarray, gt: np.ndarray, preds: dict, match_iou: float, out: Path,
    full_arm: str = FULL_ARM, tiled_arm: str = TILED_ARM,
) -> Path:
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 10.5))
    fig.patch.set_facecolor(SURFACE)

    draw(axes[0], image,
         {"GT": {"boxes": gt, "colour": GT_COLOUR}},
         "ANOTADO (ground truth)", f"{len(gt)} maçãs marcadas por anotador humano")

    for ax, (label, arm) in zip(axes[1:], [(f"IMAGEM INTEIRA · {full_arm}", full_arm),
                                           (f"TILES + FUSÃO · {tiled_arm}", tiled_arm)]):
        boxes, scores = preds[arm]
        m = match_predictions(boxes, scores, gt, match_iou)
        states = {
            "TP": {"boxes": [boxes[i] for i in m.tp], "colour": STATUS["TP"]},
            "FP": {"boxes": [boxes[i] for i in m.fp], "colour": STATUS["FP"]},
            "FN": {"boxes": [gt[i] for i in m.fn], "colour": STATUS["FN"]},
        }
        tp, fp, fn = m.counts()
        err = (len(boxes) - len(gt)) / max(len(gt), 1)
        draw(ax, image, states, label,
             f"contou {len(boxes)} ({err:+.0%})   ·   {tp} acertos · {fp} falsos · {fn} perdidas")

    fig.legend(
        handles=[Patch(facecolor=GT_COLOUR, label="anotação"),
                 Patch(facecolor=STATUS["TP"], label="acerto (TP)"),
                 Patch(facecolor=STATUS["FP"],
                       label="falso positivo — detectou o que não é anotado"),
                 Patch(facecolor=STATUS["FN"], label="perdida (FN) — anotada e não detectada")],
        loc="lower center", ncol=4, fontsize=11, frameon=False,
    )
    fig.suptitle(name, x=0.008, ha="left", fontsize=11, color=INK_SOFT)
    fig.tight_layout(rect=[0, 0.035, 1, 0.955])

    out.mkdir(parents=True, exist_ok=True)
    path = out / f"audit_{Path(name).stem}.png"
    fig.savefig(path, dpi=DPI, facecolor=SURFACE)
    plt.close(fig)
    return path


def ponto_congelado(results: Path, modelo: str) -> tuple[str, MergePolicy, float]:
    """Braço, política e limiar que a VALIDAÇÃO escolheu para este conjunto de pesos.

    Cada modelo aparece no painel com o ponto de operação que o sistema realmente usaria —
    não com um braço comum imposto a todos. Forçar o mesmo braço mediria qual deles por acaso
    vai bem naquela configuração, e não qual sistema entrega melhor contagem.
    """
    tabela = pd.read_csv(results / "model_comparison.csv")
    linhas = tabela[tabela["modelo"] == modelo]
    if linhas.empty:
        raise SystemExit(f"{modelo} não está em model_comparison.csv")
    r = linhas.loc[linhas["val_MAE"].idxmin()]
    policy, _ = load_frozen(results, r["arm"])
    return str(r["arm"]), policy, float(r["conf"])


def galeria(nome: str, image: np.ndarray, gt: np.ndarray, paineis: list[dict],
            match_iou: float, out: Path) -> Path:
    """Uma figura: anotação + um painel por modelo, cada um no seu ponto congelado."""
    n = len(paineis) + 1
    fig, axes = plt.subplots(1, n, figsize=(5.5 * n, 10.5))
    fig.patch.set_facecolor(SURFACE)

    draw(axes[0], image, {"GT": {"boxes": gt, "colour": GT_COLOUR}},
         "ANOTADO (ground truth)", f"{len(gt)} maçãs marcadas por anotador humano")

    for ax, p in zip(axes[1:], paineis):
        boxes, scores = p["boxes"], p["scores"]
        m = match_predictions(boxes, scores, gt, match_iou)
        estados = {
            "TP": {"boxes": [boxes[i] for i in m.tp], "colour": STATUS["TP"]},
            "FP": {"boxes": [boxes[i] for i in m.fp], "colour": STATUS["FP"]},
            "FN": {"boxes": [gt[i] for i in m.fn], "colour": STATUS["FN"]},
        }
        tp, fp, fn = m.counts()
        err = (len(boxes) - len(gt)) / max(len(gt), 1)
        draw(ax, image, estados, p["titulo"],
             f"contou {len(boxes)} ({err:+.0%})   ·   {tp} acertos · {fp} falsos · {fn} perdidas")

    fig.legend(
        handles=[Patch(facecolor=GT_COLOUR, label="anotação"),
                 Patch(facecolor=STATUS["TP"], label="acerto (TP)"),
                 Patch(facecolor=STATUS["FP"], label="falso positivo"),
                 Patch(facecolor=STATUS["FN"], label="perdida (FN)")],
        loc="lower center", ncol=4, fontsize=11, frameon=False,
    )
    fig.suptitle(nome, x=0.006, ha="left", fontsize=11, color=INK_SOFT)
    fig.tight_layout(rect=[0, 0.035, 1, 0.955])
    out.mkdir(parents=True, exist_ok=True)
    caminho = out / f"modelos_{Path(nome).stem}.png"
    fig.savefig(caminho, dpi=DPI, facecolor=SURFACE)
    plt.close(fig)
    return caminho


def modo_galeria(args, cfg, p, results: Path) -> None:
    modelos = [m.strip() for m in args.models.split(",") if m.strip()]
    match_iou = cfg["counting"]["match_iou"]
    n_folds = cfg["data"]["n_folds"]

    cfgs, raws = {}, {}
    print(f"galeria de {len(modelos)} modelos, cada um no ponto que a validação escolheu:")
    for m in modelos:
        arm, policy, conf = ponto_congelado(results, m)
        cfgs[m] = (arm, policy, conf)
        raws[m] = load_predictions(results, arm, n_folds, f"cmp-{m}")
        print(f"  {m:<16} {arm:<12} conf {conf:.2f} | fusão {policy.label} | "
              f"{len(raws[m])} imagens")

    gt_by_image = {
        n: g[["x0", "y0", "x1", "y1"]].to_numpy(np.float32)
        for n, g in pd.read_csv(results / "instances.csv").groupby("image")
    }
    comuns = set.intersection(*(set(r) for r in raws.values()))
    if args.images:
        nomes = [n for n in args.images if n in comuns]
    else:
        ref = modelos[-1]  # o último da lista é a referência para espalhar pela faixa de erro
        arm_ref, pol_ref, conf_ref = cfgs[ref]
        ranking = pd.DataFrame([
            {"image": n, "pred": apply(raws[ref][n], pol_ref, conf_ref).count,
             "gt": len(gt_by_image.get(n, ()))}
            for n in comuns
        ])
        nomes = pick_images(ranking, args.n, available=comuns)
    print(f"\n  {len(comuns)} imagens em comum; {len(nomes)} escolhidas\n")

    out = results / (args.out or "galeria_modelos")
    for nome in nomes:
        gt = gt_by_image.get(nome, np.zeros((0, 4), np.float32))
        image = np.asarray(Image.open(p.train_images / nome).convert("RGB"))
        paineis = []
        for m in modelos:
            arm, policy, conf = cfgs[m]
            det = apply(raws[m][nome], policy, conf)
            paineis.append({"boxes": det.boxes, "scores": det.scores,
                            "titulo": f"{m} · {arm}"})
        caminho = galeria(nome, image, gt, paineis, match_iou, out)
        # Nome inteiro, e nao so o primeiro token: com os nomes descritivos, dois modelos
        # diferentes viravam ambos "yolo26n" no print e a linha ficava ilegivel.
        contagens = "  ".join(f"{m}:{len(pl['boxes']):>3}"
                              for m, pl in zip(modelos, paineis))
        print(f"  {nome:<32} GT {len(gt):>3} | {contagens}  -> {caminho.name}")
    print(f"\n  -> {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", nargs="*", default=None)
    parser.add_argument("--n", type=int, default=6, help="quantos casos, se --images não vier")
    parser.add_argument("--model", default=None,
                        help="pesos da Etapa 12, ex.: 'yolo26n_imgsz1280_90ep'. Sem isto, usa o baseline")
    parser.add_argument("--full-arm", default=None, help="braço do painel 2")
    parser.add_argument("--tiled-arm", default=None, help="braço do painel 3")
    parser.add_argument("--out", default=None, help="pasta de saída")
    parser.add_argument("--models", default=None,
                        help="galeria COMPARANDO PESOS: 'yolo26n_imgsz640_120ep,yolo26n_imgsz1280_90ep,yolo26s_imgsz1280_200ep_aug'. "
                             "Cada painel usa o braço e o limiar que a validação escolheu para "
                             "aquele modelo, que é o que o sistema realmente rodaria")
    args = parser.parse_args()

    if args.models:
        cfg, p = experiment(), paths()
        modo_galeria(args, cfg, p, p.resolve(p.results_root))
        return

    cfg, p = experiment(), paths()
    results = p.resolve(p.results_root)
    match_iou = cfg["counting"]["match_iou"]
    n_folds = cfg["data"]["n_folds"]

    full_arm = args.full_arm or FULL_ARM
    tiled_arm = args.tiled_arm or TILED_ARM
    tag = f"cmp-{args.model}" if args.model else "grouped"
    out = results / (args.out or ("audit" if not args.model else f"audit_{args.model}"))

    gt_by_image = {
        n: g[["x0", "y0", "x1", "y1"]].to_numpy(np.float32)
        for n, g in pd.read_csv(results / "instances.csv").groupby("image")
    }
    raw = {arm: load_predictions(results, arm, n_folds, tag) for arm in (full_arm, tiled_arm)}
    frozen = {
        arm: (load_frozen_model(results, args.model, arm) if args.model
              else load_frozen(results, arm))
        for arm in (full_arm, tiled_arm)
    }

    if args.images:
        names = args.images
    else:
        # Com `--model`, a faixa de erro tem que ser a do modelo AUDITADO. O `per_image.csv`
        # da Etapa 3 traz as contagens do baseline no ponto congelado dele: ranquear por ele
        # escolheria os casos extremos de outro sistema. Medido — as 6 imagens mudam todas.
        if args.model:
            pol, cf = frozen[full_arm]
            ranking = pd.DataFrame([
                {"image": n, "pred": apply(r, pol, cf).count,
                 "gt": len(gt_by_image.get(n, ()))}
                for n, r in raw[full_arm].items()
            ])
        else:
            ranking = pd.read_csv(results / "per_image.csv")
        names = pick_images(ranking, args.n, arm=full_arm,
                            available=set(raw[full_arm]) & set(raw[tiled_arm]))
    label = args.model or "yolo26n_imgsz640_120ep"
    print(f"auditando {len(names)} imagens | pesos {label} | {full_arm} vs {tiled_arm}")
    for arm, (policy, conf) in frozen.items():
        print(f"  {arm:<12} conf {conf:.2f} | fusão {policy.label}")
    print()

    for name in names:
        if name not in raw[full_arm] or name not in raw[tiled_arm]:
            print(f"  {name}: sem predição gravada, pulando")
            continue
        gt = gt_by_image.get(name, np.zeros((0, 4), np.float32))
        image = np.asarray(Image.open(p.train_images / name).convert("RGB"))
        preds = {}
        for arm in (full_arm, tiled_arm):
            policy, conf = frozen[arm]
            det = apply(raw[arm][name], policy, conf)
            preds[arm] = (det.boxes, det.scores)
        path = audit_one(name, image, gt, preds, match_iou, out, full_arm, tiled_arm)
        full, tiled = len(preds[full_arm][0]), len(preds[tiled_arm][0])
        print(f"  {name:<32} GT {len(gt):>3} | inteira {full:>3} | tiles {tiled:>3}"
              f"  -> {path.name}")

    print(f"\n  -> {out}")


if __name__ == "__main__":
    main()
