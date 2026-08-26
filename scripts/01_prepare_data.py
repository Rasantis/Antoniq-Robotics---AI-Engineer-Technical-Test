"""Etapa 1: validar o dataset, extrair caixas das máscaras e montar os splits agrupados.

Roda uma vez. Produz, em ``results/``:

    instances.csv      uma linha por fruta anotada, com área, solidez, densidade local
    images.csv         uma linha por imagem, com sessão, contagem e estatísticas de luz
    folds.csv          composição dos 3 folds agrupados
    frame_gaps.csv     espaçamento entre quadros rotulados dentro de cada sessão

e, no diretório do dataset, os rótulos em formato YOLO mais um ``data.yaml`` por fold.

Uso:
    python scripts/01_prepare_data.py
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
import yaml
from PIL import Image
from tqdm import tqdm

from src.data.minneapple import (  # noqa: E402
    distance_to_border,
    instances_from_mask,
    local_density,
    read_mask,
    solidity,
)
from src.data.splits import (  # noqa: E402
    assert_no_session_leak,
    assert_test_covers_all,
    make_folds,
    make_random_folds,
    session_table,
)
from src.utils.config import experiment, paths  # noqa: E402
from src.utils.seed import set_seed  # noqa: E402

EXPECTED_TRAIN_IMAGES = 670
EXPECTED_SESSIONS = 10
EXPECTED_TEST_IMAGES = 331


# --------------------------------------------------------------------------- validação

def validate_dataset(p) -> list[str]:
    """Confere que a extração manual produziu o que o resto do pipeline espera.

    O download é manual (o servidor da UMN bloqueia clientes automatizados), então uma
    extração parcial ou com um nível de pasta a mais é uma falha plausível. Melhor descobrir
    aqui, com uma mensagem clara, do que num erro obscuro três etapas adiante.
    """
    for directory in (p.train_images, p.train_masks):
        if not directory.is_dir():
            raise SystemExit(
                f"Diretório não encontrado: {directory}\n"
                f"Veja scripts/00_download.md. Estrutura esperada:\n"
                f"  {p.detection}\\train\\images\\*.png\n"
                f"  {p.detection}\\train\\masks\\*.png"
            )

    images = sorted(f.name for f in p.train_images.glob("*.png"))
    masks = {f.name for f in p.train_masks.glob("*.png")}

    if len(images) != EXPECTED_TRAIN_IMAGES:
        print(f"  AVISO: {len(images)} imagens de treino, esperadas {EXPECTED_TRAIN_IMAGES}")
    missing = [i for i in images if i not in masks]
    if missing:
        raise SystemExit(f"{len(missing)} imagens sem máscara, ex.: {missing[:3]}")

    expected_wh = tuple(experiment()["data"]["image_size"])
    with Image.open(p.train_images / images[0]) as img:
        if img.size != expected_wh:
            raise SystemExit(
                f"Resolução inesperada {img.size}; esperado {expected_wh} (largura x altura). "
                f"O MinneApple é RETRATO, 720x1280 — o artigo escreve '1280 x 720', que é "
                f"altura x largura. Confira também se o download não é um mirror reprocessado."
            )

    n_test = len(list(p.test_images.glob("*.png"))) if p.test_images.is_dir() else 0
    print(f"  {len(images)} imagens de treino pareadas com máscara, "
          f"{expected_wh[0]}x{expected_wh[1]}")
    print(f"  {n_test} imagens de teste oficial (sem rótulos públicos)")
    return images


# ----------------------------------------------------------------- extração das instâncias

def build_tables(p, images: list[str], cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Tabela por instância e tabela por imagem."""
    min_side = cfg["data"]["min_box_side"]
    image_wh = tuple(cfg["data"]["image_size"])
    rows, image_rows = [], []

    for name in tqdm(images, desc="  máscaras", ncols=78):
        inst = instances_from_mask(read_mask(p.train_masks / name), min_side=min_side)
        sol = solidity(inst.boxes, inst.mask_areas)
        dens = local_density(inst.boxes)
        dist = distance_to_border(inst.boxes, image_wh)

        # Estatísticas de luz da imagem: alimentam o modo de erro "iluminação".
        with Image.open(p.train_images / name) as img:
            hsv = np.asarray(img.convert("HSV"), dtype=np.float32)
        image_rows.append(
            {
                "image": name,
                "n_apples": len(inst),
                "mean_value": float(hsv[..., 2].mean()),
                "mean_saturation": float(hsv[..., 1].mean()),
            }
        )

        wh = inst.boxes[:, 2:] - inst.boxes[:, :2]
        for i in range(len(inst)):
            # Brilho DENTRO da caixa, não o da imagem. Os dois divergem muito e é este que
            # separa as sessões: o V por imagem vai de 100 a 143 entre as dez, enquanto o V
            # dentro da fruta vai de 54 a 166 — e é essa faixa que decide se a augmentação de
            # brilho do treino alcança as sessões que o modelo erra (ver 11_experiments.py).
            x0, y0 = max(int(inst.boxes[i, 0]), 0), max(int(inst.boxes[i, 1]), 0)
            recorte = hsv[y0:int(inst.boxes[i, 3]), x0:int(inst.boxes[i, 2]), 2]
            rows.append(
                {
                    "image": name,
                    "instance_id": int(inst.instance_ids[i]),
                    "x0": float(inst.boxes[i, 0]), "y0": float(inst.boxes[i, 1]),
                    "x1": float(inst.boxes[i, 2]), "y1": float(inst.boxes[i, 3]),
                    "w": float(wh[i, 0]), "h": float(wh[i, 1]),
                    "bbox_area": float(wh[i, 0] * wh[i, 1]),
                    "mask_area": int(inst.mask_areas[i]),
                    "solidity": float(sol[i]),
                    "local_density": int(dens[i]),
                    "dist_to_border": float(dist[i]),
                    "box_value": float(recorte.mean()) if recorte.size else float("nan"),
                }
            )

    instances = pd.DataFrame(rows)
    sessions = session_table(images)
    images_df = sessions.merge(pd.DataFrame(image_rows), on="image")
    instances = instances.merge(sessions[["image", "session", "frame"]], on="image")
    return instances, images_df


def report_diagnostics(instances: pd.DataFrame, images_df: pd.DataFrame) -> None:
    """Diagnósticos que decidem escolhas do plano, impressos com números."""
    print("\n  --- instâncias ---")
    print(f"  total anotado ............ {len(instances):,}")
    print(f"  por imagem ............... media {images_df.n_apples.mean():.1f}  "
          f"mediana {images_df.n_apples.median():.0f}  max {images_df.n_apples.max()}")
    q = instances.bbox_area.quantile([0.05, 0.5, 0.95])
    print(f"  area da caixa (px^2) ..... p05 {q[0.05]:.0f}  mediana {q[0.5]:.0f}  "
          f"p95 {q[0.95]:.0f}")
    print(f"  lado medio ............... {np.sqrt(instances.bbox_area).mean():.1f} px")

    # Cortes COCO: small < 32^2, medium < 96^2.
    small = (instances.bbox_area < 32**2).mean()
    medium = ((instances.bbox_area >= 32**2) & (instances.bbox_area < 96**2)).mean()
    print(f"  COCO small/medium/large .. {small:.1%} / {medium:.1%} / {1 - small - medium:.1%}")

    # Troncos de árvore foram anotados no artigo; se estiverem nas máscaras liberadas,
    # apareceriam como instâncias enormes e muito alongadas.
    wide = instances[(instances.bbox_area > 96**2) & (instances.h / instances.w > 3)]
    print(f"  candidatas a tronco (grandes e alongadas): {len(wide)}"
          f"{'  <- inspecionar' if len(wide) > 20 else '  (nenhuma classe extra)'}")

    print(f"  solidez (proxy de oclusao) media {instances.solidity.mean():.3f} "
          f"(disco perfeito = 0,785)")
    occluded = (instances.solidity < 0.6).mean()
    print(f"  fracao com solidez < 0,60 (ocluida) ... {occluded:.1%}")

    print("\n  --- luz por sessao ---")
    per_session = images_df.groupby("session").agg(
        n=("image", "size"), brilho=("mean_value", "mean"), frutas=("n_apples", "mean")
    )
    print(per_session.round(1).to_string())


def frame_gap_analysis(images_df: pd.DataFrame) -> pd.DataFrame:
    """Espaçamento entre quadros rotulados consecutivos, por sessão.

    Decide se o bônus de dedup cross-frame pode usar sequências reais. O artigo diz que as
    670 foram "randomly selected" do conjunto de quadros extraídos; se o espaçamento típico
    for grande, quadros rotulados vizinhos quase não se sobrepõem e a passada virtual (com
    deslocamento conhecido e ground truth exato) vira o experimento principal.
    """
    rows = []
    for session, grp in images_df.sort_values("frame").groupby("session"):
        gaps = np.diff(grp.frame.to_numpy())
        rows.append(
            {
                "session": session,
                "n_images": len(grp),
                "frame_min": int(grp.frame.min()),
                "frame_max": int(grp.frame.max()),
                "gap_median": float(np.median(gaps)) if len(gaps) else np.nan,
                "gap_min": int(gaps.min()) if len(gaps) else -1,
                "frac_gap_le_5": float((gaps <= 5).mean()) if len(gaps) else 0.0,
            }
        )
    table = pd.DataFrame(rows)
    print("\n  --- espacamento entre quadros rotulados ---")
    print(table.round(2).to_string(index=False))
    consecutive = table.frac_gap_le_5.mean()
    print(f"  fracao de pares com gap <= 5 quadros: {consecutive:.1%}")
    print("  -> sequencias reais utilizaveis" if consecutive > 0.5
          else "  -> quadros esparsos; a passada virtual e o experimento principal do bonus")
    return table


# ------------------------------------------------------------------------- export YOLO

def write_yolo_labels(p, instances: pd.DataFrame, images: list[str], cfg: dict) -> Path:
    """Rótulos YOLO (uma pasta só; os folds são listas de arquivos que apontam para cá)."""
    w, h = cfg["data"]["image_size"]
    label_dir = p.train_images.parent / "labels"
    label_dir.mkdir(exist_ok=True)

    grouped = dict(tuple(instances.groupby("image")))
    for name in images:
        lines = []
        grp = grouped.get(name)
        if grp is not None:
            cx = ((grp.x0 + grp.x1) / 2 / w).to_numpy()
            cy = ((grp.y0 + grp.y1) / 2 / h).to_numpy()
            bw = (grp.w / w).to_numpy()
            bh = (grp.h / h).to_numpy()
            lines = [f"0 {a:.6f} {b:.6f} {c:.6f} {d:.6f}" for a, b, c, d in zip(cx, cy, bw, bh)]
        (label_dir / f"{Path(name).stem}.txt").write_text("\n".join(lines), encoding="utf-8")
    print(f"\n  rótulos YOLO escritos em {label_dir}")
    return label_dir


def write_fold_configs(p, folds, cfg: dict, tag: str) -> None:
    """Um data.yaml + listas de arquivos por fold, no formato que o Ultralytics consome."""
    for fold in folds:
        out = p.runs_root / "folds" / f"{tag}_fold{fold.index}"
        out.mkdir(parents=True, exist_ok=True)
        for split, names in (("train", fold.train), ("val", fold.val), ("test", fold.test)):
            (out / f"{split}.txt").write_text(
                "\n".join(str(p.train_images / n) for n in names), encoding="utf-8"
            )
        (out / "data.yaml").write_text(
            yaml.safe_dump(
                {
                    "path": str(out),
                    "train": "train.txt",
                    "val": "val.txt",
                    "test": "test.txt",
                    "names": {0: cfg["data"]["class_names"][0]},
                },
                sort_keys=False,
            ),
            encoding="utf-8",
        )


# ------------------------------------------------------------------------------- main

def main() -> None:
    # Parser sem opcao nenhuma, de proposito: este script nao tem parametro. Ele existe para
    # que `--help` mostre a docstring em vez de disparar 33 s de preparacao em silencio, e
    # para que um argumento digitado errado seja recusado em vez de ignorado.
    argparse.ArgumentParser(description=__doc__).parse_args()

    cfg = experiment()
    p = paths()
    set_seed(cfg["seed"])
    results = p.resolve(p.results_root)
    results.mkdir(parents=True, exist_ok=True)

    print("[1/5] validando o dataset")
    images = validate_dataset(p)

    print("\n[2/5] extraindo instancias das mascaras")
    instances, images_df = build_tables(p, images, cfg)
    report_diagnostics(instances, images_df)
    gaps = frame_gap_analysis(images_df)

    print("\n[3/5] montando os splits agrupados")
    folds = make_folds(images, n_folds=cfg["data"]["n_folds"])
    assert_no_session_leak(folds)
    assert_test_covers_all(folds, images)
    random_folds = make_random_folds(images, n_folds=cfg["data"]["n_folds"], seed=cfg["seed"])
    fold_table = pd.DataFrame(
        [{**f.as_record(), "scheme": "grouped"} for f in folds]
        + [{**f.as_record(), "scheme": "random"} for f in random_folds]
    )
    print(fold_table[["scheme", "fold", "n_train", "n_val", "n_test"]].to_string(index=False))
    print("  sem vazamento de sessao: OK | cada imagem testada uma unica vez: OK")

    print("\n[4/5] exportando rotulos YOLO e configs de fold")
    write_yolo_labels(p, instances, images, cfg)
    write_fold_configs(p, folds, cfg, tag="grouped")
    write_fold_configs(p, random_folds, cfg, tag="random")

    print("\n[5/5] gravando tabelas")
    instances.to_csv(results / "instances.csv", index=False)
    images_df.to_csv(results / "images.csv", index=False)
    fold_table.to_csv(results / "folds.csv", index=False)
    gaps.to_csv(results / "frame_gaps.csv", index=False)
    (results / "dataset_summary.json").write_text(
        json.dumps(
            {
                "n_images": len(images),
                "n_sessions": int(images_df.session.nunique()),
                "n_instances": len(instances),
                "mean_apples_per_image": float(images_df.n_apples.mean()),
                "median_box_area_px2": float(instances.bbox_area.median()),
                "frac_coco_small": float((instances.bbox_area < 32**2).mean()),
                "mean_solidity": float(instances.solidity.mean()),
                # Brilho medio DENTRO das caixas, por sessao. Fica aqui, num artefato
                # versionado, porque e o numero que justifica a augmentacao anti-deriva do
                # 11_experiments.py — e um numero afirmado sem artefato e um numero que
                # ninguem consegue conferir.
                "mean_box_value_by_session": {
                    s: round(float(v), 1) for s, v in
                    instances.groupby("session")["box_value"].mean().sort_values().items()
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"  -> {results}")


if __name__ == "__main__":
    main()
