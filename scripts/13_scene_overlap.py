"""Etapa 13: existe repetição de CENA entre sessões de captura diferentes?

A §2 agrupa o split por sessão, e a §8 declara que a granularidade honesta seria a
fileira — porque o teste oficial do MinneApple se chama `dataset1_front` / `dataset1_back`
e cada fileira é filmada dos dois lados. Se um par frente/trás compartilhasse pixels, agrupar
por sessão não bastaria: as mesmas maçãs estariam em treino e teste.

Este script mede isso em vez de afirmar. Para cada par de imagens de sessões diferentes,
casa descritores ORB e, quando há candidatos suficientes, estima uma homografia com RANSAC. Um
par que realmente mostre a mesma cena produz muitos inliers geometricamente consistentes; um
par de árvores parecidas em dias diferentes produz casamentos esparsos e incoerentes.

O par intra-sessão serve de controle positivo: quadros vizinhos do mesmo vídeo têm 17 cm de
deslocamento e precisam aparecer como repetição. Se o método não os detectar, ele não detecta
nada e o resultado negativo não vale.

    python scripts/13_scene_overlap.py              # amostra (rápido)
    python scripts/13_scene_overlap.py --full       # todos os pares entre sessões
"""
from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.utils import torch_first  # noqa: F401  isort:skip

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from tqdm import tqdm  # noqa: E402

from src.utils.config import paths  # noqa: E402
from src.utils.seed import set_seed  # noqa: E402

N_FEATURES = 800
RATIO = 0.75            # teste de razão de Lowe
MIN_MATCHES = 12        # abaixo disto nem vale chamar o RANSAC
INLIER_THRESHOLD = 20   # inliers geometricamente consistentes = mesma cena
DOWNSCALE = 2           # 720x1280 -> 360x640; ORB não precisa da resolução cheia aqui


def describe(path: Path, orb) -> np.ndarray | None:
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    img = cv2.resize(img, (img.shape[1] // DOWNSCALE, img.shape[0] // DOWNSCALE))
    kp, des = orb.detectAndCompute(img, None)
    if des is None or len(kp) < MIN_MATCHES:
        return None
    return np.array([k.pt for k in kp], np.float32), des


def inliers(a, b, matcher) -> int:
    """Nº de casamentos que sobrevivem à razão de Lowe E a uma homografia RANSAC."""
    (pts_a, des_a), (pts_b, des_b) = a, b
    pairs = matcher.knnMatch(des_a, des_b, k=2)
    good = [m for m, n in (p for p in pairs if len(p) == 2) if m.distance < RATIO * n.distance]
    if len(good) < MIN_MATCHES:
        return len(good)
    src = np.float32([pts_a[m.queryIdx] for m in good]).reshape(-1, 1, 2)
    dst = np.float32([pts_b[m.trainIdx] for m in good]).reshape(-1, 1, 2)
    _, mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
    return int(mask.sum()) if mask is not None else 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full", action="store_true", help="todos os pares entre sessões")
    parser.add_argument("--sample", type=int, default=8000, help="pares se não for --full")
    args = parser.parse_args()

    p = paths()
    set_seed(1337)
    results = p.resolve(p.results_root)
    images = sorted(f.name for f in p.train_images.glob("*.png"))
    session = {n: n.rsplit("_image", 1)[0] for n in images}

    orb = cv2.ORB_create(nfeatures=N_FEATURES)
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)

    print(f"descrevendo {len(images)} imagens com ORB ({N_FEATURES} features, /{DOWNSCALE})")
    feats = {}
    for n in tqdm(images, ncols=78):
        f = describe(p.train_images / n, orb)
        if f is not None:
            feats[n] = f
    print(f"  {len(feats)} imagens com descritores\n")

    usable = [n for n in images if n in feats]
    cross = [(a, b) for a, b in itertools.combinations(usable, 2) if session[a] != session[b]]
    within = [(a, b) for a, b in itertools.combinations(usable, 2) if session[a] == session[b]]
    print(f"pares ENTRE sessões: {len(cross):,}   |   DENTRO de sessão: {len(within):,}")

    rng = np.random.default_rng(1337)

    # --- controle positivo: quadros vizinhos do mesmo video TEM que aparecer como repeticao
    adjacent = [(a, b) for a, b in within
                if abs(int(a.rsplit("image", 1)[1][:-4]) - int(b.rsplit("image", 1)[1][:-4])) <= 5]
    ctrl = [adjacent[i] for i in rng.choice(len(adjacent), min(300, len(adjacent)), replace=False)]
    print(f"\ncontrole positivo: {len(ctrl)} pares de quadros vizinhos (mesma sessão, gap <= 5)")
    ctrl_in = [inliers(feats[a], feats[b], matcher) for a, b in tqdm(ctrl, ncols=78)]
    ctrl_in = np.array(ctrl_in)
    detected = int((ctrl_in >= INLIER_THRESHOLD).sum())
    # `max(..., 1)` aqui também: sem par de controle — um conjunto cujos quadros rotulados
    # estejam todos a mais de 5 de distância — a linha abaixo dividia por zero e derrubava o
    # script justamente onde ele deveria AVISAR que não há controle.
    print(f"  mediana {np.median(ctrl_in) if len(ctrl_in) else float('nan'):.0f} inliers | "
          f"{detected}/{len(ctrl)} ({detected/max(len(ctrl), 1):.0%}) acima do limiar "
          f"de {INLIER_THRESHOLD}")
    if detected / max(len(ctrl), 1) < 0.5:
        print("  AVISO: o metodo nao detecta nem repeticao conhecida — resultado negativo "
              "abaixo NAO tem valor.")

    # --- a pergunta
    amostra = rng.choice(len(cross), min(args.sample, len(cross)), replace=False)
    todo = cross if args.full else [cross[i] for i in amostra]
    print(f"\nvarrendo {len(todo):,} pares ENTRE sessões")
    rows = []
    for a, b in tqdm(todo, ncols=78):
        rows.append({"a": a, "b": b, "sessao_a": session[a], "sessao_b": session[b],
                     "inliers": inliers(feats[a], feats[b], matcher)})
    df = pd.DataFrame(rows)
    hits = df[df["inliers"] >= INLIER_THRESHOLD]

    out = results / "scene_overlap.csv"
    df.sort_values("inliers", ascending=False).head(200).to_csv(out, index=False)
    print(f"\n  pares testados ...... {len(df):,} de {len(cross):,} possíveis")
    print(f"  máximo de inliers ... {df['inliers'].max()}")
    print(f"  mediana ............. {df['inliers'].median():.0f}")
    print(f"  acima de {INLIER_THRESHOLD} ......... {len(hits)}  ({len(hits)/len(df):.3%})")
    print(f"  controle (vizinhos) . mediana {np.median(ctrl_in):.0f}, "
          f"{detected/len(ctrl):.0%} detectados")
    if len(hits):
        print("\n  pares suspeitos (maior contagem de inliers):")
        print(hits.nlargest(5, "inliers").to_string(index=False))
    print(f"\n  -> {out}")


if __name__ == "__main__":
    main()
