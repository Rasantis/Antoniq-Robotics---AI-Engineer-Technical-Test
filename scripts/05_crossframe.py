"""Etapa 6 (bônus): supressão de duplicatas entre quadros consecutivos.

Dois experimentos, com propósitos diferentes:

1. PASSADA VIRTUAL — ground truth exato, resposta quantitativa.
   Uma janela de altura cheia varre cada imagem rotulada na horizontal, com deslocamento
   conhecido. Cada posição vira um quadro, e como as anotações da imagem original são as
   mesmas em todas as posições, sabemos exatamente quais frutas se repetem. Mede-se a soma
   ingênua por quadro, a contagem após deduplicação e a contagem única verdadeira.
   Não há paralaxe nem motion blur — a limitação está declarada no relatório.

2. SEQUÊNCIA REAL — validação qualitativa, sem GT de identidade.
   As imagens rotuladas do MinneApple são quadros de vídeo, e o espaçamento mediano entre
   quadros rotulados vizinhos é de 5 quadros na maioria das sessões (medido na Etapa 1),
   o que a ~1 m/s equivale a uns 17 cm. Rodar o rastreador sobre esses trechos consecutivos
   mostra o comportamento em imagem real; o número reportado é o fator de compressão da
   contagem, não um erro contra ground truth que não existe.

Saídas em results/:
    crossframe_virtual.csv   uma linha por imagem: soma ingênua, deduplicada, verdade
    crossframe_real.csv      uma linha por trecho consecutivo real
    figures/08_crossframe.png

Uso:
    python scripts/05_crossframe.py
    python scripts/05_crossframe.py --limit 60      # subamostra, para iterar rápido
"""
from __future__ import annotations

import argparse
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

from src.sequence.crossframe import track_sequence  # noqa: E402
from src.sequence.virtual_pass import build_pass  # noqa: E402
from src.utils.config import experiment, paths  # noqa: E402
from src.utils.seed import set_seed  # noqa: E402
from src.viz import figures  # noqa: E402

MIN_RUN_LENGTH = 4  # quadros consecutivos mínimos para valer como sequência real
DETECTOR_CONF = 0.25  # ponto de operação quando a passada roda com o detector real


def virtual_experiment(
    p, cfg: dict, instances: pd.DataFrame, images: list[str], detector=None
) -> pd.DataFrame:
    """Passada virtual, medida de duas formas.

    Com ``detector=None`` o rastreador recebe as próprias ANOTAÇÕES no lugar de detecções.
    Isso isola o que se quer medir — a capacidade de converter observações repetidas numa
    contagem única — porque um resultado ruim não fica ambíguo entre falha de detecção e
    falha de deduplicação. O resíduo esperado nesse regime é zero.

    Com um detector de verdade, o número deixa de ser um teste do mecanismo e passa a ser o
    que o sistema entregaria: a deduplicação herda os falsos positivos e negativos do
    detector. As duas medidas vão para o relatório, porque reportar só a primeira seria
    apresentar um resultado de detector perfeito como se fosse desempenho de produto.
    """
    vp_cfg = cfg["sequence"]["virtual_pass"]
    match_iou = cfg["sequence"]["matcher"]["match_iou"]
    method = cfg["sequence"]["matcher"]["method"]
    by_image = dict(tuple(instances.groupby("image")))

    rows = []
    for name in tqdm(images, desc="  passada virtual", ncols=78):
        grp = by_image.get(name)
        if grp is None or len(grp) < 2:
            continue
        boxes = grp[["x0", "y0", "x1", "y1"]].to_numpy(dtype=np.float32)
        img = np.asarray(Image.open(p.train_images / name).convert("RGB"))

        vp = build_pass(
            img, boxes,
            window=tuple(vp_cfg["window"]),
            n_frames=vp_cfg["n_frames"],
            step_px=vp_cfg["step_px"],
            axis=vp_cfg.get("axis", "x"),
        )
        frame_images = [f.image for f in vp.frames]
        if detector is None:
            frame_boxes = [f.boxes for f in vp.frames]
            naive = vp.naive_sum
        else:
            predicted = detector.predict(frame_images, imgsz=cfg["train"]["imgsz"])
            frame_boxes = [
                boxes[scores >= DETECTOR_CONF] for boxes, scores in predicted
            ]
            naive = int(sum(len(b) for b in frame_boxes))

        result = track_sequence(
            frame_images, frame_boxes, method=method, match_iou=match_iou
        )
        rows.append({
            "image": name,
            "n_frames": len(vp.frames),
            "naive_sum": naive,
            "dedup_count": result.unique_count,
            "true_unique": vp.true_unique_count,
            "error": result.unique_count - vp.true_unique_count,
            "shift_dx": float(np.median([s[0] for s in result.shifts])) if result.shifts else 0.0,
        })
    return pd.DataFrame(rows)


def find_real_runs(images_df: pd.DataFrame, max_gap: int = 5) -> list[list[str]]:
    """Trechos de quadros rotulados consecutivos dentro de uma mesma sessão."""
    runs = []
    for _, grp in images_df.sort_values(["session", "frame"]).groupby("session"):
        current = [grp.iloc[0]["image"]]
        frames = grp["frame"].to_numpy()
        names = grp["image"].tolist()
        for i in range(1, len(names)):
            if frames[i] - frames[i - 1] <= max_gap:
                current.append(names[i])
            else:
                if len(current) >= MIN_RUN_LENGTH:
                    runs.append(current)
                current = [names[i]]
        if len(current) >= MIN_RUN_LENGTH:
            runs.append(current)
    return runs


def real_experiment(p, cfg: dict, instances: pd.DataFrame, runs: list[list[str]]) -> pd.DataFrame:
    """Rastreamento sobre trechos consecutivos reais, usando as anotações como detecções.

    Sem GT de identidade entre quadros, o número reportado é o fator de compressão: quantas
    observações viram quantas trilhas. Serve para verificar que a estimativa de movimento se
    comporta em imagem real, com paralaxe e mudança de luz — coisas que a passada virtual
    não tem.
    """
    match_iou = cfg["sequence"]["matcher"]["match_iou"]
    method = cfg["sequence"]["matcher"]["method"]
    by_image = dict(tuple(instances.groupby("image")))

    rows = []
    for run in tqdm(runs, desc="  sequências reais", ncols=78):
        frames, detections = [], []
        for name in run:
            grp = by_image.get(name)
            if grp is None:
                continue
            frames.append(np.asarray(Image.open(p.train_images / name).convert("RGB")))
            detections.append(grp[["x0", "y0", "x1", "y1"]].to_numpy(dtype=np.float32))
        if len(frames) < MIN_RUN_LENGTH:
            continue

        result = track_sequence(frames, detections, method=method, match_iou=match_iou)
        shifts = np.array(result.shifts) if result.shifts else np.zeros((1, 2))
        rows.append({
            "session": run[0].split("_image")[0],
            "n_frames": len(frames),
            "naive_sum": result.naive_sum,
            "dedup_count": result.unique_count,
            "suppression_rate": result.suppression_rate,
            "median_dx": float(np.median(shifts[:, 0])),
            "median_dy": float(np.median(shifts[:, 1])),
        })
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=None, help="subamostra de imagens")
    parser.add_argument(
        "--detector", default=None,
        help="caminho de pesos .pt; sem isto a passada usa as anotacoes como deteccoes",
    )
    args = parser.parse_args()

    cfg, p = experiment(), paths()
    set_seed(cfg["seed"])
    results = p.resolve(p.results_root)

    instances = pd.read_csv(results / "instances.csv")
    images_df = pd.read_csv(results / "images.csv")
    # Subamostragem ESTRATIFICADA por sessão. `images.csv` vem ordenado por sessão, então
    # um simples `[:limit]` pegava as N primeiras imagens da PRIMEIRA sessão — que por
    # acaso é a mais atípica do dataset (98,1 maçãs/imagem contra média de 42,1). O
    # resultado gravado descrevia uma sessão e era apresentado como o do dataset.
    if args.limit:
        per_session = max(1, args.limit // images_df["session"].nunique())
        names = (
            images_df.groupby("session", group_keys=False)
            .head(per_session)["image"].tolist()[: args.limit]
        )
    else:
        names = images_df["image"].tolist()

    detector = None
    if args.detector:
        from src.inference.engine import UltralyticsDetector

        detector = UltralyticsDetector(args.detector, conf=DETECTOR_CONF)
        print(f"[1/3] passada virtual com o DETECTOR ({Path(args.detector).name})")
    else:
        print("[1/3] passada virtual com as ANOTACOES como deteccoes (detector perfeito)")

    virtual = virtual_experiment(p, cfg, instances, names, detector)
    suffix = "_detector" if detector else ""
    virtual.to_csv(results / f"crossframe_virtual{suffix}.csv", index=False)

    naive, dedup, truth = (
        virtual["naive_sum"].sum(), virtual["dedup_count"].sum(), virtual["true_unique"].sum()
    )
    print(f"\n  imagens ................ {len(virtual)}")
    print(f"  soma ingênua ........... {naive:,}  ({naive / max(truth,1):.2f}x a verdade)")
    print(f"  após deduplicação ...... {dedup:,}")
    print(f"  contagem única real .... {truth:,}")
    print(f"  erro residual .......... {dedup - truth:+,}  "
          f"({(dedup - truth) / max(truth, 1):+.2%})")
    print(f"  deslocamento estimado .. {virtual['shift_dx'].median():.1f} px "
          f"(esperado {-cfg['sequence']['virtual_pass']['step_px']})")

    print("\n[2/3] sequências reais (quadros de vídeo consecutivos)")
    runs = find_real_runs(images_df)
    if args.limit:
        # Um trecho por sessão, na mesma lógica de estratificação.
        by_session: dict[str, list[str]] = {}
        for run in runs:
            by_session.setdefault(run[0].split("_image")[0], []).append(run)
        runs = [rs[0] for rs in by_session.values()]
    print(f"  {len(runs)} trechos com >= {MIN_RUN_LENGTH} quadros consecutivos "
          f"(gap <= 5 no índice do vídeo)")
    real = real_experiment(p, cfg, instances, runs)
    real.to_csv(results / "crossframe_real.csv", index=False)
    if len(real):
        print(f"  observações ............ {real['naive_sum'].sum():,}")
        print(f"  trilhas distintas ...... {real['dedup_count'].sum():,}")
        print(f"  taxa de supressão ...... {real['suppression_rate'].mean():.1%}")
        print(f"  deslocamento mediano ... dx={real['median_dx'].median():+.0f} px  "
              f"dy={real['median_dy'].median():+.0f} px")

    print("\n[3/3] figura")
    note = (
        "Deteccoes do modelo treinado; o residuo inclui os erros do detector."
        if detector else
        "Anotacoes usadas como deteccoes: isola o mecanismo de deduplicacao. "
        "Nao e desempenho de produto."
    )
    fig = figures.crossframe_counts(virtual, p.resolve(p.figures_root), note=note,
                                    name=f"08_crossframe{suffix}")
    print(f"  -> {fig}")


if __name__ == "__main__":
    main()
