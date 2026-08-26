"""Etapa 2: treina um YOLO26n por fold, com validação cruzada agrupada.

São quatro treinos:

    grouped_fold0..2   os modelos de verdade, um por fold do GroupKFold sobre as sessões
    random_fold0       um único modelo com split aleatório por imagem

O quarto existe para medir o vazamento, não para ser usado. As 670 imagens são quadros de
vídeo espaçados de ~17 cm; um split aleatório coloca quadros quase idênticos em treino e em
teste. O relatório reporta os dois lado a lado, e a diferença é a evidência de que o split
agrupado era necessário.

Uso:
    python scripts/02_train_cv.py --smoke     # 2 épocas, só para validar o pipeline
    python scripts/02_train_cv.py             # completo
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

from src.utils.config import experiment, paths  # noqa: E402
from src.utils.seed import set_seed  # noqa: E402


def _epochs_done(run_path: Path) -> int:
    """Quantas épocas o run já completou, lido de results.csv."""
    results = run_path / "results.csv"
    if not results.exists():
        return 0
    import pandas as pd

    return len(pd.read_csv(results))


def checkpoint_retomavel(ck: dict) -> bool:
    """Um checkpoint só é retomável enquanto carrega o estado do treino.

    O Ultralytics grava `epoch: -1` e descarta o otimizador quando o run termina — inclusive
    quando termina cedo por `patience`. Sem os dois, `train(resume=True)` não tem de onde
    continuar; ele avisa e começa um treino novo com os PADRÕES da biblioteca, que é um
    desastre silencioso. Predicado separado da leitura do arquivo para poder ser testado.
    """
    return ck.get("optimizer") is not None and int(ck.get("epoch", -1)) >= 0


def _pode_retomar(last: Path) -> bool:
    import torch

    try:
        return checkpoint_retomavel(torch.load(last, map_location="cpu", weights_only=False))
    except Exception:
        return False


def _summarise(
    name: str, epochs: int, started: float, project: Path, run_dir: Path, cfg: dict
) -> dict:
    """Métricas de validação do melhor checkpoint, sem retreinar."""
    from ultralytics import YOLO

    weights = project / name / "weights" / "best.pt"
    metrics = YOLO(str(weights)).val(
        data=str(run_dir / "data.yaml"),
        split="val",
        project=str(project),
        name=f"{name}_val",
        exist_ok=True,
        verbose=False,
    )
    box = metrics.box
    return {
        "run": name,
        "epochs": _epochs_done(project / name),
        "epochs_requested": epochs,
        "minutes": round((time.perf_counter() - started) / 60, 1),
        "val_mAP50_95": float(box.map),
        "val_mAP50": float(box.map50),
        "val_mAP75": float(box.map75),
        "weights": str(weights),
    }


def train_one(
    run_dir: Path, cfg: dict, epochs: int, batch: int, project: Path,
    resume: bool = False, cache: str | bool | None = None,
) -> dict:
    """Treina um modelo e devolve as métricas de validação do melhor checkpoint.

    Com ``resume=True``, continua de ``last.pt`` se ele existir. Isso existe por necessidade
    prática: nesta máquina o treino foi interrompido duas vezes por pressão de recurso, e sem
    retomada cada interrupção custaria a corrida inteira. Também importa para a metodologia —
    um fold que parasse em 34 épocas enquanto os outros rodaram 80 e 98 daria orçamentos de
    treino desiguais entre folds, e a variação entre eles deixaria de ser comparável.
    """
    from ultralytics import YOLO

    train_cfg = cfg["train"]
    name = run_dir.name
    started = time.perf_counter()
    if cache is None:
        cache = train_cfg.get("cache", False)

    last = project / name / "weights" / "last.pt"
    if resume and last.exists():
        done = _epochs_done(project / name)
        if not _pode_retomar(last):
            # "Menos épocas que o pedido" NÃO significa interrompido: com `patience`, um run
            # que para na época 80 de 120 está COMPLETO. Testar `done < epochs` mandava esses
            # runs para o `resume`, e aí o Ultralytics via um checkpoint já finalizado (sem
            # estado do otimizador), avisava que não dá para retomar e — em vez de falhar —
            # **começava um treino novo com os padrões dele**: dataset `coco8`, 100 épocas,
            # gravando em `<cwd>/runs/detect/train`. Medido: dois dos três folds caíam nisso.
            print(f"  já concluído ({done} épocas), pulando o treino")
            return _summarise(name, epochs, started, project, run_dir, cfg)

        print(f"  retomando de {last.name} na época {done + 1}/{epochs}")
        YOLO(str(last)).train(resume=True)
        return _summarise(name, epochs, started, project, run_dir, cfg)

    model = YOLO(train_cfg["model"])
    model.train(
        data=str(run_dir / "data.yaml"),
        epochs=epochs,
        imgsz=train_cfg["imgsz"],
        batch=batch,
        patience=train_cfg["patience"],
        workers=train_cfg["workers"],
        cache=cache,
        deterministic=train_cfg["deterministic"],
        seed=cfg["seed"],
        project=str(project),
        name=name,
        exist_ok=True,
        val=True,
        plots=False,
        verbose=False,
    )

    # project/name explicitos: sem eles o Ultralytics escreve em <cwd>/runs/detect/val,
    # ou seja, dentro do repositorio.
    metrics = model.val(
        data=str(run_dir / "data.yaml"),
        split="val",
        project=str(project),
        name=f"{name}_val",
        exist_ok=True,
        verbose=False,
    )
    box = metrics.box
    return {
        "run": name,
        "epochs": _epochs_done(project / name),
        "epochs_requested": epochs,
        "minutes": round((time.perf_counter() - started) / 60, 1),
        "val_mAP50_95": float(box.map),
        "val_mAP50": float(box.map50),
        "val_mAP75": float(box.map75),
        "weights": str(project / name / "weights" / "best.pt"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true", help="2 épocas, valida o pipeline")
    parser.add_argument("--batch", type=int, default=None, help="sobrepõe o batch da config")
    parser.add_argument(
        "--only", default=None, help="treina só este run, ex.: grouped_fold0"
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="continua de last.pt quando o run foi interrompido",
    )
    parser.add_argument(
        "--cache", default=None, choices=["ram", "disk", "none"],
        help=(
            "sobrepoe o cache de dataset. 'ram' e o padrao e o mais rapido quando ha memoria "
            "sobrando; sob pressao de RAM ele faz a maquina paginar e o treino fica 4x mais "
            "lento, e ai 'disk' e melhor. Afeta apenas I/O, nao a matematica do treino."
        ),
    )
    args = parser.parse_args()

    cfg, p = experiment(), paths()
    set_seed(cfg["seed"])

    cache = {"none": False}.get(args.cache, args.cache)
    epochs = 2 if args.smoke else cfg["train"]["epochs"]
    batch = args.batch or cfg["train"]["batch"]
    project = p.runs_root / ("train_smoke" if args.smoke else "train")

    runs = [f"grouped_fold{i}" for i in range(cfg["data"]["n_folds"])] + ["random_fold0"]
    if args.only:
        runs = [r for r in runs if r == args.only] or [args.only]

    print(f"YOLO26n | imgsz={cfg['train']['imgsz']} batch={batch} epochs={epochs}")
    print(f"saída: {project}\n")

    results = []
    for run in runs:
        run_dir = p.runs_root / "folds" / run
        if not (run_dir / "data.yaml").exists():
            raise SystemExit(f"{run_dir / 'data.yaml'} nao existe. Rode 01_prepare_data.py.")

        print(f"--- {run} ---")
        try:
            results.append(
                train_one(run_dir, cfg, epochs, batch, project, args.resume, cache)
            )
        except RuntimeError as exc:
            if "out of memory" not in str(exc).lower():
                raise
            print(f"  OOM com batch={batch}; repetindo com batch={batch // 2}")
            results.append(
                train_one(run_dir, cfg, epochs, batch // 2, project, args.resume, cache)
            )
        print(f"  mAP50-95={results[-1]['val_mAP50_95']:.3f}  "
              f"mAP50={results[-1]['val_mAP50']:.3f}  "
              f"({results[-1]['minutes']} min)\n")

    out = p.resolve(p.results_root) / (
        "training_smoke.json" if args.smoke else "training.json"
    )
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")

    print("=" * 62)
    for r in results:
        print(f"  {r['run']:<16} ep={r['epochs']:>3}  mAP50-95={r['val_mAP50_95']:.3f}  "
              f"mAP50={r['val_mAP50']:.3f}  {r['minutes']:>5} min")
    print(f"\n  -> {out}")
    print("\nNOTA: estas métricas são de VALIDAÇÃO, usadas só para early stopping.")
    print("Os números do relatório saem do conjunto de teste, em scripts/03_eval_arms.py.")


if __name__ == "__main__":
    main()
