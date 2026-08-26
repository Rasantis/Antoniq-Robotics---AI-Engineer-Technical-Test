"""Inferência ponta a ponta numa imagem: recorte -> detecção -> fusão -> contagem.

É o script que o enunciado pede para reexecutar o pipeline. Ele não depende de nenhuma outra
etapa: dados os pesos e uma imagem, produz a contagem e um recorte anotado.

    python scripts/run_inference.py --image caminho/da/imagem.png
    python scripts/run_inference.py --image pasta/ --arm B_full1280 --csv saida.csv
    python scripts/run_inference.py --image img.png --compare      # os quatro braços

Sem `--weights`, usa o melhor checkpoint do fold 0 e, se não houver, o que vem versionado em
`models/`. Sem `--arm` e sem `--conf`, o ponto de operação vem calibrado: para os checkpoints
de `models/` é o par (braço, limiar) que a validação DAQUELES pesos elegeu na Etapa 12; para
os demais, o ponto congelado de results/operating_point.json. Sem nenhum dos dois arquivos,
cai no padrão da configuração e avisa.
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
from PIL import Image

from src.inference.engine import Arm, UltralyticsDetector, arms_from_config, run_arm  # noqa: E402
from src.inference.postprocess import MergePolicy, apply  # noqa: E402
from src.utils.config import experiment, paths  # noqa: E402

FALLBACK_CONF = 0.25
DEFAULT_ARM = "C_tile640"
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp"}


def resolve_weights(p, given: str | None) -> Path:
    """Pesos explícitos, senão os treinados localmente, senão o checkpoint que vem no repo.

    O último caso existe para que um clone limpo rode inferência sem antes baixar 1,9 GB de
    dataset e treinar. É o melhor da §6 do relatório: YOLO26s a 1280 px com a augmentação
    anti-deriva — MAE 12,6 no teste do fold 0.
    """
    if given:
        path = Path(given)
        if not path.exists():
            raise SystemExit(f"Pesos nao encontrados: {path}")
        return path

    treinado = p.runs_root / "train" / "grouped_fold0" / "weights" / "best.pt"
    if treinado.exists():
        return treinado

    embarcado = Path(__file__).resolve().parents[1] / "models" / "yolo26s_imgsz1280_200ep_aug.pt"
    if embarcado.exists():
        print(f"  (sem pesos treinados em {treinado} — usando o checkpoint do repositório)")
        return embarcado

    raise SystemExit(
        f"Nenhum peso em {treinado} nem em {embarcado}.\n"
        f"Rode scripts/02_train_cv.py ou passe --weights."
    )


def ponto_do_modelo(p, weights: Path) -> tuple[str, float] | None:
    """Braço e limiar que a validação escolheu para ESTES pesos, se forem os da §6.

    O `operating_point.json` é do pipeline com os pesos do baseline. Aplicá-lo a um dos
    checkpoints de `models/` daria um híbrido que nunca foi medido — e era exatamente o que
    acontecia com quem clonasse o repositório sem treinar: recebia o melhor modelo rodando no
    braço e no limiar de outro, e uma contagem que não corresponde a nenhuma linha da tabela.
    """
    linha = p.resolve(p.results_root) / "model_comparison.csv"
    if weights.parent.name != "models" or not linha.exists():
        return None

    import pandas as pd

    tabela = pd.read_csv(linha)
    do_modelo = tabela[tabela["modelo"] == weights.stem]
    if do_modelo.empty:
        return None
    # O braço é o que a validação DELE elegeu — o mesmo critério da Etapa 12.
    melhor = do_modelo.loc[do_modelo["val_MAE"].idxmin()]
    return str(melhor["arm"]), float(melhor["conf"])


def resolve_operating_point(p, arm: str, given: float | None, conf_modelo: float | None = None):
    """Limiar E política de fusão do ponto congelado — os dois, ou nenhum.

    Uma versão anterior lia só o limiar de `operating_point.json` e pegava a política de fusão
    do `experiment.yaml`. O resultado era um híbrido que nunca foi avaliado: o limiar calibrado
    de um braço aplicado com a política padrão de outro. O ponto de operação é o par.

    A política de fusão vem sempre do braço: ela é propriedade do pipeline, não dos pesos. O
    que muda de um treino para outro é o limiar, porque a distribuição de scores se desloca —
    daí `conf_modelo`, que a Etapa 12 mediu para cada checkpoint de `models/`.
    """
    frozen = p.resolve(p.results_root) / "operating_point.json"
    if frozen.exists():
        data = json.loads(frozen.read_text(encoding="utf-8"))
        spec = data["per_arm"].get(arm, data)
        policy = MergePolicy(
            metric=spec["metric"], policy=spec["merge_policy"],
            threshold=spec["threshold"], drop_truncated=spec["drop_truncated"],
        )
        if given is not None:
            return given, policy, "informado na linha de comando"
        if conf_modelo is not None:
            return conf_modelo, policy, "calibrado na validação PARA ESTES pesos (§6)"
        return (float(spec["conf"]), policy,
                f"calibrado na validação ({data.get('criterion', 'menor MAE')})")

    return (
        given if given is not None else FALLBACK_CONF,
        MergePolicy.from_config(experiment()),
        "padrão — rode 03_eval_arms.py para calibrar",
    )


def collect_images(target: Path) -> list[Path]:
    if target.is_dir():
        return sorted(f for f in target.iterdir() if f.suffix.lower() in IMAGE_SUFFIXES)
    if not target.exists():
        raise SystemExit(f"Imagem nao encontrada: {target}")
    return [target]


def annotate(image: np.ndarray, boxes: np.ndarray, out_path: Path) -> None:
    """Grava a imagem com as caixas finais desenhadas. OpenCV, como o resto do pipeline."""
    import cv2

    canvas = cv2.cvtColor(image, cv2.COLOR_RGB2BGR).copy()
    for x0, y0, x1, y1 in boxes.astype(int):
        cv2.rectangle(canvas, (x0, y0), (x1, y1), (0, 131, 0), 2)
    cv2.putText(canvas, f"{len(boxes)} macas", (12, 34),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(canvas, f"{len(boxes)} macas", (12, 34),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), canvas)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True, help="arquivo de imagem ou pasta")
    parser.add_argument("--weights", default=None, help="checkpoint .pt")
    parser.add_argument("--arm", default=None,
                        help=f"estratégia de inferência (padrão: {DEFAULT_ARM}, ou o braço que "
                             "a validação elegeu para os pesos, quando forem os de models/)")
    parser.add_argument("--conf", type=float, default=None, help="limiar de confiança")
    parser.add_argument("--compare", action="store_true", help="roda os quatro braços")
    parser.add_argument("--out", default="results/inference", help="pasta de saída")
    parser.add_argument("--csv", default=None, help="grava as contagens num CSV")
    parser.add_argument("--no-image", action="store_true", help="não grava a anotada")
    args = parser.parse_args()

    cfg, p = experiment(), paths()
    weights = resolve_weights(p, args.weights)
    out_dir = p.resolve(Path(args.out))

    # Braço e limiar próprios destes pesos, quando forem um dos checkpoints da §6.
    do_modelo = ponto_do_modelo(p, weights)
    arm_padrao, conf_modelo = do_modelo if do_modelo else (DEFAULT_ARM, None)
    arm_escolhido = args.arm or arm_padrao

    all_arms = {a.name: a for a in arms_from_config(cfg)}
    if arm_escolhido not in all_arms and not args.compare:
        raise SystemExit(f"Braço desconhecido: {arm_escolhido}. Opções: {sorted(all_arms)}")
    arms: list[Arm] = list(all_arms.values()) if args.compare else [all_arms[arm_escolhido]]

    # `--compare` roda os quatro braços; o limiar medido vale só para o braço eleito, então
    # aplicá-lo aos outros três seria justamente o híbrido que este código evita.
    def ponto(arm_name: str):
        so_deste = conf_modelo if arm_name == arm_padrao else None
        return resolve_operating_point(p, arm_name, args.conf, so_deste)

    images = collect_images(Path(args.image))
    print(f"pesos ....... {weights}")
    print(f"imagens ..... {len(images)}")
    for arm in arms:
        conf, policy, origin = ponto(arm.name)
        print(f"  {arm.name:<12} conf {conf:.2f} | fusão {policy.label}   ({origin})")
    print()

    # Piso fixo: o detector entrega tudo acima de 0,01 e o limiar de cada braço é aplicado
    # depois, em `apply`. Assim os quatro braços podem ter limiares diferentes numa só passada.
    detector = UltralyticsDetector(str(weights), conf=0.01)
    rows = []

    for image_path in images:
        image = np.asarray(Image.open(image_path).convert("RGB"))
        for arm in arms:
            conf, policy, _ = ponto(arm.name)
            started = time.perf_counter()
            raw = run_arm(detector, image, arm)
            det = apply(raw, policy, conf)
            elapsed = (time.perf_counter() - started) * 1e3

            rows.append({
                "image": image_path.name, "arm": arm.name, "count": det.count,
                "raw_detections": det.n_raw, "duplicates_removed": det.duplicates_removed,
                "dropped_border": det.n_dropped_border, "tiles": raw.n_tiles,
                "latency_ms": round(elapsed, 1),
            })
            print(f"  {image_path.name}  [{arm.name}]  {det.count:>4} maçãs   "
                  f"({raw.n_tiles} tile(s), {det.n_raw} brutas, "
                  f"-{det.duplicates_removed} duplicatas, -{det.n_dropped_border} borda, "
                  f"{elapsed:.0f} ms)")

            if not args.no_image:
                suffix = f"_{arm.name}" if len(arms) > 1 else ""
                annotate(image, det.boxes, out_dir / f"{image_path.stem}{suffix}.png")

    if args.csv:
        import pandas as pd

        path = p.resolve(Path(args.csv))
        path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(path, index=False)
        print(f"\n  -> {path}")
    if not args.no_image:
        print(f"  -> {out_dir}")


if __name__ == "__main__":
    main()
