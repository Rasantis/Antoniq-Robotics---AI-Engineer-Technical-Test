"""Etapa 11: fila de experimentos de treino, sequencial e retomável.

Existe para responder às perguntas que a §4 do relatório deixa em aberto, e não para caçar
mAP — o critério de avaliação do teste não pontua acurácia. Cada experimento converte uma
ressalva declarada num número medido:

    yolo26n_imgsz1280_90ep     A maçã chega à rede com 13,6 px quando se treina a 640 sobre a
                          imagem inteira. Dobrar a resolução de entrada dobra isso. Se o ganho
                          for grande, o gargalo é resolução; se for pequeno, não é.

    yolo26n_tiles320_15ep      Slicing-aided fine-tuning. Hoje a conclusão "o tiling não compensa"
                          vale só para pesos treinados em imagem inteira — o braço de tiles roda
                          4x fora da escala de treino. Treinando em recortes de 320 px, a
                          comparação entre estratégias passa a ser legítima em vez de
                          ressalvada. É o experimento de maior retorno da fila.

    yolo11s_imgsz640_120ep     Capacidade: 9,4 M parâmetros contra 2,4 M do YOLO26n. A análise de
                          erro aponta domínio e localização como gargalo, não capacidade, então
                          a expectativa é de ganho pequeno. Um resultado negativo também
                          informa, e barato.

Todos rodam sobre o fold 0, com a mesma paciência e semente do baseline, para que a
comparação seja contra o modelo que já existe. Rodar os três folds triplicaria o custo sem
mudar a leitura relativa.

O orçamento de épocas é POR experimento, e não igual para todos, porque igualar épocas não
iguala treino: o dataset de tiles tem 3.692 imagens contra 364 do fold, então uma época ali
vale dez. Quinze épocas de tiles equivalem a ~152 mil amostras vistas contra ~44 mil das 120
épocas do baseline — já é 3,5x mais atualização de gradiente, e 120 épocas custariam 23 horas
de GPU sem ganho proporcional.

A fila é sequencial de propósito: são todos GPU-bound e disputar a placa só atrasa os dois.
Cada experimento é retomável, então uma queda custa um experimento e não a noite.

Uso:
    python scripts/11_experiments.py --list
    python scripts/11_experiments.py                  # roda a fila inteira
    python scripts/11_experiments.py --only tiles320     # casa por subcadeia
    python scripts/11_experiments.py --only yolo26s      # os dois modelos s
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
import yaml
from PIL import Image
from tqdm import tqdm

from src.tiling.remap import to_local  # noqa: E402
from src.tiling.slicer import tile_grid  # noqa: E402
from src.utils.config import experiment, paths  # noqa: E402
from src.utils.seed import set_seed  # noqa: E402

FOLD = 0
MIN_VISIBLE = 0.35  # fração da caixa que precisa sobrar no tile para o rótulo valer

# Augmentacao apontada pela analise de erro, e o numero e medido sobre as 28.182 instancias:
# o V medio (HSV) DENTRO das caixas anotadas da 98,2 nas seis sessoes de treino do fold 0 — de
# 54 a 129 por sessao — e 164,2 e 166,3 nas duas sessoes de 19/09, que sao as que o modelo mais
# erra. O `hsv_v` do Ultralytics multiplica V por um ganho em [1-g, 1+g], entao o default
# g=0,4 leva a media de treino no maximo a 98,2 x 1,40 = 137: nao alcanca 164. Com g=0,75 vai a
# 98,2 x 1,75 = 172, e alcanca. `scale` desce de 0,5 para 0,35 porque a maca mediana tem 27 px
# e reduzi-la mais a torna inaprendivel.
# (Nao confundir com o brilho por IMAGEM de results/images.csv, que da 100 a 143 por sessao —
#  aquele inclui ceu e folhagem; este e so o pixel da fruta.)
AUG_DERIVA = {"hsv_v": 0.75, "hsv_s": 0.90, "scale": 0.35, "close_mosaic": 30, "degrees": 0.0}

EXPERIMENTS = {
    "yolo26n_imgsz1280_90ep": {
        "descricao": "YOLO26n a 1280 px — testa se o gargalo e resolucao de entrada",
        "model": "yolo26n.pt", "imgsz": 1280, "batch": 4, "dataset": "full", "epochs": 90,
    },
    "yolo26n_tiles320_15ep": {
        "descricao": "YOLO26n treinado em recortes de 320 px — slicing-aided fine-tuning",
        # 15 epocas sobre 3.692 tiles = ~152 mil amostras vistas, contra ~44 mil das 120
        # epocas do baseline sobre 364 imagens. Ja e 3,5x mais atualizacoes de gradiente;
        # 120 epocas aqui seriam 23 horas de GPU sem ganho proporcional.
        "model": "yolo26n.pt", "imgsz": 640, "batch": 16, "dataset": "tiles", "epochs": 15,
    },
    "yolo11s_imgsz640_120ep": {
        "descricao": "YOLO11s a 640 px — capacidade (9,4 M contra 2,5 M parametros)",
        "model": "yolo11s.pt", "imgsz": 640, "batch": 8, "dataset": "full", "epochs": 120,
    },
    # A celula que o yolo11s nao respondeu. Ele testou capacidade a 640 px, onde a maca chega a
    # rede com 13,6 px — nao havia sinal para um modelo maior usar, e ele empatou. A 1280 px
    # ela chega com 27 px, e a pergunta e outra.
    #
    # Batch 3 e nao 4: um forward+backward sintetico do yolo26s a 1280 deu pico de 5,12 GB
    # com batch 4, de 6 GB disponiveis. A perda real do Ultralytics aloca mais, e um OOM na
    # decima hora custa mais que um batch menor.
    #
    # 200 epocas, e nao 90 como o n@1280: a melhor epoca dele foi a 83 de 90, ainda subindo.
    # Como o results.csv registra todas, o n@1280 na epoca 90 contra este na epoca 90 da a
    # comparacao de capacidade com orcamento igualado — e o que vier depois mede o que as
    # epocas extras compram. Um treino, duas perguntas.
    "yolo26s_imgsz1280_200ep": {
        "descricao": "YOLO26s a 1280 px — capacidade em ALTA resolucao (9,9 M contra 2,5 M)",
        "model": "yolo26s.pt", "imgsz": 1280, "batch": 3, "dataset": "full", "epochs": 200,
    },
    # O par controlado do yolo26m que roda no Colab. Aquele usa a augmentacao anti-deriva; o
    # o par sem aug usou o padrao. Comparar os dois mediria capacidade E augmentacao somadas — o mesmo
    # defeito do yolo11s, que trocou arquitetura, cabeca, atribuicao de rotulo e pre-treino de uma
    # vez e por isso nao pode ser lido. Este experimento iguala a augmentacao para que a
    # diferenca s -> m seja so capacidade.
    "yolo26s_imgsz1280_200ep_aug": {
        "descricao": "YOLO26s a 1280 px COM a augmentacao anti-deriva — par controlado do m",
        "model": "yolo26s.pt", "imgsz": 1280, "batch": 3, "dataset": "full", "epochs": 200,
        "aug": AUG_DERIVA,
    },
    # O n@1280 parou em 90 epocas com a melhor na 83, ainda subindo. Este mede o que as epocas
    # extras compram — e a resposta so aparece no teste: o mAP de validacao fica parado
    # (0,4478 contra 0,4480) enquanto a MAE de contagem cai 22,8%.
    #
    # RESSALVA: o checkpoint que vai em models/ nao saiu deste batch. Ele foi treinado numa
    # Tesla T4 de 16 GB com batch 10, porque a fila local estava ocupada pelos dois yolo26s (ver
    # environment.md). O `batch: 4` abaixo e a receita que CABE nos 6 GB desta maquina; quem
    # rodar aqui reproduz o experimento, nao o arquivo binario. `train_args` dentro do .pt
    # guarda o batch real de cada um.
    "yolo26n_imgsz1280_300ep": {
        "descricao": "YOLO26n a 1280 px, 300 epocas — o que o orcamento de epocas compra",
        "model": "yolo26n.pt", "imgsz": 1280, "batch": 4, "dataset": "full", "epochs": 300,
    },
    # O degrau seguinte da escada de capacidade. NAO cabe nos 6 GB desta maquina: batch 3 a
    # 1280 px pede ~10 GB, entao este tambem foi treinado na T4 de 16 GB.
    "yolo26m_imgsz1280_300ep_aug": {
        "descricao": "YOLO26m a 1280 px (21,8 M) — o degrau acima do s na escada de capacidade",
        "model": "yolo26m.pt", "imgsz": 1280, "batch": 3, "dataset": "full", "epochs": 300,
        "aug": AUG_DERIVA,
    },
}


# ------------------------------------------------- dataset de recortes (yolo26n_tiles320)

def build_tiled_dataset(p, cfg: dict, split_files: dict[str, list[str]], out: Path) -> Path:
    """Recorta as imagens do fold em tiles de 320 px e reescreve os rótulos em coordenadas locais.

    Só entram tiles com pelo menos uma fruta: um tile de céu ou de grama não ensina nada sobre
    maçã e diluiria o treino. Caixas cortadas pela aresta entram se ao menos 35% da área
    sobreviver — abaixo disso o recorte mostra uma lasca que nem um humano rotularia.
    """
    tile = cfg["arms"]["D_tile320"]["tile"]
    overlap = cfg["arms"]["D_tile320"]["overlap"]
    image_wh = tuple(cfg["data"]["image_size"])
    grid = tile_grid(image_wh, tile, overlap)
    label_dir = p.train_images.parent / "labels"

    counts = {}
    for split, names in split_files.items():
        img_out = out / "images" / split
        lbl_out = out / "labels" / split
        img_out.mkdir(parents=True, exist_ok=True)
        lbl_out.mkdir(parents=True, exist_ok=True)
        kept = 0

        for name in tqdm(names, desc=f"  recortando {split}", ncols=78):
            stem = Path(name).stem
            label_file = label_dir / f"{stem}.txt"
            if not label_file.exists():
                continue
            rows = [r.split() for r in label_file.read_text().splitlines() if r.strip()]
            if not rows:
                continue
            w, h = image_wh
            boxes = np.array(
                [[(float(r[1]) - float(r[3]) / 2) * w, (float(r[2]) - float(r[4]) / 2) * h,
                  (float(r[1]) + float(r[3]) / 2) * w, (float(r[2]) + float(r[4]) / 2) * h]
                 for r in rows], dtype=np.float32,
            )
            image = np.asarray(Image.open(p.train_images / name).convert("RGB"))

            for t, (x0, y0, x1, y1) in enumerate(grid):
                local = to_local(boxes, np.array([x0, y0, x1, y1]))
                clipped = local.copy()
                clipped[:, [0, 2]] = np.clip(local[:, [0, 2]], 0, x1 - x0)
                clipped[:, [1, 3]] = np.clip(local[:, [1, 3]], 0, y1 - y0)

                area = np.maximum((local[:, 2] - local[:, 0]) * (local[:, 3] - local[:, 1]), 1e-6)
                visible = np.maximum(clipped[:, 2] - clipped[:, 0], 0) * np.maximum(
                    clipped[:, 3] - clipped[:, 1], 0)
                keep = (visible / area) >= MIN_VISIBLE
                if not keep.any():
                    continue

                tw, th = x1 - x0, y1 - y0
                kb = clipped[keep]
                lines = [
                    f"0 {(b[0]+b[2])/2/tw:.6f} {(b[1]+b[3])/2/th:.6f} "
                    f"{(b[2]-b[0])/tw:.6f} {(b[3]-b[1])/th:.6f}"
                    for b in kb
                ]
                crop = f"{stem}_t{t:02d}"
                Image.fromarray(image[y0:y1, x0:x1]).save(img_out / f"{crop}.png")
                (lbl_out / f"{crop}.txt").write_text("\n".join(lines), encoding="utf-8")
                kept += 1
        counts[split] = kept

    data = {"path": str(out), "train": "images/train", "val": "images/val",
            "names": {0: cfg["data"]["class_names"][0]}}
    (out / "data.yaml").write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    print(f"  tiles gerados: " + " | ".join(f"{k}={v}" for k, v in counts.items()))
    return out / "data.yaml"


# ---------------------------------------------------------------------------- treino

def run_experiment(key: str, spec: dict, p, cfg: dict, project: Path) -> dict:
    from ultralytics import YOLO

    fold_dir = p.runs_root / "folds" / f"grouped_fold{FOLD}"
    if spec["dataset"] == "tiles":
        out = p.runs_root / "tiled_fold0"
        data_yaml = out / "data.yaml"
        if not data_yaml.exists():
            splits = {
                s: [Path(l).name
                   for l in (fold_dir / f"{s}.txt").read_text().splitlines() if l.strip()]
                for s in ("train", "val")
            }
            data_yaml = build_tiled_dataset(p, cfg, splits, out)
    else:
        data_yaml = fold_dir / "data.yaml"

    last = project / key / "weights" / "last.pt"
    started = time.perf_counter()
    if last.exists():
        print(f"  retomando de {last}")
        model = YOLO(str(last))
        model.train(resume=True)
    else:
        model = YOLO(spec["model"])
        model.train(
            data=str(data_yaml), epochs=spec.get("epochs", cfg["train"]["epochs"]),
            imgsz=spec["imgsz"],
            batch=spec["batch"], patience=cfg["train"]["patience"], workers=0,
            cache="disk", deterministic=True, seed=cfg["seed"],
            project=str(project), name=key, exist_ok=True, val=True, plots=False, verbose=False,
            # Sem `aug` no spec, valem os defaults do Ultralytics — que é o que os quatro sem `aug` usaram.
            # O bloco existe para que uma comparação de CAPACIDADE possa igualar a augmentação
            # entre os braços: comparar um modelo com augmentação customizada contra outro sem
            # ela mede as duas coisas somadas, que foi exatamente o erro do yolo11s.
            **spec.get("aug", {}),
        )

    weights = project / key / "weights" / "best.pt"
    metrics = YOLO(str(weights)).val(
        data=str(data_yaml), split="val", project=str(project),
        name=f"{key}_val", exist_ok=True, verbose=False,
    )
    import pandas as pd

    results_csv = project / key / "results.csv"
    epochs = len(pd.read_csv(results_csv)) if results_csv.exists() else 0
    return {
        "experimento": key, "descricao": spec["descricao"], "modelo": spec["model"],
        "imgsz": spec["imgsz"], "dataset": spec["dataset"], "epocas": epochs,
        "epocas_pedidas": spec.get("epochs", cfg["train"]["epochs"]),
        "minutos": round((time.perf_counter() - started) / 60, 1),
        "val_mAP50_95": float(metrics.box.map), "val_mAP50": float(metrics.box.map50),
        "weights": str(weights),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", default=None)
    parser.add_argument("--list", action="store_true")
    args = parser.parse_args()

    if args.list:
        for k, v in EXPERIMENTS.items():
            print(f"  {k:<16} {v['descricao']}")
        return

    cfg, p = experiment(), paths()
    set_seed(cfg["seed"])
    project = p.runs_root / "experiments"
    out_json = p.resolve(p.results_root) / "experiments.json"
    done = json.loads(out_json.read_text(encoding="utf-8")) if out_json.exists() else []
    finished = {r["experimento"] for r in done}

    # Casa por SUBCADEIA, e nao por prefixo. Com nomes descritivos isso vira recurso em vez
    # de conveniencia: `--only yolo26s` pega os dois modelos s, `--only aug` pega os dois com
    # augmentacao anti-deriva, `--only 1280` pega todos os de alta resolucao.
    if args.only:
        want = [k.strip() for k in args.only.split(",") if k.strip()]
        queue = [k for k in EXPERIMENTS if any(w in k for w in want)]
        if not queue:
            raise SystemExit(f"--only {args.only} nao casa nenhum de {list(EXPERIMENTS)}")
    else:
        queue = list(EXPERIMENTS)
    print(f"fila: {', '.join(queue)}  |  fold {FOLD}  |  saida em {project}\n")

    for key in queue:
        if key in finished and not args.only:
            print(f"--- {key}: ja concluido, pulando ---")
            continue
        spec = EXPERIMENTS[key]
        print(f"--- {key}: {spec['descricao']} ---")
        try:
            record = run_experiment(key, spec, p, cfg, project)
        except Exception as exc:  # um experimento que falha nao derruba a fila
            print(f"  FALHOU: {type(exc).__name__}: {exc}\n")
            continue
        done = [r for r in done if r["experimento"] != key] + [record]
        out_json.write_text(json.dumps(done, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"  mAP50-95={record['val_mAP50_95']:.4f}  mAP50={record['val_mAP50']:.4f}  "
              f"({record['epocas']} ep, {record['minutos']} min)\n")

    print("=" * 70)
    baseline = p.runs_root / "train" / f"grouped_fold{FOLD}" / "results.csv"
    if baseline.exists():
        import pandas as pd

        d = pd.read_csv(baseline)
        m = [c for c in d.columns if "mAP50-95" in c][0]
        print(f"  {'baseline (YOLO26n @640)':<34} mAP50-95={d[m].max():.4f}")
    for r in sorted(done, key=lambda x: -x["val_mAP50_95"]):
        print(f"  {r['experimento']:<34} mAP50-95={r['val_mAP50_95']:.4f}  "
              f"({r['epocas']} ep, {r['minutos']} min)")
    print(f"\n  -> {out_json}")
    print("\nAVISO: estas metricas sao de VALIDACAO. A comparacao que vale sai de rodar")
    print("scripts/03_eval_arms.py com os pesos novos, no conjunto de teste.")


if __name__ == "__main__":
    main()
