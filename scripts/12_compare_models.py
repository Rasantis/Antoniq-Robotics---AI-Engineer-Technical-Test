"""Etapa 12: os pesos dos experimentos medidos na mesma régua, no conjunto de teste.

A Etapa 11 treinou três variantes e reportou o mAP de VALIDAÇÃO de cada uma. Aqueles números
não se comparam entre si: o modelo de recortes valida sobre tiles de 320 px, onde há menos objetos por
imagem e cada um ocupa mais pixels relativos — a tarefa é mais fácil por construção. Colocar
0,4730 ao lado de 0,4472 numa tabela seria exatamente o tipo de comparação enviesada que o
relatório critica na literatura de tiling.

Aqui cada conjunto de pesos passa pelo PIPELINE INTEIRO — recorte, detecção, fusão, contagem —
nos quatro braços de inferência, e é medido no mesmo conjunto de teste, com a métrica de
produto (MAE de contagem), no ponto de operação escolhido na VALIDAÇÃO.

Dois cuidados de que o resultado depende:

    1. A política de fusão não é revarrida. Ela foi decidida na Etapa 3 (IoS/NMM), é
       propriedade do pipeline e não dos pesos. Aqui varre-se só o limiar de confiança, porque
       a distribuição de scores muda de modelo para modelo e um limiar fixo mediria qual deles
       por acaso está calibrado naquele ponto.

    2. A seleção é na validação, a leitura é no teste. Nenhum número de teste entra na
       escolha do limiar. É a mesma disciplina da Etapa 3.

Só o fold 0 é usado: é o único em que os pesos dos experimentos foram treinados, e avaliá-los
nos folds 1 e 2 leria imagens que estiveram no treino deles.

Uso:
    python scripts/12_compare_models.py                 # os quatro candidatos
    python scripts/12_compare_models.py --only 1280     # so os de alta resolucao
    python scripts/12_compare_models.py --analyze       # só reanalisa o que ja esta em disco
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# O torch PRECISA ser importado antes do pandas neste ambiente Windows, senao falha com
# WinError 1114 na inicializacao da DLL. Determinístico, medido — ver src/utils/torch_first.py.
from src.utils import torch_first  # noqa: F401  isort:skip

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from PIL import Image  # noqa: E402
from tqdm import tqdm  # noqa: E402

from src.inference import store  # noqa: E402
from src.inference.engine import arms_from_config, run_arm  # noqa: E402
from src.inference.postprocess import MergePolicy  # noqa: E402
from src.utils.config import experiment, paths  # noqa: E402
from src.utils.seed import set_seed  # noqa: E402

FOLD = 0


def _load_eval_arms():
    """Importa scripts/03_eval_arms.py, cujo nome começa com dígito e não é importável direto.

    Reaproveitar `evaluate` e `load_ground_truth` de lá não é conveniência: é o que garante
    que este comparativo mede EXATAMENTE o que a Etapa 3 mede. Uma segunda implementação da
    mesma métrica é uma segunda chance de divergir dela sem ninguém notar.
    """
    path = ROOT / "scripts" / "03_eval_arms.py"
    spec = importlib.util.spec_from_file_location("eval_arms", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["eval_arms"] = module
    spec.loader.exec_module(module)
    return module


# ------------------------------------------------------------------------------- candidatos

def candidates(p) -> dict[str, Path]:
    """Nome -> caminho dos pesos, descobertos em disco.

    Qualquer pasta em ``runs/experiments/*/weights/best.pt`` entra automaticamente — inclusive
    pesos treinados noutra máquina e copiados para cá. Uma lista fixa aqui significava editar
    o script a cada experimento novo, e um esquecimento aparecia como "esse modelo não existe"
    em vez de erro.

    O baseline entra sempre, como referência.
    """
    # Primeiro os checkpoints versionados em `models/`, para que um clone limpo reproduza a
    # tabela sem treinar nada. Depois `runs/experiments/`, que tem precedencia: numa maquina
    # que treinou, o resultado local e o que vale.
    #
    # `parent.parent` e nao `parent`: o pai imediato de best.pt e a pasta `weights`, e usar
    # ela colapsava todos os experimentos numa unica chave chamada "weights".
    repo = {f.stem: f for f in sorted((ROOT / "models").glob("*.pt"))}
    treinados = {
        d.parent.parent.name: d
        for d in sorted((p.runs_root / "experiments").glob("*/weights/best.pt"))
    }
    base = p.runs_root / "train" / f"grouped_fold{FOLD}" / "weights" / "best.pt"
    if base.exists():
        treinados["yolo26n_imgsz640_120ep"] = base
    return {**repo, **treinados}


def frozen_policies(results: Path) -> dict[str, MergePolicy]:
    """Política de fusão já congelada por braço na Etapa 3."""
    data = json.loads((results / "operating_point.json").read_text(encoding="utf-8"))
    out = {}
    for arm, spec in data["per_arm"].items():
        out[arm] = MergePolicy(
            metric=spec["metric"], policy=spec["merge_policy"],
            threshold=spec["threshold"], drop_truncated=spec["drop_truncated"],
        )
    return out


# ------------------------------------------------------------------------------- inferência

def infer(p, results, name, weights, arms, device, ev, force=False) -> None:
    from src.inference.engine import UltralyticsDetector

    todo = [
        (arm, split) for split in ("val", "test") for arm in arms
        if force or not store.path_for(results, FOLD, arm.name, split, f"cmp-{name}").exists()
    ]
    if not todo:
        print(f"  {name}: detecções brutas já em disco, pulando")
        return
    if not weights.exists():
        print(f"  {name}: PESOS AUSENTES em {weights}, pulando")
        return

    detector = UltralyticsDetector(str(weights), device=device)
    for arm, split in todo:
        images = ev.fold_images(p.runs_root, FOLD, split)
        per_image = {}
        for img_name in tqdm(images, desc=f"  {name:<13}{split:<5}{arm.name}", ncols=78):
            img = np.asarray(Image.open(p.train_images / img_name).convert("RGB"))
            per_image[img_name] = run_arm(detector, img, arm)
        store.save(store.path_for(results, FOLD, arm.name, split, f"cmp-{name}"), per_image)


# --------------------------------------------------------------------------------- análise

def analyse(results, name, arms, policies, gt, confs, match_iou, ev) -> list[dict]:
    """Varre o limiar na validação, congela, e lê o teste UMA vez no ponto congelado."""
    rows = []
    for arm in arms:
        policy = policies.get(arm.name, MergePolicy())
        val = ev.load_split(results, [FOLD], arm.name, "val", f"cmp-{name}")
        test = ev.load_split(results, [FOLD], arm.name, "test", f"cmp-{name}")
        if not val or not test:
            continue

        # --- seleção, só na validação
        sweep = []
        for conf in confs:
            m, _ = ev.evaluate(val, gt, policy, conf, match_iou, with_coco=False)
            sweep.append({"conf": conf, **m})
        sw = pd.DataFrame(sweep)

        # Mesma regra da Etapa 3, e o MESMO limite: importado de lá, não recopiado. Uma
        # segunda constante com o mesmo nome é uma segunda chance de divergir em silêncio.
        ok = sw[sw["bias_rel"].abs() <= ev.MAX_ABS_BIAS]
        bias_ok = not ok.empty
        pool = ok if bias_ok else sw
        best = pool.loc[pool["MAE"].idxmin()]
        conf = float(best["conf"])

        # --- leitura, no teste, no ponto que a validação escolheu
        tm, table = ev.evaluate(test, gt, policy, conf, match_iou, with_coco=True)
        rows.append({
            # `val_bias_ok`, e não `bias_ok`: a flag é da VALIDAÇÃO e fica na mesma linha do
            # `bias_rel` de TESTE. Com o nome curto ela lia como se qualificasse o número ao
            # lado — e é anti-correlacionada com ele, então dizia o contrário do que quer dizer.
            "modelo": name, "arm": arm.name, "conf": conf, "val_bias_ok": bias_ok,
            "val_MAE": float(best["MAE"]), "val_f1": float(best["f1"]),
            "MAE": tm["MAE"], "MAPE": tm["MAPE"], "bias_rel": tm["bias_rel"],
            "f1": tm["f1"], "precision": tm["precision"], "recall": tm["recall"],
            "AP": tm.get("AP", float("nan")), "AP50": tm.get("AP50", float("nan")),
            "AP_small": tm.get("AP_small", float("nan")),
            "R2": tm.get("R2", float("nan")),
            "latency_ms": tm["latency_total_ms"], "n_tiles": tm["n_tiles"],
            "n_imgs": len(table),
        })
        # A marca fica colada ao val_MAE, que é o número que ela qualifica.
        mark = "" if bias_ok else "!"
        print(f"    {arm.name:<12} conf {conf:.2f}  val_MAE {best['MAE']:>5.2f}{mark:<1}  ->  "
              f"teste MAE {tm['MAE']:>5.2f}  viés {tm['bias_rel']:+.1%}  "
              f"F1 {tm['f1']:.3f}  AP50 {tm.get('AP50', float('nan')):.3f}")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", default=None, help="ex.: 'yolo26n_imgsz1280_90ep,yolo11s_imgsz640_120ep'")
    parser.add_argument("--analyze", action="store_true", help="pula a inferência")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--device", default=None, help="padrão: GPU se houver, senão CPU")
    parser.add_argument("--arms", default=None,
                        help="limita os bracos, ex.: 'A_full640,B_full1280'. Util quando a GPU "
                             "esta ocupada: os bracos com tile custam 6 e 15 passes por imagem")
    args = parser.parse_args()

    cfg, p = experiment(), paths()
    set_seed(cfg["seed"])
    ev = _load_eval_arms()
    results = p.resolve(p.results_root)
    arms = arms_from_config(cfg)
    if args.arms:
        querido = [a.strip() for a in args.arms.split(",") if a.strip()]
        arms = [a for a in arms if a.name in querido]
        if not arms:
            raise SystemExit(f"--arms {args.arms} nao casa nenhum braco conhecido")
    gt = ev.load_ground_truth(results)
    policies = frozen_policies(results)
    # A MESMA grade da Etapa 3, importada de lá. Usar o `conf_sweep` de passo 0,05 daqui
    # reintroduziria o defeito que a Etapa 3 corrigiu: a grade grossa não amostra o
    # cruzamento do viés e marca como inviáveis braços que não são. Medido: com a grade
    # grossa o baseline aparece à frente do yolo11s; com a fina, o yolo11s passa.
    confs = ev.conf_grid(cfg)
    match_iou = cfg["counting"]["match_iou"]

    todo = candidates(p)
    if args.only:
        # Casa por SUBCADEIA. Com os nomes descritivos, `--only yolo26s` pega os dois modelos
        # s e `--only aug` pega os dois com augmentação anti-deriva — o filtro passa a ser
        # legível em vez de exigir o nome inteiro.
        want = [k.strip() for k in args.only.split(",") if k.strip()]
        todo = {k: v for k, v in todo.items() if any(w in k for w in want)}
        if not todo:
            raise SystemExit(f"--only {args.only} não casa nenhum de {sorted(candidates(p))}")

    print(f"comparativo | fold {FOLD} | {len(arms)} braços | {len(confs)} limiares")
    print(f"  política de fusão por braço, congelada na Etapa 3:")
    for arm, pol in policies.items():
        print(f"    {arm:<12} {pol.label}")
    print()

    if not args.analyze:
        print("[1/2] inferência")
        for name, weights in todo.items():
            infer(p, results, name, weights, arms, args.device, ev, args.force)

    print("\n[2/2] seleção na validação, leitura no teste")
    rows = []
    for name in todo:
        print(f"  {name}")
        rows.extend(analyse(results, name, arms, policies, gt, confs, match_iou, ev))

    if not rows:
        raise SystemExit("Nada avaliado. Rode sem --analyze para gerar as detecções.")

    table = pd.DataFrame(rows)
    out = results / "model_comparison.csv"
    # Merge, não sobrescrita: com `--only` a gravação apagava as linhas dos outros modelos
    # sem aviso, e a auditoria visual passava a morrer com "Sem linha para (baseline, ...)".
    if out.exists():
        antigo = pd.read_csv(out)
        antigo = antigo[~antigo["modelo"].isin(table["modelo"].unique())]
        table = pd.concat([antigo, table], ignore_index=True)
    # Ordem fixa: sem isto a linha de cada modelo aparece na ordem em que `candidates()` o
    # encontrou, que muda conforme o modelo esteja em models/ ou em runs/experiments/. As
    # metricas sao identicas, mas o arquivo inteiro aparece reescrito no diff — e um diff
    # ruidoso e um diff que ninguem le.
    table = table.sort_values(["modelo", "arm"], ignore_index=True)
    table.to_csv(out, index=False)

    print("\n" + "=" * 78)
    print("MELHOR BRAÇO DE CADA MODELO (escolhido pela MAE de VALIDAÇÃO)")
    print("=" * 78)
    best = table.loc[table.groupby("modelo")["val_MAE"].idxmin()].sort_values("MAE")
    cols = ["modelo", "arm", "conf", "MAE", "MAPE", "bias_rel", "f1", "AP50", "latency_ms"]
    print(best[cols].to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(f"\n  -> {out}")


if __name__ == "__main__":
    main()
