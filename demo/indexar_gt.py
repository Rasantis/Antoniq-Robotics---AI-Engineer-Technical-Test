"""Indexa as 670 imagens rotuladas para o demo saber a verdade da imagem que recebeu.

O demo aceita qualquer imagem, mas anotação só existe para as 670 do MinneApple. Sem uma
forma de reconhecer a imagem, a tela só poderia mostrar a contagem predita — que sozinha não
diz nada, porque não há com o que comparar.

Duas vias de reconhecimento, nesta ordem:

    sha1 do arquivo   pega a imagem mesmo renomeada. É a via boa.
    nome do arquivo   pega a imagem re-salva ou reconvertida, desde que o nome tenha sobrado.

Além da contagem, o índice leva as CAIXAS anotadas — é o que permite ao console pintar
TP/FP/FN em vez de só mostrar um número. Empacotadas como inteiros separados por espaço numa
única coluna: repetir o nome do arquivo em 28.182 linhas custaria 4x o tamanho do arquivo.

O artefato é versionado justamente porque o dataset NÃO é: a máquina da apresentação não vai
ter `ml/minneapple`, e o demo precisa funcionar assim mesmo.

Uso:
    python demo/indexar_gt.py          # regrava demo/gt_index.csv
"""
from __future__ import annotations

import csv
import hashlib
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from src.utils.config import paths  # noqa: E402

SAIDA = Path(__file__).parent / "gt_index.csv"


def sha1(caminho: Path) -> str:
    h = hashlib.sha1()
    with caminho.open("rb") as fh:
        for pedaco in iter(lambda: fh.read(1 << 20), b""):
            h.update(pedaco)
    return h.hexdigest()


def caixas_por_imagem() -> dict[str, list[tuple[int, int, int, int]]]:
    """Caixas anotadas por imagem, de results/instances.csv (a saida da Etapa 1).

    Arredondadas para inteiro: sao coordenadas de pixel, e a fracao vinha do centroide da
    mascara. Meio pixel nao muda IoU de uma caixa de 27 px e corta o arquivo pela metade.
    """
    fonte = REPO / "results" / "instances.csv"
    if not fonte.exists():
        raise SystemExit(f"{fonte} nao existe. Rode scripts/01_prepare_data.py.")
    saida: dict[str, list[tuple[int, int, int, int]]] = {}
    with fonte.open(encoding="utf-8") as fh:
        for l in csv.DictReader(fh):
            saida.setdefault(l["image"], []).append(
                tuple(round(float(l[c])) for c in ("x0", "y0", "x1", "y1"))
            )
    return saida


def main() -> int:
    imagens = paths().dataset_root / "detection" / "train" / "images"
    if not imagens.exists():
        raise SystemExit(f"{imagens} nao existe. Veja scripts/00_download.md.")

    fonte = REPO / "results" / "images.csv"
    if not fonte.exists():
        raise SystemExit(f"{fonte} nao existe. Rode scripts/01_prepare_data.py.")
    # `n_apples` de images.csv e a contagem ANOTADA — uma instancia por maca na mascara.
    gt = {l["image"]: int(l["n_apples"]) for l in csv.DictReader(fonte.open(encoding="utf-8"))}

    caixas = caixas_por_imagem()

    linhas = []
    for p in sorted(imagens.glob("*.png")):
        if p.name not in gt:
            continue
        b = caixas.get(p.name, [])
        if len(b) != gt[p.name]:
            raise SystemExit(
                f"{p.name}: {len(b)} caixas em instances.csv contra {gt[p.name]} em "
                "images.csv. Os dois saem da Etapa 1 — reprocesse antes de indexar."
            )
        linhas.append({
            "image": p.name, "sha1": sha1(p), "gt": gt[p.name],
            "boxes": " ".join(str(v) for caixa in b for v in caixa),
        })

    faltando = sorted(set(gt) - {l["image"] for l in linhas})
    with SAIDA.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["image", "sha1", "gt", "boxes"])
        w.writeheader()
        w.writerows(linhas)

    total = sum(l["gt"] for l in linhas)
    print(f"  {len(linhas)} imagens, {total} caixas anotadas -> "
          f"{SAIDA.relative_to(REPO)} ({SAIDA.stat().st_size // 1024} KB)")
    if faltando:
        print(f"  AVISO: {len(faltando)} em images.csv sem arquivo: {faltando[:3]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
