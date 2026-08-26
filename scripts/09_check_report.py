"""Etapa 9: confere que os números do relatório existem nos artefatos que os produziram.

Um relatório escrito à mão sobre resultados que mudam é uma fonte silenciosa de erro: basta
reprocessar uma etapa e um número do texto passa a divergir do CSV. Este script lê
`results/.csv` e `results/.json` e verifica que cada valor-chave aparece no markdown, no
formato brasileiro (vírgula decimal).

Não é uma prova de correção do texto — ele não sabe se o número está no lugar certo, e não
verifica prosa. É uma trava contra a divergência mais provável: reprocessar e esquecer de
atualizar.

Uso:
    python scripts/09_check_report.py
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "results"
REPORT = REPO / "report" / "relatorio.md"


def br(value: float, decimals: int) -> str:
    """Formata no padrão do relatório: vírgula decimal, ponto de milhar."""
    return f"{value:,.{decimals}f}".replace(",", "\x00").replace(".", ",").replace("\x00", ".")


def collect() -> list[tuple[str, str]]:
    """(rótulo, texto esperado) para cada número que o relatório deveria citar."""
    wanted: list[tuple[str, str]] = []

    arms_csv = RESULTS / "arms.csv"
    if arms_csv.exists():
        for row in csv.DictReader(arms_csv.open(encoding="utf-8")):
            arm = row["arm"]
            for col, dec in (("AP", 3), ("AP50", 3), ("f1", 3), ("MAE", 1)):
                wanted.append((f"arms.{arm}.{col}", br(float(row[col]), dec)))

    # A tabela da §6 tem oito linhas e seis colunas de metrica cada — 48 numeros que sao
    # justamente os que mais mudam, porque cada treino novo reescreve uma linha. Conferir so
    # o `arms.csv` deixava essa tabela inteira sem rede.
    modelos = RESULTS / "model_comparison.csv"
    if modelos.exists():
        # Cada modelo entra pelo braco que a validacao DELE elegeu, que e o que a §6 reporta.
        por_modelo: dict[str, dict] = {}
        for row in csv.DictReader(modelos.open(encoding="utf-8")):
            atual = por_modelo.get(row["modelo"])
            if atual is None or float(row["val_MAE"]) < float(atual["val_MAE"]):
                por_modelo[row["modelo"]] = row
        for nome, row in sorted(por_modelo.items()):
            for col, dec, pct in (("MAE", 1, False), ("MAPE", 1, True), ("f1", 3, False),
                                  ("recall", 3, False), ("AP50", 3, False),
                                  ("AP_small", 3, False)):
                valor = float(row[col]) * (100 if pct else 1)
                wanted.append((f"modelos.{nome}.{col}", br(valor, dec) + ("%" if pct else "")))

    # A comparação pareada IoS vs IoU é o achado da Tarefa 2 e vale 25% da avaliação. Antes
    # ela vivia SÓ no texto — nenhum script a calculava, nenhum artefato a guardava — e quando
    # foi recalculada do CSV versionado deu outro número. Agora ela é conferida como as demais.
    pareado = RESULTS / "merge_paired.csv"
    if pareado.exists():
        for row in csv.DictReader(pareado.open(encoding="utf-8")):
            taxa = float(row["win_rate"]) * 100
            # 0,4% precisa de uma casa; 77% não pode virar "77,0%", que o texto não escreve.
            dec = 1 if taxa < 1 else 0
            wanted.append((f"fusao.{row['metrica']}.win_rate", br(taxa, dec) + "%"))
            wanted.append((f"fusao.{row['metrica']}.pairs", br(int(row["pairs"]), 0)))

    # Os otimos de calibracao: mesma historia do merge_paired. Estavam so no texto, e quando
    # foram recalculados dois dos tres nao batiam — um deles sustentava uma afirmacao ("tres
    # pontos distintos") que o dado nao suporta.
    otimos = RESULTS / "calibration_optima.json"
    if otimos.exists():
        dados = json.loads(otimos.read_text(encoding="utf-8"))
        for nome in ("melhor_f1", "menor_mae"):
            spec = dados.get(nome)
            if spec:
                wanted.append((f"calib.{nome}.conf", br(spec["conf"], 2)))
        vies = dados.get("melhor_f1", {}).get("bias_rel")
        if vies is not None:
            wanted.append(("calib.melhor_f1.bias", br(abs(vies) * 100, 1) + "%"))

    summary = RESULTS / "dataset_summary.json"
    if summary.exists():
        data = json.loads(summary.read_text(encoding="utf-8"))
        wanted.append(("dataset.n_instances", br(data["n_instances"], 0)))
        wanted.append(("dataset.n_images", br(data["n_images"], 0)))

    balance = RESULTS / "error_balance.json"
    if balance.exists():
        data = json.loads(balance.read_text(encoding="utf-8"))
        for key, dec in (("false_negatives", 0), ("false_positives", 0), ("gross_error", 0)):
            wanted.append((f"erro.{key}", br(data[key], dec)))

    cross = RESULTS / "crossframe_virtual.csv"
    if cross.exists():
        rows = list(csv.DictReader(cross.open(encoding="utf-8")))
        naive = sum(int(r["naive_sum"]) for r in rows)
        truth = sum(int(r["true_unique"]) for r in rows)
        wanted.append(("crossframe.naive_sum", br(naive, 0)))
        wanted.append(("crossframe.true_unique", br(truth, 0)))
        wanted.append(("crossframe.inflacao", br(naive / max(truth, 1), 2)))

    return wanted


def main() -> int:
    if not REPORT.exists():
        raise SystemExit(f"{REPORT} nao existe.")
    text = REPORT.read_text(encoding="utf-8")

    wanted = collect()
    if not wanted:
        raise SystemExit("Nenhum artefato em results/. Rode os scripts 01 a 06 antes.")

    missing = [(label, value) for label, value in wanted if value not in text]
    print(f"  {len(wanted) - len(missing)}/{len(wanted)} valores do results/ aparecem no relatório")
    for label, value in missing:
        print(f"    AUSENTE  {label:<28} = {value}")

    if "PENDENTE" in text:
        print(f"\n  {text.count('PENDENTE')} marcador(es) PENDENTE ainda no relatório")

    if missing:
        print("\n  Um valor ausente pode ser intencional (nem todo número entra no texto),")
        print("  mas confira se não é resultado reprocessado que o relatório não acompanhou.")
    return 1 if missing else 0


if __name__ == "__main__":
    raise SystemExit(main())
