"""Figuras do relatório.

Regras de visualização seguidas aqui, e por quê:

* Nenhum gráfico de eixo duplo. A calibração do limiar compara F1, MAE de contagem e
  viés — três grandezas em unidades diferentes. Sobrepor as três num eixo só produziria uma
  comparação visual sem sentido. Viram painéis empilhados com o eixo x (confiança)
  compartilhado, que é a forma correta para medidas de escalas distintas.
* Cores categóricas em ordem fixa, nunca cicladas, e validadas para daltonismo
  (separação CVD ΔE >= 8 em OKLab, verificada com o validador da paleta).
* Rótulo direto em todo valor que importa. Três das cores ficam abaixo de 3:1 de
  contraste contra o fundo claro, o que obriga identificação que não dependa só de cor.
* Traço fino, grade discreta, texto em tinta — nunca na cor da série.

Saída em PNG a 200 dpi, para entrar no PDF do relatório.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch, Rectangle  # noqa: E402

# ----------------------------------------------------------------------- paleta e estilo

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SOFT = "#52514e"
GRID = "#e4e3df"

# Ordem categórica fixa. Não ciclar, não reordenar por valor.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]

# Polaridade (subcontagem vs sobrecontagem): par divergente frio/quente.
NEGATIVE, POSITIVE, NEUTRAL = "#2a78d6", "#e34948", "#f0efec"

# Estado de uma detecção. Não reutilizar para série categórica.
STATUS = {"TP": "#008300", "FP": "#e34948", "FN": "#eda100"}

ARM_LABELS = {
    "A_full640": "A · imagem 640",
    "B_full1280": "B · imagem 1280",
    "C_tile640": "C · tile 640",
    "D_tile320": "D · tile 320",
}


def _setup() -> None:
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
        "axes.labelcolor": INK_SOFT, "text.color": INK,
        "xtick.color": INK_SOFT, "ytick.color": INK_SOFT,
        "axes.edgecolor": GRID, "axes.linewidth": 0.8,
        "grid.color": GRID, "grid.linewidth": 0.6,
        "legend.frameon": False, "figure.dpi": 200, "savefig.bbox": "tight",
    })


def _clean(ax, grid_axis: str = "y") -> None:
    """Grade discreta, eixos recessivos: o dado é que tem de ficar visível."""
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(axis=grid_axis, alpha=0.7, zorder=0)
    ax.set_axisbelow(True)


def _save(fig, out_dir: Path, name: str) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def _label_bars(ax, bars, values, fmt="{:.0f}", dy=0.01) -> None:
    span = max(ax.get_ylim()[1] - ax.get_ylim()[0], 1e-9)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + span * dy,
                fmt.format(value), ha="center", va="bottom", fontsize=8, color=INK)


# ------------------------------------------------------------------ 1. visão do dataset

def dataset_overview(instances: pd.DataFrame, images: pd.DataFrame, out_dir: Path) -> Path:
    """Por que este problema é 'objeto pequeno': as duas distribuições que definem a tarefa."""
    _setup()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3.2))

    side = np.sqrt(instances["bbox_area"])
    ax1.hist(side, bins=60, color=SERIES[0], edgecolor=SURFACE, linewidth=0.3)
    ax1.axvline(32, color=INK_SOFT, linestyle="--", linewidth=1)
    small = (instances["bbox_area"] < 32**2).mean()
    ax1.text(33, ax1.get_ylim()[1] * 0.88,
             f"corte COCO 'small'\n{small:.0%} das frutas abaixo",
             fontsize=8, color=INK, va="top")
    ax1.set_xlabel("lado equivalente da caixa (px)")
    ax1.set_ylabel("frutas")
    ax1.set_title(f"Tamanho do alvo — mediana {np.median(side):.0f} px", color=INK, loc="left")
    _clean(ax1)

    ax2.hist(images["n_apples"], bins=40, color=SERIES[0], edgecolor=SURFACE, linewidth=0.3)
    ax2.axvline(images["n_apples"].mean(), color=INK_SOFT, linestyle="--", linewidth=1)
    ax2.text(images["n_apples"].mean() + 2, ax2.get_ylim()[1] * 0.9,
             f"média {images['n_apples'].mean():.0f}", fontsize=8, color=INK)
    ax2.set_xlabel("maçãs anotadas por imagem")
    ax2.set_ylabel("imagens")
    ax2.set_title(f"Densidade — até {images['n_apples'].max()} numa imagem",
                  color=INK, loc="left")
    _clean(ax2)

    fig.tight_layout()
    return _save(fig, out_dir, "01_dataset")


# ------------------------------------------------------------- 2. splits e vazamento

def split_composition(folds: pd.DataFrame, images: pd.DataFrame, out_dir: Path) -> Path:
    """Mapa sessão x fold: a evidência visual de que nenhuma sessão cruza os conjuntos."""
    _setup()
    grouped = folds[folds["scheme"] == "grouped"]
    sessions = sorted(images["session"].unique())
    role_colour = {"train": SERIES[0], "val": SERIES[3], "test": SERIES[1]}

    fig, ax = plt.subplots(figsize=(9, 2.6))
    for _, row in grouped.iterrows():
        for role in ("train", "val", "test"):
            for session in str(row[f"{role}_sessions"]).split(","):
                if session in sessions:
                    ax.add_patch(Rectangle(
                        (sessions.index(session) - 0.44, row["fold"] - 0.4), 0.88, 0.8,
                        facecolor=role_colour[role], edgecolor=SURFACE, linewidth=1.5))

    counts = images["session"].value_counts()
    ax.set_xticks(range(len(sessions)))
    ax.set_xticklabels([f"{s[-6:]}\n({counts[s]})" for s in sessions], fontsize=7.5)
    ax.set_yticks(range(len(grouped)))
    ax.set_yticklabels([f"fold {i}" for i in grouped["fold"]])
    ax.set_xlim(-0.6, len(sessions) - 0.4)
    ax.set_ylim(len(grouped) - 0.5, -0.5)
    ax.set_xlabel("sessão de captura (hora do vídeo) e nº de imagens")
    ax.set_title("Split agrupado — nenhuma sessão aparece em dois conjuntos",
                 color=INK, loc="left")
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(False)
    ax.tick_params(length=0)
    ax.legend(handles=[Patch(facecolor=c, label=r) for r, c in role_colour.items()],
              loc="upper center", bbox_to_anchor=(0.5, -0.35), ncol=3)
    fig.tight_layout()
    return _save(fig, out_dir, "02_splits")


# ------------------------------------------------------------------ 3. os quatro braços

def arms_comparison(arms: pd.DataFrame, out_dir: Path) -> Path:
    """Três painéis, três unidades. Nunca no mesmo eixo."""
    _setup()
    cols = {"AP50": "AP50", "AP_small": "AP_small", "MAE": "MAE",
            "latency": "latency_total_ms", "tiles": "n_tiles"}
    summary = arms.set_index("arm")[list(cols.values())].rename(
        columns={v: k for k, v in cols.items()}
    )
    for name, source in (("AP50_sd", "AP50_sd"), ("MAE_sd", "MAE_sd")):
        summary[name] = arms.set_index("arm")[source] if source in arms else float("nan")
    summary = summary.reindex([a for a in ARM_LABELS if a in arms["arm"].unique()])

    labels = [ARM_LABELS[a] for a in summary.index]
    x = np.arange(len(summary))
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.4))

    panels = [
        (axes[0], "AP50", "AP50_sd", "AP@0,50 (detecção)", "{:.3f}"),
        (axes[1], "MAE", "MAE_sd", "MAE de contagem (maçãs/imagem)", "{:.1f}"),
        (axes[2], "latency", None, "latência por imagem (ms)", "{:.0f}"),
    ]
    for ax, col, err_col, title, fmt in panels:
        bars = ax.bar(x, summary[col], width=0.66, color=SERIES[0],
                      edgecolor=SURFACE, linewidth=1.5, zorder=3)
        if err_col and summary[err_col].notna().any():
            ax.errorbar(x, summary[col], yerr=summary[err_col], fmt="none",
                        ecolor=INK_SOFT, elinewidth=1, capsize=3, zorder=4)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=8)
        ax.set_title(title, color=INK, loc="left")
        ax.margins(y=0.18)
        _label_bars(ax, bars, summary[col], fmt)
        _clean(ax)

    axes[2].set_xlabel("barra de erro = desvio entre os 3 folds", fontsize=8)
    # O custo em passes entra no PRÓPRIO rótulo do eixo do painel de latência. Como anotação
    # separada abaixo do eixo ele colidia com os rótulos rotacionados.
    axes[2].set_xticklabels(
        [f"{lab}\n{n} passe{'s' if n > 1 else ''}"
         for lab, n in zip(labels, summary["tiles"])],
        rotation=20, ha="right", fontsize=8,
    )

    fig.suptitle("Quatro estratégias, mesmos pesos — a variável é a inferência",
                 x=0.01, ha="left", color=INK, fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    return _save(fig, out_dir, "03_bracos")


# ---------------------------------------------------------------- 4. ablação da fusão

def merge_paired_wins(pareado: pd.DataFrame, out_dir: Path) -> Path:
    """Em que fração das configurações pareadas o IoS supera o IoU, por métrica.

    Esta figura substituiu uma grade de MAE ao longo do limiar de casamento, e o motivo é o
    achado em si: no ponto de operação congelado as quatro células empatam (MAE 18,2 a 18,5;
    F1 0,655 a 0,661). A versão anterior parecia mostrar diferença porque fazia média sobre a
    grade INTEIRA de confiança — e o que separava as curvas era a cauda de conf alto, onde o
    detector quase não devolve caixa e a fusão deixa de importar. Ou seja: a figura mostrava um
    artefato da média, não o efeito da política.

    A comparação pareada é a leitura honesta. Ela não depende de escala, não é dominada pelo
    regime morto, e expõe a dissociação que é o ponto da Tarefa 2: o IoS domina em precisão e
    perde em contagem.
    """
    _setup()
    rotulos = {"precision": "precisão", "recall": "recall", "f1": "F1",
               "MAE": "MAE de contagem"}
    ordem = ["precision", "f1", "MAE", "recall"]
    d = pareado.set_index("metrica").reindex([m for m in ordem if m in set(pareado["metrica"])])

    fig, ax = plt.subplots(figsize=(9, 2.5))
    y = np.arange(len(d))
    # Verde acima de 50%, vermelho abaixo: a linha dos 50% e' o "tanto faz", e de que lado a
    # barra cai e' a informacao inteira.
    cores = [STATUS["TP"] if v > 0.5 else STATUS["FP"] for v in d["win_rate"]]
    ax.barh(y, d["win_rate"], height=0.6, color=cores, edgecolor=SURFACE, linewidth=1.5,
            zorder=3)
    ax.axvline(0.5, color=INK_SOFT, linewidth=1.2, linestyle="--", zorder=4)
    # Abaixo da última barra e à direita da linha: em cima colidia com o título, e
    # colado na linha colidia com o rótulo de 50% do eixo.
    ax.text(0.515, len(d) - 0.55, "empate", fontsize=8, color=INK_SOFT, va="center")

    for i, (nome, r) in enumerate(d.iterrows()):
        ax.text(r["win_rate"] + 0.015, i, f"{r['win_rate']:.0%}".replace("%", "%  ")
                + f"({int(r['wins'])} de {int(r['pairs'])})",
                va="center", fontsize=8.5, color=INK)

    ax.set_yticks(y, [rotulos.get(n, n) for n in d.index])
    ax.set_xlim(0, 1.0)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0], ["0%", "25%", "50%", "75%", "100%"])
    ax.set_xlabel("configurações pareadas em que o IoS supera o IoU")
    ax.set_title("IoS contra IoU, pareado — ganho de detecção, não de contagem",
                 color=INK, loc="left")
    ax.invert_yaxis()
    _clean(ax, grid_axis="x")
    fig.tight_layout()
    return _save(fig, out_dir, "04_ablacao_fusao")

def conf_calibration(sweep: pd.DataFrame, out_dir: Path, arm: str | None = None) -> Path:
    """O achado da Tarefa 3: os três ótimos não coincidem.

    Painéis empilhados, eixo x compartilhado. Cada métrica tem a sua escala — juntá-las
    num eixo só seria um gráfico de eixo duplo, que é exatamente o que não se faz.
    """
    _setup()
    data = sweep if arm is None else sweep[sweep["arm"] == arm]
    pooled = data.groupby(["arm", "conf"])[["f1", "MAE", "bias"]].mean().reset_index()
    arms = [a for a in ARM_LABELS if a in pooled["arm"].unique()]

    fig, axes = plt.subplots(3, 1, figsize=(7.2, 6.4), sharex=True)
    panels = [("f1", "F1 no ponto de operação", "max"),
              ("MAE", "MAE de contagem", "min"),
              ("bias", "viés (previsto − real)", "zero")]

    for ax, (col, title, best) in zip(axes, panels):
        for i, arm_name in enumerate(arms):
            grp = pooled[pooled["arm"] == arm_name].sort_values("conf")
            ax.plot(grp["conf"], grp[col], color=SERIES[i], linewidth=2, marker="o",
                    markersize=4, markeredgecolor=SURFACE, markeredgewidth=1,
                    label=ARM_LABELS[arm_name], zorder=3)
            values = grp[col].to_numpy()
            idx = {"max": np.argmax(values), "min": np.argmin(values),
                   "zero": np.argmin(np.abs(values))}[best]
            ax.plot(grp["conf"].iloc[idx], values[idx], marker="o", markersize=10,
                    markerfacecolor="none", markeredgecolor=SERIES[i],
                    markeredgewidth=1.8, zorder=4)
        if best == "zero":
            ax.axhline(0, color=INK_SOFT, linewidth=1, linestyle="--", zorder=2)
        # O título já nomeia a grandeza; repetir no eixo y só rouba espaço do gráfico.
        ax.set_title(title, color=INK, loc="left")
        _clean(ax)

    axes[0].legend(loc="lower center", ncol=len(arms), fontsize=8,
                   bbox_to_anchor=(0.5, 1.18))
    axes[-1].set_xlabel("limiar de confiança")
    fig.suptitle("O círculo marca o ótimo de cada painel — eles não coincidem",
                 x=0.01, ha="left", color=INK, fontsize=11, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    return _save(fig, out_dir, "05_calibracao")


# --------------------------------------------------------------- 6. modos de erro

def error_mode_ranking(ranking: pd.DataFrame, out_dir: Path) -> Path:
    """Barras divergentes: cada modo pelo que custa em maçãs, com sinal."""
    _setup()
    data = ranking.sort_values("delta_count")
    colours = [NEGATIVE if v < 0 else POSITIVE for v in data["delta_count"]]

    # Deitada, pelo mesmo motivo da figura de fusao: no relatorio a altura e o limite, e uma
    # figura quase quadrada acaba escalada ate os rotulos sumirem. Barras horizontais toleram
    # bem o formato largo — o comprimento da barra, que e a informacao, so fica mais legivel.
    fig, ax = plt.subplots(figsize=(12, 0.34 * len(data) + 1.3))
    y = np.arange(len(data))
    ax.barh(y, data["delta_count"], height=0.66, color=colours,
            edgecolor=SURFACE, linewidth=1.5, zorder=3)
    ax.axvline(0, color=INK_SOFT, linewidth=1, zorder=4)

    span = float(np.abs(data["delta_count"]).max())
    for i, row in enumerate(data.itertuples()):
        offset = span * 0.02
        ax.text(row.delta_count + (offset if row.delta_count > 0 else -offset), i,
                f"{row.delta_count:+,d}  ({row.share:.0%})",
                va="center", ha="left" if row.delta_count > 0 else "right",
                fontsize=8, color=INK)

    ax.set_yticks(y)
    ax.set_yticklabels(data["modo"], fontsize=8)
    ax.set_xlabel("contribuição ao erro de contagem (maçãs)")
    ax.set_xlim(-span * 1.55, span * 1.55)
    ax.set_title("Modos de erro, ranqueados pelo que custam na contagem",
                 color=INK, loc="left")
    ax.legend(handles=[Patch(facecolor=NEGATIVE, label="subcontagem (falso negativo)"),
                       Patch(facecolor=POSITIVE, label="sobrecontagem (falso positivo)")],
              loc="lower right", fontsize=8)
    _clean(ax, grid_axis="x")
    fig.tight_layout()
    return _save(fig, out_dir, "06_modos_de_erro")


# ------------------------------------------------------------------- 7. qualitativa

def _busiest_crop(case: dict, size: int, image_shape: tuple[int, int]) -> tuple[int, int]:
    """Canto do recorte quadrado que contém mais caixas — é onde a diferença aparece.

    Centrado nos ERROS (falsos negativos e positivos das duas estratégias), não em todas as
    detecções: a comparação existe para mostrar onde os braços divergem.
    """
    errors = [
        box
        for strategy in ("full", "tiled")
        for state in ("FP", "FN")
        for box in case[strategy].get(state, [])
    ] or [b for s in ("full", "tiled") for b in case[s].get("TP", [])]
    h, w = image_shape[:2]
    if not errors:
        return (w - size) // 2, (h - size) // 2
    centres = np.array([[(b[0] + b[2]) / 2, (b[1] + b[3]) / 2] for b in errors])
    cx, cy = np.median(centres, axis=0)
    return (
        int(np.clip(cx - size / 2, 0, max(w - size, 0))),
        int(np.clip(cy - size / 2, 0, max(h - size, 0))),
    )


def qualitative_panel(
    cases: list[dict], out_dir: Path, name: str = "07_qualitativa", crop: int = 460
) -> Path:
    """Comparação visual imagem inteira vs tiling, com TP/FP/FN em cores de estado.

    Duas decisões de forma, ambas por legibilidade no papel:

    * Recorte, não quadro cheio. Uma maçã de 27 px num quadro de 1280 px impresso a 178 mm
      de largura sai com menos de 3 mm — a caixa some. O recorte de 460 px mostra a fruta num
      tamanho em que dá para julgar o que aconteceu.
    * Casos nas colunas, estratégias nas linhas. O arranjo transposto, com as imagens
      retrato empilhadas, produzia uma figura de 375 mm de altura: mais que uma página A4
      inteira, para uma figura só.

    ``cases`` é uma lista de dicionários com ``image`` (array), ``title`` e, por estratégia, a
    lista de caixas por estado.
    """
    _setup()
    n = len(cases)
    fig, axes = plt.subplots(2, n, figsize=(3.2 * n, 7.0), squeeze=False)

    for col, case in enumerate(cases):
        ox, oy = _busiest_crop(case, crop, case["image"].shape)
        window = case["image"][oy : oy + crop, ox : ox + crop]
        for row, strategy in enumerate(["full", "tiled"]):
            ax = axes[row, col]
            ax.imshow(window)
            for state, colour in STATUS.items():
                for x0, y0, x1, y1 in case[strategy].get(state, []):
                    if ox <= (x0 + x1) / 2 <= ox + crop and oy <= (y0 + y1) / 2 <= oy + crop:
                        ax.add_patch(Rectangle((x0 - ox, y0 - oy), x1 - x0, y1 - y0,
                                               fill=False, edgecolor=colour, linewidth=1.2))
            counts = {s: len(case[strategy].get(s, [])) for s in STATUS}
            ax.set_title(
                f"{'imagem inteira' if row == 0 else 'tiles + fusão'}\n"
                f"TP {counts['TP']} · FP {counts['FP']} · FN {counts['FN']}  (quadro inteiro)",
                color=INK, loc="left", fontsize=8)
            ax.set_xlim(0, crop), ax.set_ylim(crop, 0)
            ax.set_xticks([]), ax.set_yticks([])
            for side in ax.spines.values():
                side.set_visible(False)
        axes[0, col].set_xlabel(case["title"], fontsize=8.5, color=INK, labelpad=4)
        axes[0, col].xaxis.set_label_position("top")

    fig.legend(handles=[Patch(facecolor=c, label=s) for s, c in STATUS.items()],
               loc="lower center", ncol=3, fontsize=9)
    fig.tight_layout(rect=[0, 0.045, 1, 1])
    # Folga entre as linhas: o título de duas linhas da linha de baixo escrevia por cima da
    # imagem da linha de cima, porque o tight_layout não reserva espaço para títulos de eixo
    # que ficam entre painéis.
    fig.subplots_adjust(hspace=0.22)
    return _save(fig, out_dir, name)


# ------------------------------------------------------------------ 8. cross-frame

def crossframe_counts(
    summary: pd.DataFrame, out_dir: Path, note: str | None = None,
    name: str = "08_crossframe",
) -> Path:
    """Soma ingênua vs deduplicada vs verdade, na passada virtual.

    ``note`` grava na figura a origem das detecções. Sem isso o gráfico pode ser lido fora de
    contexto: com anotações no lugar de detecções o resíduo é zero por construção, e esse
    número não é o que um detector real entregaria.
    """
    _setup()
    labels = ["soma por quadro\n(sem deduplicar)", "após deduplicação", "contagem única real"]
    values = [summary["naive_sum"].sum(), summary["dedup_count"].sum(),
              summary["true_unique"].sum()]
    colours = [POSITIVE, SERIES[0], "#555452"]

    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    bars = ax.bar(np.arange(3), values, width=0.62, color=colours,
                  edgecolor=SURFACE, linewidth=1.5, zorder=3)
    ax.set_xticks(np.arange(3))
    ax.set_xticklabels(labels, fontsize=8.5)
    ax.set_ylabel("frutas contadas (total da passada)")
    ax.margins(y=0.18)
    _label_bars(ax, bars, values, "{:,.0f}")

    inflation = values[0] / max(values[2], 1)
    residual = (values[1] - values[2]) / max(values[2], 1)
    ax.set_title(
        f"Sem deduplicar, a contagem infla {inflation:.2f}x; "
        f"depois sobra {residual:+.1%} de resíduo",
        color=INK, loc="left")
    if note:
        ax.text(0.0, -0.30, note, transform=ax.transAxes, fontsize=7.5,
                color=INK_SOFT, va="top")
    _clean(ax)
    fig.tight_layout()
    # `name` parametrizado porque este gerador tem DOIS modos — anotações como detecções, e
    # detecções do modelo. Com o nome fixo, rodar os dois numa passada fazia o segundo
    # sobrescrever o primeiro sem aviso, e a legenda do relatório passava a descrever a figura
    # errada.
    return _save(fig, out_dir, name)
