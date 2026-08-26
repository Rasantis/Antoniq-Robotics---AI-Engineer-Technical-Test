"""Comparação PAREADA entre as duas métricas de casamento da fusão.

Existe porque a média marginal engana aqui. A ablação varre 90 limiares de confiança, e nos
limiares altos o detector quase não devolve caixa: a MAE tende à contagem real e a política de
fusão deixa de importar. Uma média sobre a grade inteira soma esse regime morto a todas as
células por igual e comprime a diferença que se quer medir.

O pareamento resolve: para cada configuração idêntica em tudo — política, limiar de casamento,
descarte de borda, limiar de confiança e fold — compara IoS contra IoU e conta quem venceu. A
unidade passa a ser "em que fração das condições o IoS é melhor", que não depende de escala e
não é dominada pelo regime morto.

O resultado é o achado central da Tarefa 2, e é contra-intuitivo: o IoS domina em precisão,
nunca vence em recall (ele sempre remove mais), e na MAE de contagem perde na maioria. Ou
seja, o ganho é de qualidade de detecção e não de contagem.
"""
from __future__ import annotations

import pandas as pd

# Tudo que precisa ser igual para dois registros serem "a mesma configuração, métrica trocada".
CHAVE = ["arm", "policy", "threshold", "drop_truncated", "conf", "fold"]

# Métrica -> se maior é melhor. A MAE é a única em que menor vence, e tratá-la junto com as
# outras foi a origem de mais de um número invertido neste projeto.
METRICAS = {"precision": True, "recall": True, "f1": True, "MAE": False}


def paired_wins(ablation: pd.DataFrame) -> pd.DataFrame:
    """Fração das configurações pareadas em que o IoS supera o IoU, por métrica.

    Devolve uma linha por métrica com ``wins``, ``pairs`` e ``win_rate``. Empates não contam
    como vitória — o que se pergunta é onde o IoS é *melhor*, não onde ele não é pior.
    """
    chave = [c for c in CHAVE if c in ablation.columns]
    ios = ablation[ablation["metric"] == "ios"].set_index(chave)
    iou = ablation[ablation["metric"] == "iou"].set_index(chave)

    comum = ios.index.intersection(iou.index)
    if comum.empty:
        return pd.DataFrame(columns=["metrica", "wins", "pairs", "win_rate"])
    a, b = ios.loc[comum], iou.loc[comum]

    linhas = []
    for nome, maior_melhor in METRICAS.items():
        if nome not in a.columns:
            continue
        wins = int((a[nome] > b[nome]).sum() if maior_melhor else (a[nome] < b[nome]).sum())
        linhas.append({"metrica": nome, "wins": wins, "pairs": len(comum),
                       "win_rate": wins / len(comum)})
    return pd.DataFrame(linhas)
