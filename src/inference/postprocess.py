"""Pós-processamento das detecções brutas: limiar de confiança, borda e fusão.

A ordem importa e é deliberada:

    1. filtrar por confiança
    2. descartar caixas truncadas em aresta interna de tile
    3. fundir o que sobrou

Filtrar ANTES de fundir é o que um sistema em produção faz: o detector roda no seu ponto de
operação e a fusão recebe apenas o que passou. Fundir antes e filtrar depois pelo score do
líder daria números ligeiramente melhores e não corresponderia ao que roda no robô.

Como a fusão é barata (NumPy sobre algumas centenas de caixas) e a inferência é cara, todo o
varrimento de limiares e a ablação de fusão rodam aqui, sobre detecções brutas gravadas em
disco, sem tocar no detector.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import product

import numpy as np

from src.inference.engine import RawDetections
from src.tiling.merge import merge_detections

# Limiar da supressão de duplicata DENTRO de uma imagem, para os braços de um passe só. É o
# valor padrão de NMS na literatura de detecção, e não faz parte da ablação de fusão entre
# tiles — são dois problemas diferentes com dois limiares diferentes.
WITHIN_IMAGE_NMS_IOU = 0.7


@dataclass(frozen=True)
class MergePolicy:
    """Uma configuração de fusão. Hashable, serve de chave nas tabelas de ablação."""

    metric: str = "ios"
    policy: str = "nmm"
    threshold: float = 0.5
    drop_truncated: bool = True

    @property
    def label(self) -> str:
        border = "drop" if self.drop_truncated else "keep"
        return f"{self.metric}-{self.policy}@{self.threshold:g}-{border}"

    @classmethod
    def from_config(cls, cfg: dict) -> "MergePolicy":
        block = cfg["merge"]["default"]
        return cls(
            metric=block["metric"],
            policy=block["policy"],
            threshold=block["threshold"],
            drop_truncated=block["drop_truncated"],
        )


def ablation_grid(cfg: dict) -> list[MergePolicy]:
    """Produto cartesiano do bloco ``merge.ablation`` da configuração."""
    block = cfg["merge"]["ablation"]
    return [
        MergePolicy(metric=m, policy=p, threshold=t, drop_truncated=d)
        for m, p, t, d in product(
            block["metric"], block["policy"], block["threshold"], block["drop_truncated"]
        )
    ]


@dataclass
class Detections:
    """Detecções finais de uma imagem, prontas para contar e avaliar."""

    boxes: np.ndarray        # (K, 4) xyxy
    scores: np.ndarray       # (K,)
    cluster_size: np.ndarray  # (K,) quantas detecções brutas formaram cada caixa
    n_raw: int               # detecções acima do limiar, ANTES do controle de duplicata
    n_dropped_border: int    # removidas pela política de borda
    n_merged: int            # removidas pela fusão

    @property
    def count(self) -> int:
        """A contagem de frutas da imagem. É a saída que o produto entrega."""
        return len(self.boxes)

    @property
    def duplicates_removed(self) -> int:
        """Total removido pelo controle de duplicata: descarte de borda MAIS fusão.

        O descarte de borda é parte do controle de duplicata — ele elimina a vista truncada
        justamente porque outro tile vê a mesma fruta inteira. Contar só o que a fusão
        removeu invertia a leitura: a política que eliminava duas duplicatas por descarte
        reportava zero, e a que deixava uma passar reportava uma.
        """
        return self.n_raw - len(self.boxes)


def apply(raw: RawDetections, policy: MergePolicy, conf: float) -> Detections:
    """Aplica limiar, política de borda e fusão a um conjunto de detecções brutas.

    A fusão é pulada quando há um único tile. Ela existe para resolver duplicata ENTRE
    recortes sobrepostos; numa passada de imagem inteira não há nada disso, e o que sobra é
    uma segunda supressão que o sistema implantado não teria. Pior neste projeto: a cabeça
    um-para-um do YOLO26 é NMS-free e já entrega saída sem duplicata, então rodar IoS por cima
    só consegue remover fruta legítima em cacho — uma maçã pequena contida na caixa de uma
    vizinha maior casa com IoS alto e some.

    O custo medido, antes desta guarda, numa imagem real com 72 maçãs anotadas: 61 detecções
    caíam para 50, uma perda de 18% aplicada só aos braços de imagem inteira. A comparação
    entre estratégias ficava enviesada a favor do tiling exatamente pelo passo que deveria ser
    neutro.
    """
    # float() de propósito: sob as regras de promoção do NumPy 2, comparar um array float32
    # com um np.float64 promove de forma forte e float32(0,95) = 0,949999988 cai abaixo de
    # um limiar de 0,95. Com float nativo do Python a promoção é fraca e a comparação faz o
    # que se espera. Diverge em 2 dos 13 valores da varredura de confiança.
    keep = raw.scores >= float(conf)
    n_above_conf = int(keep.sum())

    if policy.drop_truncated:
        keep &= ~raw.truncated
    n_dropped = n_above_conf - int(keep.sum())

    if raw.n_tiles <= 1:
        # Supressão DENTRO da imagem, com IoU e limiar padrão — não a política cross-tile.
        #
        # A cabeça um-para-um do YOLO26 é NMS-free, mas a saída ainda traz duplicata:
        # 24,2% dos falsos positivos do braço de imagem inteira são uma segunda caixa numa
        # maçã que outra detecção já casou (IoU >= 0,5 com uma anotação já tomada).
        #
        # Aplicar NMS por IoU a 0,7 remove 2.931 dessas duplicatas custando 62 acertos, leva o
        # F1 de 0,607 para 0,637 e o viés de +7,8% para −2,8% — de fora para dentro da trava de
        # 3% do critério de aceitação.
        #
        # IoU e não IoS, de propósito: o IoS casa uma caixa pequena contida numa maior e
        # apagaria fruta legítima em cacho, que é exatamente o risco que a §4 do relatório
        # discute. Aqui não há aresta de tile cortando fruta, então o argumento a favor do IoS
        # não se aplica — o problema é a caixa repetida, e o IoU basta.
        result = merge_detections(
            raw.boxes[keep], raw.scores[keep],
            metric="iou", policy="nms", threshold=WITHIN_IMAGE_NMS_IOU,
        )
    else:
        result = merge_detections(
            raw.boxes[keep],
            raw.scores[keep],
            metric=policy.metric,
            policy=policy.policy,
            threshold=policy.threshold,
        )
    return Detections(
        boxes=result.boxes,
        scores=result.scores,
        cluster_size=result.cluster_size,
        n_raw=n_above_conf,
        n_dropped_border=n_dropped,
        n_merged=int(keep.sum()) - len(result.boxes),
    )

