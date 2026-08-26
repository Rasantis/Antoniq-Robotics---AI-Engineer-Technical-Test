"""Fusão de detecções vindas de tiles sobrepostos.

Este módulo é o núcleo da Tarefa 2. O ponto central é a escolha da métrica de casamento:

    Uma fruta cortada pela aresta de um tile produz uma caixa com aproximadamente metade da
    área da caixa que o tile vizinho gera para a mesma fruta. O IoU entre as duas é ~0,5 —
    exatamente em cima do limiar usual — então a duplicata escapa de forma imprevisível.
    O IoS (interseção dividida pela área da MENOR das duas caixas) dá ~1,0 no mesmo par, e
    remove a duplicata sem ambiguidade.

Isso importa porque o baseline "Tiled Faster R-CNN" do próprio artigo do MinneApple ficou
ABAIXO da inferência em imagem inteira (AP 0,341 vs 0,438), e os autores atribuem a perda ao
passo de supressão no final. Este módulo permite reproduzir essa falha (metric="iou") e
corrigi-la (metric="ios"), medindo os dois.

Contrapartida honesta, medida na ablação: o IoS também casa uma caixa pequena inteiramente
contida numa caixa maior. Em cacho de frutas isso pode suprimir uma fruta pequena legítima
que caia dentro da caixa de uma vizinha maior. O relatório reporta esse custo.

Ordem de grandeza medida no dataset, para calibrar expectativas: a caixa MÉDIA tem
27,8 x 28,5 px (a mediana, 27 x 28) e a maior tem 81 px de largura por 89 px de altura. Com a
faixa de
sobreposição de 64 px do tile 320, apenas 0,23% das frutas (64 de 28.182) são grandes
demais para caber inteiras em algum tile; com os 128 px do tile 640, nenhuma é.

Tudo em NumPy puro.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

_EPS = 1e-9


def box_areas(boxes: np.ndarray) -> np.ndarray:
    boxes = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
    wh = np.clip(boxes[:, 2:] - boxes[:, :2], 0.0, None)
    return wh[:, 0] * wh[:, 1]


def pairwise_intersection(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Área de interseção entre cada caixa de `a` e cada caixa de `b`. Saída (N, M)."""
    a = np.asarray(a, dtype=np.float64).reshape(-1, 4)
    b = np.asarray(b, dtype=np.float64).reshape(-1, 4)
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    wh = np.clip(rb - lt, 0.0, None)
    return wh[..., 0] * wh[..., 1]


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    inter = pairwise_intersection(a, b)
    union = box_areas(a)[:, None] + box_areas(b)[None, :] - inter
    return inter / np.maximum(union, _EPS)


def ios_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Interseção sobre a área da menor caixa ("intersection over smaller")."""
    inter = pairwise_intersection(a, b)
    smaller = np.minimum(box_areas(a)[:, None], box_areas(b)[None, :])
    return inter / np.maximum(smaller, _EPS)


_METRICS = {"iou": iou_matrix, "ios": ios_matrix}


@dataclass(frozen=True)
class MergeResult:
    boxes: np.ndarray        # (K, 4) xyxy
    scores: np.ndarray       # (K,)
    cluster_size: np.ndarray  # (K,) quantas detecções brutas formaram cada saída
    kept_index: np.ndarray   # (K,) índice, na entrada, da detecção de maior score do cluster


def merge_detections(
    boxes: np.ndarray,
    scores: np.ndarray,
    metric: str = "ios",
    policy: str = "nmm",
    threshold: float = 0.5,
) -> MergeResult:
    """Funde detecções sobrepostas com um passe guloso, do maior para o menor score.

    Args:
        boxes: (N, 4) em xyxy, coordenadas da imagem inteira.
        scores: (N,) confiança.
        metric: "iou" ou "ios" — como se decide que duas caixas são a mesma fruta.
        policy: "nms" mantém apenas a caixa de maior score do cluster;
                "nmm" devolve a união (envoltória) das caixas do cluster, o que recupera a
                extensão real de uma fruta que foi cortada por um tile.
        threshold: limiar de casamento na métrica escolhida.

    O algoritmo é "semeia e absorve até o ponto fixo": a detecção de maior score livre vira
    semente, absorve tudo que casa com ela, e — sob NMM — a caixa resultante é recomparada com
    o que sobrou, até não absorver mais nada.

    Isso difere do GREEDYNMM do SAHI, que não recompara a caixa fundida. A recomparação foi
    adicionada depois de medir que a versão sem ela deixava duplicatas na saída pelo próprio
    critério dela (1,0% das cenas), num padrão determinístico onde duas linhas de tiles se
    encostam. Efeito colateral bem-vindo: sem a recomparação, ``policy`` não alterava a
    contagem em nenhum caso — NMS e NMM produziam clusters idênticos e diferiam só na extensão
    da caixa. Agora NMM é genuinamente mais agressivo, e o eixo mede alguma coisa.

    Resíduo conhecido e medido: duas uniões formadas a partir de sementes diferentes podem
    acabar se sobrepondo entre si acima do limiar. Em cenas aleatórias isso aparece em ~0,05%
    dos casos, mas nos pontos de operação que de fato rodam o número é bem maior — medido
    sobre as 670 imagens de teste, atinge 4,3% das imagens no braço C e 8,2% no D (o limiar
    congelado é 0,3, não 0,5, e limiar mais baixo funde mais). Em caixas o efeito continua
    pequeno: 0,10% e 0,15% do total. Não é o mesmo defeito do laço sem recomparação — é a
    tendência do NMM de inchar a caixa, que é precisamente o custo que a ablação existe para
    medir. Fundir também as uniões entre si tornaria a política mais agressiva de um jeito que
    pode apagar fruta legítima, então isso fica declarado em vez de silenciosamente corrigido.
    """
    if metric not in _METRICS:
        raise ValueError(f"metric deve ser 'iou' ou 'ios', recebido {metric!r}")
    if policy not in ("nms", "nmm"):
        raise ValueError(f"policy deve ser 'nms' ou 'nmm', recebido {policy!r}")

    boxes = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    if len(boxes) != len(scores):
        raise ValueError(f"boxes ({len(boxes)}) e scores ({len(scores)}) têm tamanhos diferentes")

    # Caixas degeneradas (área zero ou invertidas) são descartadas antes de qualquer coisa.
    # Elas não são fruta: o recorte de coordenadas do detector colapsa para largura ou altura
    # zero quando uma predição cai fora do tile. E, se entrassem, seriam indestrutíveis — o
    # denominador do IoS é a área da menor caixa, que vale zero, então a similaridade contra
    # qualquer coisa (inclusive contra uma cópia idêntica) dá zero e a caixa nunca funde,
    # somando +1 permanente à contagem.
    wh = boxes[:, 2:] - boxes[:, :2]
    valid = np.flatnonzero((wh[:, 0] > 0) & (wh[:, 1] > 0))
    if valid.size == 0:
        return MergeResult(
            boxes=np.zeros((0, 4), np.float32),
            scores=np.zeros(0, np.float32),
            cluster_size=np.zeros(0, np.int32),
            kept_index=np.zeros(0, np.int64),
        )

    similarity = _METRICS[metric]
    order = valid[np.argsort(-scores[valid], kind="stable")]
    alive = np.zeros(len(boxes), dtype=bool)
    alive[valid] = True

    out_boxes, out_scores, out_sizes, out_index = [], [], [], []
    for i in order:
        if not alive[i]:
            continue
        alive[i] = False
        representative = boxes[i].copy()
        cluster = 1

        # Itera até o ponto fixo. Sob NMM a caixa representativa CRESCE a cada absorção, e
        # precisa ser recomparada: a união pode passar a conter integralmente uma caixa que
        # sobreviveu à rodada anterior. Sem isso a fusão não é idempotente — medido em 1,0%
        # das cenas aleatórias, e concentrado numa faixa determinística da imagem, onde duas
        # linhas de tiles se encostam. Ali uma única maçã produzia fragmento-topo, caixa
        # inteira e fragmento-base: o líder absorvia a inteira, a união resultante continha o
        # fragmento-base com IoS 1,0, e ele saía como uma segunda maçã.
        #
        # Sob NMS o representante nunca muda, então a segunda rodada não encontraria nada
        # novo e o laço termina de imediato. É por isso que, ANTES desta correção, `policy`
        # era literalmente um no-op para a contagem: as duas políticas produziam exatamente
        # os mesmos clusters (0 divergências em 18.000 comparações) e só diferiam na extensão
        # da caixa. Agora NMM é genuinamente mais agressivo que NMS, e o eixo da ablação
        # passa a significar alguma coisa.
        while True:
            candidates = np.flatnonzero(alive)
            if candidates.size == 0:
                break
            # Compara só contra o que ainda está vivo: evita materializar a matriz N x N, que
            # para 15 tiles x 300 detecções seria de centenas de MB.
            sim = similarity(representative.reshape(1, 4), boxes[candidates])[0]
            matched = candidates[sim >= threshold]
            if matched.size == 0:
                break

            alive[matched] = False
            cluster += int(matched.size)
            if policy == "nms":
                break
            representative = np.array([
                min(representative[0], boxes[matched, 0].min()),
                min(representative[1], boxes[matched, 1].min()),
                max(representative[2], boxes[matched, 2].max()),
                max(representative[3], boxes[matched, 3].max()),
            ])

        out_boxes.append(representative)
        out_scores.append(scores[i])
        out_sizes.append(cluster)
        out_index.append(i)

    return MergeResult(
        boxes=np.asarray(out_boxes, dtype=np.float32).reshape(-1, 4),
        scores=np.asarray(out_scores, dtype=np.float32),
        cluster_size=np.asarray(out_sizes, dtype=np.int32),
        kept_index=np.asarray(out_index, dtype=np.int64),
    )
