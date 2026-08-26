"""Testes do tiling.

É o único código do projeto em que um bug é silencioso: coordenadas trocadas ou duplicatas
não removidas não geram exceção, só degradam a contagem de um jeito difícil de perceber.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.tiling.merge import ios_matrix, iou_matrix, merge_detections
from src.tiling.remap import is_truncated, to_global, to_local
from src.tiling.slicer import coverage_map, crop_tiles, stitch, tile_grid

IMAGE_WH = (720, 1280)  # resolução nativa do MinneApple: retrato, largura x altura


# --------------------------------------------------------------------------- grade e recorte

@pytest.mark.parametrize(("tile", "overlap", "expected"), [(640, 0.2, 6), (320, 0.2, 15)])
def test_grid_tem_o_numero_de_tiles_previsto(tile, overlap, expected):
    """Os números da tabela de custo do relatório.

    Em 720x1280: tile 640 dá 2 colunas x 3 linhas = 6; tile 320 dá 3 x 5 = 15.
    """
    assert len(tile_grid(IMAGE_WH, tile, overlap)) == expected


@pytest.mark.parametrize("tile", [320, 512, 640])
def test_reconstrucao_e_exata(tile):
    """Costurar os tiles de volta tem de devolver a imagem original pixel a pixel."""
    rng = np.random.default_rng(0)
    img = rng.integers(0, 256, (IMAGE_WH[1], IMAGE_WH[0], 3), dtype=np.uint8)
    grid = tile_grid(IMAGE_WH, tile, 0.2)
    assert np.array_equal(stitch(crop_tiles(img, grid), grid, IMAGE_WH), img)


@pytest.mark.parametrize(("tile", "overlap"), [(320, 0.0), (320, 0.2), (640, 0.25), (512, 0.5)])
def test_nenhum_pixel_fica_fora_de_todos_os_tiles(tile, overlap):
    assert coverage_map(tile_grid(IMAGE_WH, tile, overlap), IMAGE_WH).min() >= 1


def test_tile_maior_que_a_imagem_vira_um_unico_recorte():
    grid = tile_grid(IMAGE_WH, 2048, 0.2)
    assert len(grid) == 1
    assert tuple(grid[0]) == (0, 0, 720, 1280)


def test_overlap_invalido_e_rejeitado():
    with pytest.raises(ValueError):
        tile_grid(IMAGE_WH, 640, 1.0)


# ------------------------------------------------------------------- transformação de coordenadas

def test_ida_e_volta_de_coordenadas_preserva_a_caixa():
    boxes = np.array([[10.0, 20.0, 50.0, 60.0], [0.0, 0.0, 5.0, 5.0]])
    for tile in tile_grid(IMAGE_WH, 640, 0.2):
        assert np.allclose(to_local(to_global(boxes, tile), tile), boxes)


def test_to_global_desloca_pela_origem_do_tile():
    tile = np.array([512, 80, 1152, 720])
    assert np.allclose(to_global(np.array([[1.0, 2.0, 3.0, 4.0]]), tile), [[513, 82, 515, 84]])


def test_caixas_vazias_nao_quebram_o_remap():
    empty = np.zeros((0, 4))
    assert to_global(empty, np.array([0, 0, 1, 1])).shape == (0, 4)
    assert is_truncated(empty, np.array([0, 0, 1, 1]), IMAGE_WH).shape == (0,)


# -------------------------------------------------------------------------- borda truncada

def test_truncada_marca_aresta_interna_mas_nao_borda_da_imagem():
    # Tile do canto superior esquerdo: as arestas em x=0 e y=0 SÃO a borda da imagem;
    # as arestas em x=640 e y=640 são internas.
    tile = np.array([0, 0, 640, 640])
    boxes = np.array(
        [
            [0.0, 300.0, 30.0, 340.0],     # colada em x=0  -> borda real da imagem
            [300.0, 0.0, 340.0, 30.0],     # colada em y=0  -> borda real da imagem
            [610.0, 300.0, 640.0, 340.0],  # colada em x=640 -> aresta interna
            [300.0, 610.0, 340.0, 640.0],  # colada em y=640 -> aresta interna
            [300.0, 300.0, 340.0, 340.0],  # no meio
        ]
    )
    assert is_truncated(boxes, tile, IMAGE_WH).tolist() == [False, False, True, True, False]


# ------------------------------------------------------------------------------- merge

def test_caixas_identicas_colapsam_em_uma():
    box = np.array([[100.0, 100.0, 140.0, 140.0]])
    boxes = np.vstack([box, box])
    for metric in ("iou", "ios"):
        result = merge_detections(boxes, np.array([0.9, 0.8]), metric=metric, threshold=0.5)
        assert len(result.boxes) == 1
        assert result.cluster_size.tolist() == [2]


def test_ios_remove_a_duplicata_de_borda_que_o_iou_deixa_passar():
    """O teste que justifica a escolha de IoS. É o achado central da Tarefa 2.

    Uma maçã de 40x40 px cortada pela aresta de um tile a 45% da sua largura gera uma caixa de
    18x40. Contra a caixa inteira que o tile vizinho produz:
        IoU = 18*40 / (40*40) = 0,45  -> abaixo de 0,5, a duplicata SOBREVIVE e é contada duas vezes
        IoS = 18*40 / (18*40) = 1,00  -> acima de 0,5, a duplicata é removida
    """
    full = np.array([[100.0, 100.0, 140.0, 140.0]])
    truncated = np.array([[100.0, 100.0, 118.0, 140.0]])  # 45% da largura

    assert iou_matrix(full, truncated)[0, 0] == pytest.approx(0.45)
    assert ios_matrix(full, truncated)[0, 0] == pytest.approx(1.00)

    boxes, scores = np.vstack([full, truncated]), np.array([0.9, 0.85])
    assert len(merge_detections(boxes, scores, metric="iou", threshold=0.5).boxes) == 2  # falha
    assert len(merge_detections(boxes, scores, metric="ios", threshold=0.5).boxes) == 1  # corrige


def test_nmm_recupera_a_extensao_real_quando_a_caixa_cortada_tem_score_maior():
    """União vs supressão, no caso em que a diferença aparece.

    Detector confiante no pedaço cortado (0,90) e menos na fruta inteira (0,85) é comum: o
    recorte remove contexto ambíguo. A supressão então fica com a caixa errada; a união
    recupera a extensão real.
    """
    truncated = np.array([[100.0, 100.0, 118.0, 140.0]])
    full = np.array([[100.0, 100.0, 140.0, 140.0]])
    boxes, scores = np.vstack([truncated, full]), np.array([0.90, 0.85])

    nmm = merge_detections(boxes, scores, metric="ios", policy="nmm", threshold=0.5)
    nms = merge_detections(boxes, scores, metric="ios", policy="nms", threshold=0.5)

    assert nmm.boxes[0].tolist() == [100.0, 100.0, 140.0, 140.0]  # extensão real recuperada
    assert nms.boxes[0].tolist() == truncated[0].tolist()          # mantém o pedaço cortado


def test_fruta_maior_que_a_faixa_de_sobreposicao_vira_duas_deteccoes():
    """Modo de falha estrutural do tiling, fixado em teste porque vai para o relatório.

    Com tile de 320 px e sobreposição de 0,2, a faixa comum entre tiles vizinhos tem 64 px.
    Uma fruta mais larga que isso pode ser cortada em TODOS os tiles que a enxergam — nenhum
    vê a fruta inteira. As duas metades têm IoS baixo entre si, então nem NMS nem NMM as unem,
    e uma fruta é contada duas vezes.

    No MinneApple a largura de caixa chega a 188 px, então o regime existe de verdade.
    É o principal argumento contra baixar o tamanho do tile indefinidamente.
    """
    left = np.array([[100.0, 100.0, 122.0, 140.0]])
    right = np.array([[118.0, 100.0, 140.0, 140.0]])
    boxes, scores = np.vstack([left, right]), np.array([0.9, 0.85])

    assert ios_matrix(left, right)[0, 0] < 0.5
    for policy in ("nms", "nmm"):
        result = merge_detections(boxes, scores, metric="ios", policy=policy, threshold=0.5)
        assert len(result.boxes) == 2, f"{policy} deveria falhar aqui, e falha"


def test_frutas_vizinhas_distintas_nao_sao_fundidas():
    """Guarda contra o custo do IoS: duas maçãs encostadas continuam sendo duas."""
    boxes = np.array([[100.0, 100.0, 140.0, 140.0], [138.0, 100.0, 178.0, 140.0]])
    result = merge_detections(boxes, np.array([0.9, 0.9]), metric="ios", threshold=0.5)
    assert len(result.boxes) == 2


def test_ios_suprime_caixa_pequena_contida_numa_maior():
    """Contrapartida conhecida do IoS, fixada em teste para que apareça no relatório.

    Uma fruta pequena inteiramente dentro da caixa de uma vizinha maior é removida pelo IoS,
    mas sobrevive ao IoU. É o custo que a ablação mede.
    """
    big = np.array([[100.0, 100.0, 200.0, 200.0]])
    small_inside = np.array([[120.0, 120.0, 145.0, 145.0]])
    boxes, scores = np.vstack([big, small_inside]), np.array([0.9, 0.85])

    assert len(merge_detections(boxes, scores, metric="ios", threshold=0.5).boxes) == 1
    assert len(merge_detections(boxes, scores, metric="iou", threshold=0.5).boxes) == 2


def test_merge_preserva_a_ordem_por_score():
    boxes = np.array([[0.0, 0.0, 10.0, 10.0], [500.0, 500.0, 510.0, 510.0]])
    result = merge_detections(boxes, np.array([0.3, 0.95]), metric="ios", threshold=0.5)
    assert result.scores.tolist() == pytest.approx([0.95, 0.3], abs=1e-6)
    assert result.kept_index.tolist() == [1, 0]


def test_entrada_vazia_devolve_saida_vazia():
    result = merge_detections(np.zeros((0, 4)), np.zeros(0))
    assert result.boxes.shape == (0, 4) and len(result.scores) == 0


def test_tamanhos_incompativeis_sao_rejeitados():
    with pytest.raises(ValueError):
        merge_detections(np.zeros((3, 4)), np.zeros(2))


# ------------------------------------------------------------------- regressões conhecidas

def test_fusao_absorve_ate_o_ponto_fixo():
    """A união recomparada. Sem isto a saída continha duplicatas pelo próprio critério dela.

    Caso real, reproduzido de uma revisão: uma maçã de 28x28 px na linha em que duas fileiras
    de tiles se encostam (y=640 na grade de 640) produz seis detecções brutas — dois
    fragmentos de topo, duas caixas inteiras e dois fragmentos de base. A semente absorve as
    inteiras; a união resultante contém o fragmento de base com IoS 1,0, mas a versão sem
    recomparação nunca o testava contra a união e ele saía como uma segunda maçã.
    """
    boxes = np.array(
        [[300, 625, 328, 640], [300, 625, 328, 640],
         [300, 625, 328, 653], [300, 625, 328, 653],
         [300, 640, 328, 653], [300, 640, 328, 653]], dtype=float
    )
    result = merge_detections(boxes, np.full(6, 0.9), metric="ios", policy="nmm", threshold=0.5)
    assert len(result.boxes) == 1
    assert result.boxes[0].tolist() == [300.0, 625.0, 328.0, 653.0]


@pytest.mark.parametrize("scores", [[0.90, 0.85, 0.80], [0.85, 0.90, 0.80], [0.80, 0.85, 0.90]])
def test_resultado_nao_depende_da_ordem_dos_scores(scores):
    """Fragmento, caixa inteira e fragmento da mesma fruta dão 1, seja qual for o mais confiante.

    Antes da recomparação, o resultado alternava entre 1 e 2 conforme qual das três vistas
    tivesse o maior score — ou seja, a contagem dependia da ordem de enumeração dos tiles.
    """
    boxes = np.array([[279.0, 621.0, 313.0, 640.0],
                      [279.0, 621.0, 313.0, 653.0],
                      [279.0, 640.0, 313.0, 653.0]])
    assert len(merge_detections(boxes, np.array(scores), metric="ios", threshold=0.5).boxes) == 1


def test_nmm_pode_fundir_mais_que_nms():
    """``policy`` precisa ser um eixo real da ablação, e não decoração.

    Enquanto a união não era recomparada, NMS e NMM produziam exatamente os mesmos clusters em
    18.000 combinações — o parâmetro só mudava a extensão da caixa. Aqui a união do primeiro
    par cresce o bastante para alcançar a terceira caixa, coisa que o representante fixo do
    NMS não faz.
    """
    boxes = np.array([[100.0, 100.0, 140.0, 140.0],
                      [100.0, 100.0, 180.0, 140.0],
                      [150.0, 105.0, 175.0, 135.0]])
    scores = np.array([0.9, 0.85, 0.8])
    nms = merge_detections(boxes, scores, metric="ios", policy="nms", threshold=0.5)
    nmm = merge_detections(boxes, scores, metric="ios", policy="nmm", threshold=0.5)
    assert len(nmm.boxes) < len(nms.boxes)


def test_caixas_degeneradas_sao_descartadas():
    """Caixa de área zero é indestrutível se entrar: o IoS divide pela área da menor.

    O recorte de coordenadas do detector colapsa para largura zero quando a predição cai fora
    do tile. Uma dessas caixas nunca casa com nada — nem com uma cópia idêntica de si mesma —
    e soma +1 permanente à contagem.
    """
    ghost = np.array([[720.0, 1060.0, 720.0, 1100.0]])   # largura zero
    real = np.array([[695.0, 1060.0, 720.0, 1100.0]])
    inverted = np.array([[140.0, 140.0, 100.0, 100.0]])  # x1 < x0

    assert len(merge_detections(np.vstack([ghost, ghost]), np.array([0.9, 0.8])).boxes) == 0
    assert len(merge_detections(np.vstack([inverted, inverted]), np.array([0.9, 0.8])).boxes) == 0
    # A caixa real sobrevive; o fantasma ao lado dela some.
    kept = merge_detections(np.vstack([real, ghost]), np.array([0.9, 0.5]))
    assert len(kept.boxes) == 1
    assert kept.boxes[0].tolist() == real[0].tolist()


def test_o_dataset_nao_tem_fruta_maior_que_a_faixa_de_640():
    """Fixa os números reais do dataset, que uma revisão mostrou estarem errados no texto.

    A afirmação anterior — "a largura de caixa chega a 188 px" — vinha de uma fonte de
    terceiros e é falsa: o máximo medido nas 28.182 instâncias é 81 px de largura e 89 px de
    lado. O argumento contra tiles pequenos continua de pé, mas o número muda de escala: com
    a faixa de sobreposição de 64 px do tile 320, apenas 0,23% das frutas são grandes demais;
    com os 128 px do tile 640, nenhuma é.
    """
    csv = Path(__file__).resolve().parents[1] / "results" / "instances.csv"
    if not csv.exists():
        pytest.skip("results/instances.csv ausente — rode scripts/01_prepare_data.py")

    inst = pd.read_csv(csv)
    longest = np.maximum(inst["w"], inst["h"])
    assert longest.max() <= 100, "o máximo real é 89 px; revise o texto se isto mudar"
    assert (longest > 64).mean() < 0.01     # faixa do tile 320
    assert (longest > 128).sum() == 0       # faixa do tile 640
