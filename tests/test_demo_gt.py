"""A verdade que o demo mostra na tela tem de ser a da imagem que ele recebeu.

Mostrar a contagem anotada ERRADA é pior que não mostrar nenhuma: a tela passaria a exibir um
erro de contagem inventado, e num assessment isso é o pior tipo de defeito — silencioso e
convincente. Estes testes cobrem as três saídas de `resolver_gt`, a integridade do índice
de caixas anotadas e a partição TP/FP/FN que a tela pinta.
"""
from __future__ import annotations

import csv
import hashlib
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
INDICE = REPO / "demo" / "gt_index.csv"


def _do_app(*nomes: str):
    """Extrai funções do app sem executá-lo (o import carregaria Flask, torch e os pesos)."""
    fonte = (REPO / "demo" / "app.py").read_text(encoding="utf-8")
    espaco: dict = {"Path": Path, "np": np}
    for nome in nomes:
        inicio = fonte.index(f"def {nome}(")
        fim = fonte.index("\n\n\n", inicio)
        exec(compile(fonte[inicio:fim], "app.py", "exec"), espaco)
    return [espaco[n] for n in nomes]


def _resolver_gt():
    return _do_app("resolver_gt")[0]


def _caixa(x: float, y: float, lado: float = 27.0) -> list[float]:
    """Uma maçã do tamanho mediano deste dataset."""
    return [x, y, x + lado, y + lado]


def _saida(caixas: list[list[float]]) -> SimpleNamespace:
    arr = np.asarray(caixas, dtype=np.float32).reshape(-1, 4)
    return SimpleNamespace(boxes=arr, scores=np.linspace(0.9, 0.5, len(arr), dtype=np.float32))


@pytest.fixture(scope="module")
def indice() -> list[dict]:
    if not INDICE.exists():
        pytest.skip("demo/gt_index.csv ausente; rode demo/indexar_gt.py")
    with INDICE.open(encoding="utf-8") as fh:
        return [dict(l, gt=int(l["gt"])) for l in csv.DictReader(fh)]


def test_indice_cobre_as_670_sem_hash_repetido(indice):
    assert len(indice) == 670
    assert len({l["sha1"] for l in indice}) == 670, "dois arquivos com o mesmo sha1"
    assert len({l["image"] for l in indice}) == 670
    assert all(l["gt"] > 0 for l in indice)


def test_caixas_do_indice_batem_com_a_contagem(indice):
    """Cada imagem leva exatamente `gt` caixas, de quatro inteiros cada.

    Se uma caixa se perdesse no empacotamento, a tela pintaria um falso negativo a mais e o
    apresentador defenderia um erro que não existe.
    """
    total = 0
    for l in indice:
        vals = l["boxes"].split()
        assert len(vals) == 4 * l["gt"], f"{l['image']}: {len(vals) / 4} caixas para gt {l['gt']}"
        assert all(v.lstrip("-").isdigit() for v in vals), f"{l['image']}: coordenada nao inteira"
        total += l["gt"]
    assert total == 28182, "o total tem de bater com dataset_summary.json"


def test_caixas_do_indice_sao_as_de_instances_csv(indice):
    """O índice é conveniência do demo; a fonte é a Etapa 1. As caixas não podem divergir."""
    fonte = REPO / "results" / "instances.csv"
    if not fonte.exists():
        pytest.skip("results/instances.csv ausente (fica fora do repositório por tamanho)")
    oficial: dict[str, list[tuple[int, ...]]] = {}
    with fonte.open(encoding="utf-8") as fh:
        for linha in csv.DictReader(fh):
            oficial.setdefault(linha["image"], []).append(
                tuple(round(float(linha[c])) for c in ("x0", "y0", "x1", "y1"))
            )
    for l in indice[:50]:                      # 50 imagens bastam para pegar deslocamento
        vals = [int(v) for v in l["boxes"].split()]
        empacotadas = [tuple(vals[i : i + 4]) for i in range(0, len(vals), 4)]
        assert empacotadas == oficial[l["image"]], f"{l['image']} divergiu"


def test_hash_reconhece_arquivo_renomeado(indice):
    resolver = _resolver_gt()
    por_nome = {l["image"]: l for l in indice}
    alvo = indice[0]
    dados = b"conteudo qualquer"
    por_sha = {hashlib.sha1(dados).hexdigest(): alvo}

    linha, origem = resolver(dados, "outro_nome_qualquer.png", por_sha, por_nome)
    assert linha["gt"] == alvo["gt"]
    assert "idêntico" in origem


def test_nome_reconhece_arquivo_recodificado(indice):
    """Bytes diferentes, nome preservado: cai na segunda via, e ela também tem de acertar."""
    resolver = _resolver_gt()
    alvo = indice[0]
    linha, origem = resolver(b"bytes que nao batem", alvo["image"], {},
                             {alvo["image"]: alvo})
    assert linha["gt"] == alvo["gt"]
    assert "nome" in origem


def test_imagem_de_fora_nao_inventa_verdade(indice):
    resolver = _resolver_gt()
    por_sha = {l["sha1"]: l for l in indice}
    por_nome = {l["image"]: l for l in indice}
    linha, origem = resolver(b"foto do celular", "IMG_4021.jpg", por_sha, por_nome)
    assert linha is None and origem is None


def test_caminho_completo_e_ignorado_so_o_nome_conta(indice):
    """O navegador manda `C:\\fakepath\\nome.png` em alguns casos; o resolvedor usa a folha."""
    resolver = _resolver_gt()
    alvo = indice[0]
    linha, _ = resolver(b"x", f"C:/fakepath/{alvo['image']}", {}, {alvo["image"]: alvo})
    assert linha["gt"] == alvo["gt"]


def test_exemplos_embutidos_estao_no_indice(indice):
    """Os dois exemplos da tela precisam ter verdade — é o que o demo mostra ao vivo."""
    por_sha = {l["sha1"]: l for l in indice}
    for nome in ("facil", "dificil"):
        caminho = REPO / "demo" / "exemplos" / f"{nome}.png"
        if not caminho.exists():
            pytest.skip(f"{caminho} ausente")
        linha = por_sha.get(hashlib.sha1(caminho.read_bytes()).hexdigest())
        assert linha is not None, f"{nome}.png não bate com nenhuma das 670"
        assert linha["gt"] > 0


def test_gt_do_indice_bate_com_o_artefato_do_pipeline(indice):
    """O índice é conveniência do demo; a fonte é results/images.csv. Não podem divergir."""
    fonte = REPO / "results" / "images.csv"
    if not fonte.exists():
        pytest.skip("results/images.csv ausente")
    with fonte.open(encoding="utf-8") as fh:
        oficial = {l["image"]: int(l["n_apples"]) for l in csv.DictReader(fh)}
    divergentes = [l["image"] for l in indice if oficial.get(l["image"]) != l["gt"]]
    assert not divergentes, f"{len(divergentes)} divergem, ex.: {divergentes[:3]}"


# --------------------------------------------------------------- TP / FP / FN da tela

def test_sem_anotacao_nao_ha_o_que_classificar():
    (classificar,) = _do_app("classificar")
    assert classificar(_saida([_caixa(10, 10)]), None) is None


def test_predicao_perfeita_e_toda_acerto():
    (classificar,) = _do_app("classificar")
    caixas = [_caixa(10, 10), _caixa(200, 300), _caixa(500, 900)]
    e = classificar(_saida(caixas), np.asarray(caixas, dtype=np.float32))
    assert len(e["TP"]) == 3 and len(e["FP"]) == 0 and len(e["FN"]) == 0
    assert len(e["GT"]) == 3


def test_sem_sobreposicao_tudo_vira_fp_e_fn():
    (classificar,) = _do_app("classificar")
    pred = [_caixa(10, 10), _caixa(60, 10)]
    gt = np.asarray([_caixa(400, 400)], dtype=np.float32)
    e = classificar(_saida(pred), gt)
    assert len(e["TP"]) == 0 and len(e["FP"]) == 2 and len(e["FN"]) == 1


def test_a_particao_e_exata_e_a_identidade_fecha():
    """TP+FP = o que o modelo apontou; TP+FN = o que estava anotado.

    Disso sai `contagem − verdade = FP − FN`, que é a frase que a tela sustenta. Se a partição
    vazasse, o console mostraria três números que não fecham — e é justamente essa conta que
    explica por que uma MAE baixa pode ser erro se cancelando.
    """
    (classificar,) = _do_app("classificar")
    rng = np.random.default_rng(7)
    for _ in range(40):
        gt = np.asarray([_caixa(*rng.uniform(0, 600, 2)) for _ in range(rng.integers(1, 25))],
                        dtype=np.float32)
        # metade das predicoes sao as anotadas deslocadas, metade sao inventadas
        desloc = gt[: len(gt) // 2] + rng.uniform(-14, 14, (len(gt) // 2, 1)).astype(np.float32)
        inventadas = np.asarray([_caixa(*rng.uniform(0, 600, 2)) for _ in range(rng.integers(0, 12))],
                                dtype=np.float32).reshape(-1, 4)
        pred = np.concatenate([desloc, inventadas]) if len(inventadas) else desloc
        e = classificar(_saida(pred.tolist()), gt)
        tp, fp, fn = (len(e[k]) for k in ("TP", "FP", "FN"))
        assert tp + fp == len(pred), "predicao que nao virou nem TP nem FP"
        assert tp + fn == len(gt), "anotacao que nao virou nem TP nem FN"
        assert len(pred) - len(gt) == fp - fn
        assert len(e["GT"]) == len(gt)


def test_uma_anotacao_nao_e_casada_duas_vezes():
    """Duas predições sobre a mesma maçã: uma é TP, a outra tem de ser FP."""
    (classificar,) = _do_app("classificar")
    gt = np.asarray([_caixa(100, 100)], dtype=np.float32)
    e = classificar(_saida([_caixa(100, 100), _caixa(102, 101)]), gt)
    assert len(e["TP"]) == 1 and len(e["FP"]) == 1 and len(e["FN"]) == 0


# ------------------------------------------------------- as cores tem de CHEGAR na tela

def _pixels_da_cor(uri: str, cor: tuple[int, int, int]) -> int:
    """Quantos pixels da imagem devolvida estão nessa cor, com folga para o JPEG."""
    import base64
    import io

    from PIL import Image

    bruto = base64.b64decode(uri.split(",", 1)[1])
    px = np.asarray(Image.open(io.BytesIO(bruto)).convert("RGB")).reshape(-1, 3).astype(np.int16)
    return int((np.abs(px - np.asarray(cor)).sum(1) < 60).sum())


def _desenhar():
    """`desenhar` depende de constantes de módulo, então elas entram no mesmo espaço."""
    import base64

    fonte = (REPO / "demo" / "app.py").read_text(encoding="utf-8")
    espaco: dict = {"np": np, "Path": Path, "base64": base64}
    for nome in ("ESTADOS", "ANOTACAO", "VERDE"):
        for linha in fonte.splitlines():
            if linha.startswith(nome + " "):
                exec(linha, espaco)
                break
    inicio = fonte.index("def desenhar(")
    exec(compile(fonte[inicio : fonte.index("\n\n\n", inicio)], "app.py", "exec"), espaco)
    return espaco["desenhar"], espaco


def test_as_quatro_cores_sobrevivem_ao_jpeg():
    """Regressão de um defeito medido, não hipotético.

    O JPEG subamostra croma em 4:2:0 por padrão: guarda a cor em METADE da resolução. Uma
    linha de 1 px colorida sobre fundo neutro simplesmente desaparecia — a tela mostrava só
    verde e vermelho, e o âmbar dos falsos negativos e o azul da anotação sumiam sem erro
    nenhum. O apresentador defenderia uma imagem que não mostra metade do diagnóstico.
    """
    desenhar, espaco = _desenhar()
    fundo = np.zeros((300, 300, 3), dtype=np.uint8) + 128
    c = lambda x, y: [x, y, x + 27, y + 27]  # noqa: E731 — maçã do tamanho mediano
    uri = desenhar(fundo, np.zeros((0, 4)), "teste", {
        "TP": np.array([c(20, 20)], dtype=float),
        "FP": np.array([c(80, 20)], dtype=float),
        "FN": np.array([c(140, 20)], dtype=float),
        "GT": np.array([c(20, 20), c(140, 20)], dtype=float),
    })
    esperadas = {**espaco["ESTADOS"], "anotação": espaco["ANOTACAO"]}
    ausentes = [nome for nome, cor in esperadas.items() if _pixels_da_cor(uri, cor) < 40]
    assert not ausentes, f"cores que nao chegaram na imagem: {ausentes}"


def test_sem_estados_desenha_so_as_deteccoes():
    desenhar, espaco = _desenhar()
    fundo = np.zeros((300, 300, 3), dtype=np.uint8) + 128
    uri = desenhar(fundo, np.array([[20, 20, 47, 47]], dtype=float), "teste")
    assert _pixels_da_cor(uri, espaco["VERDE"]) > 40
    assert _pixels_da_cor(uri, espaco["ANOTACAO"]) < 40, "sem verdade nao ha anotacao a pintar"
