"""Console de apresentação — slides para navegar sob sabatina + demo de inferência ao vivo.

Ferramenta de apresentação, não entregável — o enunciado não pede isto. Ela importa o
pipeline de verdade de `src/`, então o que a tela mostra é o mesmo código que produziu o
relatório, e não uma reimplementação que poderia divergir dele.

    pip install -r demo/requirements.txt
    python demo/app.py       -> http://127.0.0.1:5000
    python demo/app.py --port 8080

Atalhos na tela: setas navegam, 1-9/0 pulam para a estação, D abre o demo, S o backup.
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent      # demo/ mora dentro do repositório
sys.path.insert(0, str(REPO))

# Mesma regra do repositório: torch antes de pandas, senão WinError 1114 neste Windows.
from src.utils import torch_first  # noqa: E402,F401  isort:skip

import numpy as np  # noqa: E402
from flask import Flask, jsonify, request, send_file, send_from_directory  # noqa: E402
from PIL import Image  # noqa: E402

from src.inference.engine import arms_from_config, run_arm  # noqa: E402
from src.inference.postprocess import MergePolicy, apply  # noqa: E402
from src.utils.config import experiment, paths  # noqa: E402

app = Flask(__name__, static_folder=None)
app.config["MAX_CONTENT_LENGTH"] = 24 * 1024 * 1024

CFG = experiment()
P = paths()
FIGURAS = REPO / "results" / "figures"
PESOS = {
    "yolo26s_imgsz1280_200ep_aug": REPO / "models" / "yolo26s_imgsz1280_200ep_aug.pt",
    "yolo26n_imgsz640_120ep": REPO / "models" / "yolo26n_imgsz640_120ep.pt",
}
# Os mesmos STATUS de src/viz/figures.py: a tela e a figura impressa do relatorio precisam
# significar a mesma coisa, senao a plateia tem de reaprender a legenda no meio da defesa.
ESTADOS = {"TP": (0, 131, 0), "FP": (227, 73, 72), "FN": (237, 161, 0)}
ANOTACAO = (42, 120, 214)   # --serie: a caixa ANOTADA do acerto, para ver a localizacao
VERDE = ESTADOS["TP"]

_detectores: dict[str, object] = {}


def lado_mediano_px() -> float:
    """Lado da maçã mediana no quadro original, medido nas 28.182 instâncias da Etapa 1."""
    dados = json.loads((REPO / "results" / "dataset_summary.json").read_text(encoding="utf-8"))
    return float(dados["median_box_area_px2"]) ** 0.5


def ponto_congelado() -> dict[str, tuple[MergePolicy, float]]:
    """Braço -> (política de fusão, limiar), lidos do ponto que a Etapa 3 congelou.

    O mesmo para os dois conjuntos de pesos, de propósito: o demo compara BRAÇOS, e trocar o
    limiar junto com o braço tornaria a comparação ilegível.
    """
    dados = json.loads((REPO / "results" / "operating_point.json").read_text(encoding="utf-8"))
    saida = {}
    for nome, spec in dados["per_arm"].items():
        saida[nome] = (
            MergePolicy(metric=spec["metric"], policy=spec["merge_policy"],
                        threshold=spec["threshold"], drop_truncated=spec["drop_truncated"]),
            float(spec["conf"]),
        )
    return saida


def conf_por_modelo() -> dict[tuple[str, str], float]:
    """(pesos, braço) -> limiar que a validação DAQUELES pesos escolheu, da Etapa 12.

    Sem isto o demo aplicaria a todos os modelos o limiar congelado com o baseline, e a tela
    contradiria a §6 do relatório: o braço que a tabela elege apareceria pior que o vizinho,
    só porque estaria rodando no limiar de outro treino.
    """
    import csv

    caminho = REPO / "results" / "model_comparison.csv"
    if not caminho.exists():
        return {}
    with caminho.open(encoding="utf-8") as fh:
        return {(l["modelo"], l["arm"]): float(l["conf"]) for l in csv.DictReader(fh)}


def _desempacotar(bruto: str) -> np.ndarray:
    """A coluna `boxes` do indice: inteiros separados por espaco, quatro por caixa."""
    if not bruto:
        return np.zeros((0, 4), dtype=np.float32)
    return np.fromstring(bruto, sep=" ", dtype=np.float32).reshape(-1, 4)


def indice_gt() -> tuple[dict[str, dict], dict[str, dict]]:
    """(sha1 -> linha, nome -> linha) das 670 imagens anotadas. Ver demo/indexar_gt.py."""
    import csv

    caminho = Path(__file__).parent / "gt_index.csv"
    if not caminho.exists():
        return {}, {}
    with caminho.open(encoding="utf-8") as fh:
        linhas = [dict(l, gt=int(l["gt"]), boxes=_desempacotar(l.get("boxes", "")))
                  for l in csv.DictReader(fh)]
    return {l["sha1"]: l for l in linhas}, {l["image"]: l for l in linhas}


def resolver_gt(
    dados: bytes, nome: str | None, por_sha: dict[str, dict], por_nome: dict[str, dict]
) -> tuple[dict | None, str | None]:
    """Linha do índice da imagem recebida, se ela for uma das 670.

    O hash vem primeiro porque sobrevive a renomear; o nome é a segunda via, para o arquivo
    que foi re-salvo e mudou de bytes. Quando nenhuma das duas casa, devolve None — e a tela
    diz que não há verdade para comparar, em vez de inventar uma. Mostrar a verdade ERRADA
    seria pior que não mostrar nenhuma, e por isso os índices entram como argumento: dá para
    testar as três saídas sem subir o app.
    """
    import hashlib

    linha = por_sha.get(hashlib.sha1(dados).hexdigest())
    if linha is not None:
        return linha, "arquivo idêntico ao do dataset"
    linha = por_nome.get(Path(nome or "").name)
    if linha is not None:
        return linha, "reconhecida pelo nome do arquivo"
    return None, None


POR_SHA, POR_NOME = indice_gt()
LADO_MEDIANO = lado_mediano_px()
PONTOS = ponto_congelado()
CONF_MODELO = conf_por_modelo()
BRACOS = list(arms_from_config(CFG))


def detector(nome: str):
    """Carrega uma vez e reusa. A primeira chamada paga o aquecimento; as seguintes não."""
    if nome not in _detectores:
        from src.inference.engine import UltralyticsDetector
        caminho = PESOS.get(nome)
        if caminho is None or not caminho.exists():
            raise FileNotFoundError(f"Pesos ausentes: {caminho}")
        _detectores[nome] = UltralyticsDetector(str(caminho), conf=0.01)
    return _detectores[nome]


def rotulo_do_braco(arm, n_tiles: int) -> str:
    """O que o braço FAZ, em português.

    `A_full640` é o código da §4 do relatório, e fica — é por ele que a tela se liga à tabela.
    Mas sozinho ele foi lido como nome do modelo, que é outra coisa: os quatro braços rodam os
    MESMOS pesos, e a variável controlada é a estratégia de inferência.
    """
    if arm.tile is None:
        return f"imagem inteira · entrada {arm.imgsz} px · 1 passe"
    return f"{n_tiles} recortes de {arm.tile} px · entrada {arm.imgsz} px · {n_tiles} passes"


def maca_na_rede(arm) -> str:
    """Com quantos pixels a maçã mediana chega à rede neste braço.

    Trocou o "0,5x" da versão anterior, que dizia "maior" para um braço que ENCOLHE o objeto.
    O número absoluto também é mais útil: é o que a §4 usa para explicar por que recortar em
    640 e rodar em 640 não magnifica nada.
    """
    px = f"{LADO_MEDIANO * arm.magnification:.1f}".replace(".", ",")   # vírgula decimal
    return f"a maçã mediana chega à rede com {px} px"


def rotulo_da_imagem(braco: str, saida, estados: dict[str, np.ndarray] | None) -> str:
    if estados is None:
        return f"{braco} - {saida.count} deteccoes (sem anotacao para comparar)"
    tp, fp, fn = (len(estados[k]) for k in ("TP", "FP", "FN"))
    return f"{braco} - {saida.count} = {tp} TP + {fp} FP  |  {fn} FN"


def classificar(saida, gt_boxes: np.ndarray | None) -> dict[str, np.ndarray] | None:
    """Parte as caixas em TP / FP / FN, no MESMO critério que produziu os números do relatório.

    `match_predictions` é a função de `src/eval/detection.py`: IoU 0,5, guloso da maior
    confiança para a menor, um-para-um. Reusá-la é o que garante que o que a tela pinta e o
    que o relatório reporta não podem divergir — se divergissem, um dos dois estaria errado e
    não haveria como saber qual.

    `GT` são TODAS as caixas anotadas — a verdade inteira, desenhada como camada própria. Não
    é um quarto estado: cada anotação já é ou um TP ou um FN. Ela existe para a verdade poder
    ser vista como uma coisa só, e para se julgar a *localização* do acerto, que a caixa verde
    sozinha não mostra.
    """
    if gt_boxes is None:
        return None
    from src.eval.detection import match_predictions

    m = match_predictions(saida.boxes, saida.scores, gt_boxes, iou_threshold=0.5)
    casadas = m.pred_to_gt >= 0
    return {
        "TP": saida.boxes[casadas],
        "FP": saida.boxes[~casadas],
        "FN": gt_boxes[m.gt_to_pred < 0],
        "GT": gt_boxes,
    }


def desenhar(imagem: np.ndarray, caixas: np.ndarray, rotulo: str,
             estados: dict[str, np.ndarray] | None = None) -> str:
    """Imagem anotada como data URI. Caixa fina, porque a maçã tem 27 px.

    Com `estados`, pinta o diagnóstico em vez de só as detecções.

    A anotação é desenhada RECUADA 2 px para dentro, e não sobre a linha do estado. Medido: na
    versão anterior ela caía exatamente em cima, e o JPEG misturava as duas linhas de 1 px até
    as duas sumirem — a contagem de pixels da cor dava ZERO para âmbar e azul, enquanto o
    vermelho, que não tinha azul por cima, sobrevivia. Recuar separa as duas linhas.
    """
    import cv2

    tela = cv2.cvtColor(imagem, cv2.COLOR_RGB2BGR).copy()
    if estados is None:
        for x0, y0, x1, y1 in caixas.astype(int):
            cv2.rectangle(tela, (x0, y0), (x1, y1), VERDE[::-1], 2)
    else:
        for nome in ("FN", "FP", "TP"):
            for x0, y0, x1, y1 in estados[nome].astype(int):
                cv2.rectangle(tela, (x0, y0), (x1, y1), ESTADOS[nome][::-1], 2)
        for x0, y0, x1, y1 in estados["GT"].astype(int):
            cv2.rectangle(tela, (x0 + 2, y0 + 2), (x1 - 2, y1 - 2), ANOTACAO[::-1], 1)
    alt = tela.shape[0]
    cv2.rectangle(tela, (0, alt - 46), (tela.shape[1], alt), (255, 255, 255), -1)
    cv2.putText(tela, rotulo, (14, alt - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (11, 11, 11), 2,
                cv2.LINE_AA)
    # 4:4:4, e nao o padrao 4:2:0. Medido: com subamostragem de croma o JPEG guarda a cor em
    # METADE da resolucao, e uma linha de 1 px colorida sobre fundo cinza some — a contagem de
    # pixels da cor azul dava ZERO. Sem subamostragem ela sobrevive. Custa ~15% de tamanho.
    ok, buf = cv2.imencode(".jpg", tela, [
        int(cv2.IMWRITE_JPEG_QUALITY), 94,
        int(cv2.IMWRITE_JPEG_SAMPLING_FACTOR), int(cv2.IMWRITE_JPEG_SAMPLING_FACTOR_444),
    ])
    if not ok:
        raise RuntimeError("falha ao codificar a imagem anotada")
    return "data:image/jpeg;base64," + base64.b64encode(buf).decode()


@app.get("/")
def raiz():
    return send_file(Path(__file__).parent / "console.html")


@app.get("/fig/<nome>")
def figura(nome: str):
    if not (FIGURAS / nome).exists():
        return ("figura desconhecida", 404)
    return send_from_directory(FIGURAS, nome, max_age=3600)


@app.get("/exemplo/<tipo>")
def exemplo(tipo: str):
    """Duas imagens embarcadas. Rede de segurança: se o arrastar-e-soltar falhar na hora, ou
    se a máquina da apresentação não tiver o dataset, o demo continua funcionando."""
    caminho = Path(__file__).parent / "exemplos" / f"{tipo}.png"
    if not caminho.exists():
        return ("exemplo desconhecido", 404)
    return send_file(caminho, max_age=3600)


@app.get("/api/saude")
def saude():
    """Diz o que está carregado. Serve para aquecer o modelo ANTES da apresentação."""
    return jsonify({
        "pesos_disponiveis": {n: c.exists() for n, c in PESOS.items()},
        "carregados": sorted(_detectores),
        "bracos": [a.name for a in BRACOS],
        "imagens_com_verdade": len(POR_SHA),
        "pontos": {n: {"conf": c, "fusao": p.label} for n, (p, c) in PONTOS.items()},
    })


@app.post("/api/aquecer")
def aquecer():
    t0 = time.perf_counter()
    nome = (request.json or {}).get("modelo", "yolo26s_imgsz1280_200ep_aug")
    detector(nome)
    return jsonify({"modelo": nome, "ms": round((time.perf_counter() - t0) * 1e3)})


@app.post("/api/inferir")
def inferir():
    """Uma imagem, os quatro braços, no ponto congelado. É a tabela da §4 ao vivo."""
    arquivo = request.files.get("imagem")
    if arquivo is None or not arquivo.filename:
        return jsonify({"erro": "Nenhuma imagem enviada."}), 400
    modelo = request.form.get("modelo", "yolo26s_imgsz1280_200ep_aug")

    dados = arquivo.read()
    try:
        imagem = np.asarray(Image.open(io.BytesIO(dados)).convert("RGB"))
    except Exception as exc:
        return jsonify({"erro": f"Não consegui abrir a imagem: {exc}"}), 400

    verdade, gt_origem = resolver_gt(dados, arquivo.filename, POR_SHA, POR_NOME)
    gt = None if verdade is None else verdade["gt"]
    gt_boxes = None if verdade is None else verdade["boxes"]

    try:
        det = detector(modelo)
    except FileNotFoundError as exc:
        return jsonify({"erro": str(exc)}), 500

    linhas = []
    for arm in BRACOS:
        politica, conf_congelado = PONTOS.get(arm.name, (MergePolicy(), 0.25))
        # A política de fusão é do pipeline; o limiar é do treino. Por isso um vem do ponto
        # congelado e o outro, quando existe, da calibração daqueles pesos.
        proprio = CONF_MODELO.get((modelo, arm.name))
        conf = proprio if proprio is not None else conf_congelado
        t0 = time.perf_counter()
        bruto = run_arm(det, imagem, arm)
        saida = apply(bruto, politica, conf)
        ms = (time.perf_counter() - t0) * 1e3
        # Fora do cronômetro: classificar não faz parte da inferência, e incluí-la
        # inflaria uma latência que a tela apresenta como custo do braço.
        estados = classificar(saida, gt_boxes)
        linhas.append({
            "braco": arm.name,
            "estrategia": rotulo_do_braco(arm, int(bruto.n_tiles)),
            "escala": maca_na_rede(arm),
            "contagem": int(saida.count),
            "brutas": int(saida.n_raw),
            "duplicatas": int(saida.duplicates_removed),
            "borda": int(saida.n_dropped_border),
            "tiles": int(bruto.n_tiles),
            "ms": round(ms, 1),
            "conf": conf,
            "conf_origem": "os próprios pesos (§6)" if proprio is not None else "ponto congelado (§4)",
            "fusao": politica.label,
            # Assinado de proposito: o SINAL e a informacao. Sobrecontagem e subcontagem tem
            # causas opostas, e o vies (que e a media disto) e uma das travas de aceitacao.
            "erro": None if gt is None else int(saida.count) - gt,
            # Rotulo gravado na imagem: ASCII puro, porque a fonte Hershey do OpenCV nao tem
            # acento. Leva o diagnostico junto para a imagem ampliada se explicar sozinha.
            "imagem": desenhar(imagem, saida.boxes, rotulo_da_imagem(arm.name, saida, estados),
                               estados),
            **({} if estados is None else {
                "tp": int(len(estados["TP"])),
                "fp": int(len(estados["FP"])),
                "fn": int(len(estados["FN"])),
            }),
        })
    return jsonify({"modelo": modelo, "altura": imagem.shape[0], "largura": imagem.shape[1],
                    "gt": gt, "gt_origem": gt_origem, "arms": linhas})


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--port", type=int, default=5000)
    ap.add_argument("--host", default="127.0.0.1")
    args = ap.parse_args()
    faltando = [n for n, c in PESOS.items() if not c.exists()]
    if faltando:
        print(f"  AVISO: pesos ausentes ({', '.join(faltando)}) — o demo vai falhar neles.")
    print(f"\n  Console em http://{args.host}:{args.port}\n"
          f"  Dica: abra e aperte 'D' para aquecer o modelo ANTES de comecar.\n")
    app.run(host=args.host, port=args.port, debug=False, threaded=False)
