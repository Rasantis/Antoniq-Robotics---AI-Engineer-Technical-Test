"""Execução dos quatro braços de inferência comparados na Tarefa 2.

    A  full-image @ 640    1 passe,  objeto a 0,5x (o lado longo, 1280, cai para 640)
    B  full-image @ 1280   1 passe,  objeto a 1,0x
    C  tile 640 ovl 0,2    6 passes (2 x 3), objeto a 1,0x
    D  tile 320 ovl 0,2   15 passes (3 x 5), objeto a 2,0x

O MinneApple é retrato, 720 de largura por 1280 de altura, e a grade de tiles reflete isso.

Os quatro usam os MESMOS pesos. A variável controlada é a estratégia de inferência, não o
modelo. Todo recorte é alimentado à rede em ``imgsz``, então o custo é proporcional ao número
de tiles e a magnificação real do objeto é ``imgsz / tile``.

Vale explicitar porque quase nunca é dito: recortar em 640 e rodar em 640 não aumenta o
objeto. O braço C só reduz a densidade de objetos por passe. A magnificação de verdade
aparece no braço D, onde um recorte de 320 px é levado a 640.

Arquitetura: este módulo produz detecções BRUTAS, por tile, num limiar de confiança baixo, e
não funde nada. A fusão e a filtragem por confiança ficam em ``postprocess.py``. A separação
existe porque a ablação de merge tem 24 combinações e a calibração de contagem varre 13
limiares: fundir e filtrar fora da inferência transforma 24 x 13 execuções do detector em uma.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np

from src.tiling.remap import is_truncated, to_global
from src.tiling.slicer import crop_tiles, tile_grid

# Piso de confiança da inferência bruta. Tudo acima disto é guardado para que a varredura de
# limiares aconteça depois, sem chamar o detector de novo.
RAW_CONF = 0.01
# Teto de detecções por PASSE. Não é limite arquitetural do YOLO26: é este valor que o
# Ultralytics repassa à cabeça (`k = min(max_det, ancoras)`). Importa nos braços de imagem
# inteira — numa imagem de 123 maçãs em conf 0,01, o braço B chega perto de saturar. Os
# braços com tile têm orçamento de 6 x 300 e 15 x 300, e não chegam perto.
MAX_DET = 300


class Detector(Protocol):
    """Interface mínima que o motor precisa. Permite testar sem GPU nem pesos."""

    def predict(self, images: list[np.ndarray], imgsz: int) -> list[tuple[np.ndarray, np.ndarray]]:
        """Recebe imagens RGB e devolve (boxes xyxy, scores) por imagem.

        RGB é a convenção de todo o pipeline, porque é o que o PIL entrega. Adaptadores para
        bibliotecas que esperam BGR convertem internamente.
        """
        ...


@dataclass
class RawDetections:
    """Detecções antes de qualquer fusão, já em coordenadas da imagem inteira."""

    boxes: np.ndarray       # (N, 4) xyxy na imagem original
    scores: np.ndarray      # (N,)
    tile_index: np.ndarray  # (N,) qual tile produziu a detecção (0 para imagem inteira)
    truncated: np.ndarray   # (N,) bool: encostada numa aresta interna do tile
    n_tiles: int
    latency_ms: dict[str, float] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.boxes)


# Lado mais longo da imagem nativa. O redimensionamento do Ultralytics leva o lado MAIOR
# a `imgsz`, entao e ele que determina a escala do objeto na inferencia de imagem inteira.
NATIVE_LONG_SIDE = 1280


@dataclass(frozen=True)
class Arm:
    """Configuração de um braço. Espelha um bloco de ``configs/experiment.yaml``."""

    name: str
    tile: int | None
    imgsz: int
    overlap: float
    long_side: int = NATIVE_LONG_SIDE

    @property
    def magnification(self) -> float:
        """Fator de escala aplicado ao objeto antes de chegar à rede.

        Recortar em 640 e rodar em 640 dá 1,0 — ou seja, nenhuma magnificação. Só um tile
        menor que ``imgsz`` aumenta o objeto de fato.
        """
        return self.imgsz / (self.tile if self.tile else self.long_side)

    @classmethod
    def from_config(cls, name: str, cfg: dict) -> "Arm":
        return cls(name=name, tile=cfg["tile"], imgsz=cfg["imgsz"], overlap=cfg["overlap"])


def run_arm(
    detector: Detector,
    image: np.ndarray,
    arm: Arm,
    truncation_margin: float | None = None,
) -> RawDetections:
    """Roda um braço sobre uma imagem e devolve as detecções brutas em coordenadas globais.

    Os tiles vão numa única chamada, em lote, que é como se faria num Jetson — medir tile a
    tile superestimaria o custo do tiling. Ressalva para o relatório: uma engine TensorRT é
    construída com batch fixo, então uma engine de batch 1 perderia essa amortização e o custo
    dos braços com tile subiria.

    Sobre as duas parcelas de tempo: ``slice`` mede perto de zero, porque ``crop_tiles``
    devolve VIEWS e não cópias — a cópia real acontece dentro do pré-processamento do
    detector e é cobrada em ``infer``. A soma está certa; a repartição não. Quem consome estes
    números deve somar as duas, e é o que ``03_eval_arms.py`` faz.
    """
    if truncation_margin is None:
        # Vem da configuração, não de um default no código. O README promete "alterar aqui,
        # nunca no código"; com o valor fixo na assinatura, editar o YAML não fazia nada.
        from src.utils.config import experiment

        truncation_margin = float(
            experiment()["merge"]["default"].get("truncation_margin_px", 2.0)
        )

    h, w = image.shape[:2]
    image_wh = (w, h)

    t0 = time.perf_counter()
    if arm.tile is None:
        grid = np.array([[0, 0, w, h]], dtype=np.int32)
        crops = [image]
    else:
        grid = tile_grid(image_wh, arm.tile, arm.overlap)
        crops = crop_tiles(image, grid)
    t_slice = (time.perf_counter() - t0) * 1e3

    t0 = time.perf_counter()
    per_crop = detector.predict(crops, imgsz=arm.imgsz)
    t_infer = (time.perf_counter() - t0) * 1e3

    # `Detector` é um Protocol: nada garante que uma implementação devolva um resultado
    # por recorte, nem que caixas e scores tenham o mesmo tamanho. Sem estas checagens um
    # descasamento vira perda silenciosa de tiles ou — pior — arrays de tamanhos
    # diferentes dentro de RawDetections, que o `store` grava com deslocamentos errados e
    # faz TODAS as imagens seguintes carregarem scores trocados, sem erro nenhum.
    if len(per_crop) != len(grid):
        raise ValueError(
            f"detector devolveu {len(per_crop)} resultados para {len(grid)} tiles"
        )

    boxes_all, scores_all, tile_all, trunc_all = [], [], [], []
    for idx, ((boxes, scores), tile) in enumerate(zip(per_crop, grid)):
        if len(boxes) != len(scores):
            raise ValueError(f"tile {idx}: {len(boxes)} caixas e {len(scores)} scores")
        if len(boxes) == 0:
            continue
        global_boxes = to_global(np.asarray(boxes, dtype=np.float32), tile)
        boxes_all.append(global_boxes)
        scores_all.append(np.asarray(scores, dtype=np.float32))
        tile_all.append(np.full(len(global_boxes), idx, dtype=np.int32))
        trunc_all.append(
            is_truncated(global_boxes, tile, image_wh, margin=truncation_margin)
            if arm.tile is not None
            else np.zeros(len(global_boxes), dtype=bool)
        )

    if not boxes_all:
        return RawDetections(
            boxes=np.zeros((0, 4), np.float32),
            scores=np.zeros(0, np.float32),
            tile_index=np.zeros(0, np.int32),
            truncated=np.zeros(0, bool),
            n_tiles=len(grid),
            latency_ms={"slice": t_slice, "infer": t_infer},
        )

    return RawDetections(
        boxes=np.concatenate(boxes_all),
        scores=np.concatenate(scores_all),
        tile_index=np.concatenate(tile_all),
        truncated=np.concatenate(trunc_all),
        n_tiles=len(grid),
        latency_ms={"slice": t_slice, "infer": t_infer},
    )


def resolve_device(requested: str | None = None) -> str:
    """Traduz o device pedido para um que exista nesta máquina.

    Os scripts assumem `cuda:0` porque foi onde tudo rodou. Mas o repositório também precisa
    rodar num clone sem GPU — container, CI, notebook de quem for revisar. Sem esta tradução
    o Ultralytics levanta `Invalid CUDA 'device=0' requested` e o pipeline inteiro morre numa
    máquina que teria feito o trabalho em CPU sem reclamar, só mais devagar.
    """
    import torch

    if requested and requested.lower().startswith("cpu"):
        return "cpu"
    if torch.cuda.is_available():
        return requested or "cuda:0"
    if requested:
        print(f"  (nenhuma GPU visível — '{requested}' virou 'cpu')")
    return "cpu"


class UltralyticsDetector:
    """Adaptador para um modelo Ultralytics.

    Importado preguiçosamente para que os testes do motor rodem sem torch instalado.
    """

    def __init__(
        self,
        weights: str,
        device: str | None = None,
        conf: float = RAW_CONF,
        warmup: bool = True,
    ):
        # Aqui, e não no topo dos scripts: é o único ponto do projeto que realmente exige
        # torch. A análise offline (`03_eval_arms.py --analyze`) roda sobre as detecções já
        # gravadas e não deve morrer por causa dele. Sem esta checagem o WinError 1114 sai
        # como um erro de DLL sem contexto, no meio do import do ultralytics.
        from src.utils.torch_first import require_torch

        require_torch()
        from ultralytics import YOLO

        self.model = YOLO(weights)
        self.device = resolve_device(device)
        self.conf = conf
        if warmup:
            self._warmup()

    def _warmup(self) -> None:
        """Passe descartado, para tirar o custo de inicialização das medidas de latência.

        A primeira chamada paga transferência de pesos para o device, fusão de camadas e
        autotune do cuDNN. Em CPU isso mediu 332 ms contra ~75 ms estáveis; em GPU o custo de
        primeira chamada é de 0,5 a 3 s, contra poucos milissegundos por passe.

        Sem o aquecimento esse custo cairia inteiro na primeira imagem do PRIMEIRO braço
        avaliado — que é o braço de imagem inteira, justamente o que a comparação deveria
        mostrar como o mais barato. A tabela de latência ficaria enviesada contra ele.
        """
        blank = np.zeros((64, 64, 3), dtype=np.uint8)
        for imgsz in (640, 1280):
            self.model.predict(
                [blank], imgsz=imgsz, conf=0.99, device=self.device, verbose=False
            )

    def predict(self, images: list[np.ndarray], imgsz: int):
        """Recebe RGB e entrega BGR ao Ultralytics.

        Esta conversão não é cosmética. Ao receber um array NumPy, o Ultralytics assume que
        ele está em BGR — a convenção do ``cv2.imread`` — e inverte os canais no
        pré-processamento para entregar RGB ao modelo. Passar RGB direto faz o modelo receber
        BGR, com os canais trocados.

        O efeito medido numa imagem real com 95 maçãs anotadas, mesmos pesos, ``conf=0.25``:

            RGB (errado) ...  1 detecção
            BGR (correto) ..  5 detecções   <- igual a passar o caminho do arquivo

        O resto do pipeline trabalha em RGB, que é o que o PIL entrega e o que o matplotlib
        espera nas figuras. A conversão fica aqui, na fronteira com a biblioteca, que é onde
        a convenção muda.

        A cópia contígua é necessária: ``[..., ::-1]`` produz stride negativo, que várias
        rotinas do OpenCV recusam. Ela também é o custo real de recorte que ``run_arm`` não
        conseguia medir, já que ``crop_tiles`` devolve views.
        """
        results = self.model.predict(
            [np.ascontiguousarray(im[..., ::-1]) if im.ndim == 3 else im for im in images],
            imgsz=imgsz,
            conf=self.conf,
            max_det=MAX_DET,
            device=self.device,
            verbose=False,
        )
        return [
            (r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy())
            if r.boxes is not None and len(r.boxes)
            else (np.zeros((0, 4), np.float32), np.zeros(0, np.float32))
            for r in results
        ]


def arms_from_config(cfg: dict) -> list[Arm]:
    return [Arm.from_config(name, block) for name, block in cfg["arms"].items()]
