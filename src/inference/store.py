"""Persistência das detecções brutas, antes de qualquer fusão.

Existe para desacoplar o que é caro do que é barato. A inferência dos quatro braços sobre os
três folds custa alguns minutos de GPU; a ablação de fusão tem 24 combinações e a calibração
de contagem varre 13 limiares. Rodar o detector 24 x 13 vezes seria absurdo — então ele roda
UMA vez, num limiar de confiança baixo, e o resultado bruto vai para disco.

Formato: um único ``.npz`` por (fold, braço), com os arrays de todas as imagens concatenados e
um vetor de deslocamentos no estilo CSR. Guardar um arquivo por imagem geraria dezenas de
milhares de arquivos pequenos, que no Windows é lento de escrever e pior ainda de ler.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from src.inference.engine import RawDetections


def save(path: Path, per_image: dict[str, RawDetections]) -> None:
    """Grava as detecções brutas de um (fold, braço)."""
    names = sorted(per_image)
    offsets = np.zeros(len(names) + 1, dtype=np.int64)
    for i, name in enumerate(names):
        offsets[i + 1] = offsets[i] + len(per_image[name])

    def stack(attr: str, dtype) -> np.ndarray:
        parts = [getattr(per_image[n], attr) for n in names]
        parts = [p for p in parts if len(p)]
        if not parts:
            shape = (0, 4) if attr == "boxes" else (0,)
            return np.zeros(shape, dtype=dtype)
        return np.concatenate(parts).astype(dtype)

    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        image_names=np.array(names, dtype=object),
        offsets=offsets,
        boxes=stack("boxes", np.float32),
        scores=stack("scores", np.float32),
        tile_index=stack("tile_index", np.int32),
        truncated=stack("truncated", bool),
        n_tiles=np.array([per_image[n].n_tiles for n in names], dtype=np.int32),
        latency=np.array(
            [
                [per_image[n].latency_ms.get("slice", 0.0),
                 per_image[n].latency_ms.get("infer", 0.0)]
                for n in names
            ],
            dtype=np.float32,
        ),
    )


def load(path: Path) -> dict[str, RawDetections]:
    """Lê de volta o que ``save`` gravou."""
    data = np.load(path, allow_pickle=True)
    names = [str(n) for n in data["image_names"]]
    offsets = data["offsets"]

    out: dict[str, RawDetections] = {}
    for i, name in enumerate(names):
        a, b = int(offsets[i]), int(offsets[i + 1])
        out[name] = RawDetections(
            boxes=data["boxes"][a:b],
            scores=data["scores"][a:b],
            tile_index=data["tile_index"][a:b],
            truncated=data["truncated"][a:b],
            n_tiles=int(data["n_tiles"][i]),
            latency_ms={
                "slice": float(data["latency"][i, 0]),
                "infer": float(data["latency"][i, 1]),
            },
        )
    return out


def path_for(
    root: Path, fold: int, arm: str, split: str = "test", tag: str = "grouped"
) -> Path:
    """Caminho do arquivo bruto de um (esquema de split, fold, braço, conjunto).

    O ``split`` no nome é o que permite manter validação e teste separados em disco — sem
    isso, a seleção do ponto de operação acabaria lendo o conjunto de teste.
    """
    return root / "raw_detections" / f"{tag}_fold{fold}_{arm}_{split}.npz"
