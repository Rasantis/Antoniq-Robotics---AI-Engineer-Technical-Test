"""Carregamento das configurações. Nenhum caminho ou constante fica hard-coded no código."""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]


def _load(name: str) -> dict[str, Any]:
    with open(REPO_ROOT / "configs" / name, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


@lru_cache(maxsize=1)
def paths() -> "Paths":
    """Todos os caminhos do YAML, com os relativos ancorados na raiz do repositório.

    A âncora é o repositório e não o diretório de trabalho de propósito: `ml/minneapple` tem
    que apontar para o mesmo lugar rodando de `scripts/`, de `demo/` ou da raiz. Antes só
    `results_root` passava por essa resolução — `dataset_root` e `runs_root` eram usados crus,
    então um valor relativo neles teria dependido de onde o comando foi disparado.
    """
    return Paths(**{k: _ancorado(Path(v)) for k, v in _load("paths.yaml").items()})


def _ancorado(p: Path) -> Path:
    return p if p.is_absolute() else REPO_ROOT / p


@lru_cache(maxsize=1)
def experiment() -> dict[str, Any]:
    return _load("experiment.yaml")


@dataclass(frozen=True)
class Paths:
    dataset_root: Path
    runs_root: Path
    results_root: Path
    figures_root: Path

    @property
    def detection(self) -> Path:
        return self.dataset_root / "detection"

    @property
    def train_images(self) -> Path:
        return self.detection / "train" / "images"

    @property
    def train_masks(self) -> Path:
        return self.detection / "train" / "masks"

    @property
    def test_images(self) -> Path:
        return self.detection / "test" / "images"

    def resolve(self, p: Path | str) -> Path:
        """Caminhos relativos são resolvidos contra a raiz do repositório."""
        p = Path(p)
        return p if p.is_absolute() else REPO_ROOT / p
