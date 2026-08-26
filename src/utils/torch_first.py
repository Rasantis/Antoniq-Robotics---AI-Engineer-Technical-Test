"""Importa o PyTorch antes do pandas. Precisa ser o primeiro import não-stdlib do processo.

Existe por um defeito concreto e reprodutível deste ambiente Windows: importar ``pandas``
antes de ``torch`` faz o torch falhar com

    OSError: [WinError 1114] Uma rotina de inicialização da biblioteca de vínculo dinâmico
    (DLL) falhou. Error loading "...\\torch\\lib\\c10.dll" or one of its dependencies.

Medido, e determinístico — não é intermitência nem pressão de memória, que foi a primeira
hipótese e estava errada:

    import torch, pandas          -> ok
    import pandas; import torch   -> FALHA
    import numpy; import torch    -> ok
    import cv2; import torch      -> ok
    import sklearn; import torch  -> ok
    import scipy; import torch    -> ok
    import PIL; import torch      -> ok

Só o pandas dispara. É o conflito clássico de runtime OpenMP: o pandas carrega uma versão de
``libiomp5md.dll`` e a inicialização da DLL do torch não sobrevive a ela. ``KMP_DUPLICATE_LIB_OK
=TRUE``, que é o paliativo mais citado para essa família de erro, foi testado e não resolve.

O que resolve é a ordem. Por isso este módulo é importado antes de qualquer coisa que possa
puxar pandas — inclusive indiretamente, através de ``src.eval``.

O import é tolerante a falha de propósito: as etapas de preparação de dados e de análise são
NumPy/Pillow/pandas puro e não devem morrer porque uma biblioteca de GPU não está instalada
ou não inicializa.
"""
from __future__ import annotations

TORCH_AVAILABLE = False
TORCH_ERROR: Exception | None = None

try:
    import torch  # noqa: F401

    TORCH_AVAILABLE = True
except Exception as exc:  # ImportError, OSError (WinError 1114), e o que mais aparecer
    TORCH_ERROR = exc


def require_torch() -> None:
    """Falha alto, e com a explicação, quando uma etapa realmente precisa de GPU."""
    if not TORCH_AVAILABLE:
        raise SystemExit(
            f"PyTorch nao pode ser carregado: {type(TORCH_ERROR).__name__}: {TORCH_ERROR}\n"
            f"Se for WinError 1114, algum modulo importou pandas antes deste. "
            f"`from src.utils import torch_first` tem de ser o primeiro import nao-stdlib."
        )
