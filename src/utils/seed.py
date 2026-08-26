"""Semente única para random/numpy/torch. Chamada no início de todo script."""
from __future__ import annotations

import os
import random

import numpy as np


def set_seed(seed: int, deterministic: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    # O torch e opcional aqui: a preparacao de dados e NumPy/Pillow puro e nao deve falhar
    # porque uma biblioteca de GPU nao inicializou. Alem de ImportError, capturamos OSError
    # porque no Windows um torch instalado mas com DLL indisponivel levanta WinError 1114 --
    # observado nesta maquina sob contencao de recurso.
    try:
        import torch
    except (ImportError, OSError) as exc:
        print(f"  aviso: semente do torch nao aplicada ({type(exc).__name__}); "
              f"etapas que usam GPU vao falhar, as de CPU seguem deterministicas")
        return

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        # cuBLAS exige isto para reprodutibilidade em algumas operações de GEMM.
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
