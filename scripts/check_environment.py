"""Verificação de ambiente. Roda antes de qualquer etapa cara.

Existe por um motivo concreto encontrado nesta máquina: ``importlib.metadata`` reportava
numpy 2.2.6 e opencv-python 4.11.0.86, enquanto o que estava de fato importado era numpy 2.3.5
e OpenCV 4.10.0.84 (do pacote opencv-contrib-python). A causa eram diretórios ``.dist-info``
obsoletos, deixados por atualizações incompletas.

A consequência prática é séria para um trabalho avaliado por reprodutibilidade: um
``pip freeze`` teria pinado versões que não são as que produziram os resultados. Por isso as
versões aqui vêm sempre de ``module.__version__``, que é a única fonte confiável, e nunca dos
metadados de pacote.

Uso:
    python scripts/check_environment.py
"""
from __future__ import annotations

import platform
import sys

# (rótulo, módulo, atributo de versão, nome do pacote no PyPI)
CHECKS = [
    ("NumPy", "numpy", "__version__", "numpy"),
    ("OpenCV", "cv2", "__version__", "opencv-contrib-python"),
    ("Pillow", "PIL", "__version__", "pillow"),
    ("PyTorch", "torch", "__version__", "torch"),
    ("Ultralytics", "ultralytics", "__version__", "ultralytics"),
    ("pandas", "pandas", "__version__", "pandas"),
    ("scikit-learn", "sklearn", "__version__", "scikit-learn"),
    ("SciPy", "scipy", "__version__", "scipy"),
    ("Matplotlib", "matplotlib", "__version__", "matplotlib"),
]


def collect() -> dict[str, str]:
    import importlib

    found = {}
    for label, module, attr, _pypi in CHECKS:
        try:
            found[label] = getattr(importlib.import_module(module), attr, "?")
        except ImportError:
            found[label] = "AUSENTE"
    return found


def main() -> int:
    print(f"Python      {sys.version.split()[0]}  ({platform.platform()})")
    versions = collect()
    for label, version in versions.items():
        print(f"{label:<13} {version}")

    try:
        import torch

        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            total = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(f"{'GPU':<13} {name} ({total:.1f} GiB)")
        else:
            print(f"{'GPU':<13} indisponivel - o treino vai rodar em CPU e sera inviavel")
    except ImportError:
        pass

    missing = [k for k, v in versions.items() if v == "AUSENTE"]
    if missing:
        print(f"\nFALTANDO: {', '.join(missing)}")
        print("Instale com: pip install -r requirements.txt")
        return 1

    # Aviso sobre a divergencia entre metadados e runtime, se ela existir.
    import importlib.metadata as md

    stale = []
    for label, module, attr, pypi in CHECKS:
        try:
            meta = md.version(pypi)
        except md.PackageNotFoundError:
            continue
        runtime = versions[label]
        if runtime != "AUSENTE" and not meta.startswith(runtime.split("+")[0][:6]):
            stale.append(f"{pypi}: metadado {meta} != runtime {runtime}")
    if stale:
        print("\nAVISO - metadados de pacote inconsistentes com o que esta importado:")
        for line in stale:
            print(f"  {line}")
        print("  As versoes validas sao as de runtime, acima. Um `pip freeze` mentiria aqui.")

    print("\nAmbiente OK.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
