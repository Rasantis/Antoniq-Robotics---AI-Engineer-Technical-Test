import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# A mesma regra que os scripts seguem, aplicada aqui: o torch tem de ser importado antes do
# pandas neste Windows, senão falha com WinError 1114 ao carregar c10.dll. O conftest roda
# antes de qualquer módulo de teste, então é daqui que a ordem é garantida — sem isto, o
# primeiro teste a tocar em torch quebra porque outro módulo de teste já puxou pandas.
# Ver src/utils/torch_first.py para a medição.
from src.utils import torch_first  # noqa: E402,F401  isort:skip
