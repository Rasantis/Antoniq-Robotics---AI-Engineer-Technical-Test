"""Splits agrupados por sessão de captura.

Esta é a peça mais importante do trabalho do ponto de vista metodológico.

As 670 imagens rotuladas do MinneApple não são independentes: são quadros extraídos de dez
vídeos, gravados caminhando ao longo de fileiras de macieira a cerca de 1 m/s, com um quadro
a cada cinco (Häni, Roy & Isler, 2019, §III). Quadros vizinhos de um mesmo vídeo mostram as
mesmas maçãs, dos mesmos ângulos, sob a mesma luz.

Um split aleatório por imagem coloca quadros quase idênticos em treino e em teste. A métrica
resultante mede memorização, não generalização — e é exatamente o erro que o enunciado cobra
em "clean splits, honest metrics".

O agrupamento é pela sessão de captura, que o próprio nome do arquivo entrega:

    20150921_131453_image1101.png
    |_____________|       |____|
        sessão            índice do quadro no vídeo

São exatamente dez sessões, correspondendo aos dez vídeos descritos no artigo.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, KFold

# Treino rotulado: 20150921_131453_image1101.png
_TRAIN_RE = re.compile(r"^(?P<session>\d{8}_\d{6})_image(?P<frame>\d+)$")
# Teste oficial, sem rótulos públicos: dataset1_back_1021.png
_TEST_RE = re.compile(r"^(?P<session>dataset\d+_(?:front|back))_(?P<frame>\d+)$")


def parse_name(filename: str) -> tuple[str, int]:
    """(sessão, índice do quadro) a partir do nome do arquivo."""
    stem = Path(filename).stem
    for pattern in (_TRAIN_RE, _TEST_RE):
        m = pattern.match(stem)
        if m is not None:
            return m["session"], int(m["frame"])
    raise ValueError(f"nome de arquivo fora do padrao MinneApple: {filename!r}")


def session_table(filenames: list[str]) -> pd.DataFrame:
    """Tabela imagem / sessão / índice do quadro, ordenada temporalmente dentro da sessão."""
    parsed = [parse_name(f) for f in filenames]
    return pd.DataFrame(
        {
            "image": filenames,
            "session": [s for s, _ in parsed],
            "frame": [f for _, f in parsed],
        }
    ).sort_values(["session", "frame"], ignore_index=True)


@dataclass(frozen=True)
class Fold:
    index: int
    train: list[str]
    val: list[str]
    test: list[str]

    @property
    def sessions(self) -> dict[str, list[str]]:
        return {
            name: sorted({parse_name(f)[0] for f in files})
            for name, files in (
                ("train", self.train),
                ("val", self.val),
                ("test", self.test),
            )
        }

    def as_record(self) -> dict:
        s = self.sessions
        return {
            "fold": self.index,
            "n_train": len(self.train),
            "n_val": len(self.val),
            "n_test": len(self.test),
            "train_sessions": ",".join(s["train"]),
            "val_sessions": ",".join(s["val"]),
            "test_sessions": ",".join(s["test"]),
        }


def _pick_val_session(
    table: pd.DataFrame, sessions: list[str], target: float, already_used: set[str]
) -> str:
    """Sessão de validação: a mais próxima de ``target`` do trainval, sem repetir entre folds.

    Determinístico e sem sorteio — desempate por nome — para que o split seja reproduzível
    sem depender de semente.

    ``already_used`` evita que a mesma sessão vire validação em mais de um fold. Escolher só
    por tamanho fazia dois dos três folds usarem exatamente a mesma sessão para early
    stopping, e por acaso a mais atípica do dataset (98,1 maçãs por imagem contra média de
    42,1). Os folds deixavam de ser réplicas independentes, e o desvio entre eles passava a
    subestimar a variação real.
    """
    counts = table[table.session.isin(sessions)].session.value_counts()
    total = counts.sum()
    fresh = [s for s in sessions if s not in already_used] or sessions
    return min(fresh, key=lambda s: (abs(counts[s] / total - target), s))


def make_folds(
    filenames: list[str], n_folds: int = 3, val_fraction: float = 0.15
) -> list[Fold]:
    """GroupKFold sobre as sessões: nenhuma sessão aparece em dois conjuntos.

    Cada imagem cai no conjunto de teste de exatamente um fold, então as métricas podem ser
    agrupadas sobre as 670 imagens e ainda assim reportadas com desvio entre folds.

    A validação também sai de uma sessão inteira e separada: usá-la para early stopping com
    quadros do mesmo vídeo do treino reintroduziria o vazamento pela porta dos fundos.
    """
    table = session_table(filenames)
    splitter = GroupKFold(n_splits=n_folds)

    folds: list[Fold] = []
    used_for_val: set[str] = set()
    for i, (trainval_idx, test_idx) in enumerate(
        splitter.split(table.image, groups=table.session)
    ):
        trainval, test = table.iloc[trainval_idx], table.iloc[test_idx]
        val_session = _pick_val_session(
            table, sorted(trainval.session.unique()), val_fraction, used_for_val
        )
        used_for_val.add(val_session)
        folds.append(
            Fold(
                index=i,
                train=trainval[trainval.session != val_session].image.tolist(),
                val=trainval[trainval.session == val_session].image.tolist(),
                test=test.image.tolist(),
            )
        )
    return folds


def make_random_folds(
    filenames: list[str], n_folds: int = 3, seed: int = 1337
) -> list[Fold]:
    """Split aleatório por imagem, ignorando a sessão.

    Existe para uma coisa só: quantificar o quanto a metodologia errada infla a métrica.
    O relatório reporta os dois lado a lado. Não usar para o modelo final.
    """
    table = session_table(filenames)
    splitter = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    rng = np.random.default_rng(seed)

    folds: list[Fold] = []
    for i, (trainval_idx, test_idx) in enumerate(splitter.split(table.image)):
        trainval = table.iloc[trainval_idx]
        n_val = max(1, int(round(len(trainval) * 0.15)))
        is_val = np.zeros(len(trainval), dtype=bool)
        is_val[rng.choice(len(trainval), size=n_val, replace=False)] = True
        folds.append(
            Fold(
                index=i,
                train=trainval.image[~is_val].tolist(),
                val=trainval.image[is_val].tolist(),
                test=table.iloc[test_idx].image.tolist(),
            )
        )
    return folds


def assert_no_session_leak(folds: list[Fold]) -> None:
    """Falha alto se qualquer sessão vazar entre conjuntos. Chamado por 01_prepare_data.py."""
    for fold in folds:
        s = fold.sessions
        for a, b in (("train", "test"), ("val", "test"), ("train", "val")):
            overlap = set(s[a]) & set(s[b])
            if overlap:
                raise AssertionError(
                    f"fold {fold.index}: sessoes {sorted(overlap)} aparecem em {a} e em {b}"
                )


def assert_test_covers_all(folds: list[Fold], filenames: list[str]) -> None:
    """Cada imagem deve ser testada exatamente uma vez ao longo dos folds."""
    seen: list[str] = []
    for fold in folds:
        seen.extend(fold.test)
    if sorted(seen) != sorted(filenames):
        raise AssertionError(
            f"cobertura do teste incorreta: {len(seen)} imagens testadas, "
            f"{len(set(seen))} distintas, {len(filenames)} no dataset"
        )
