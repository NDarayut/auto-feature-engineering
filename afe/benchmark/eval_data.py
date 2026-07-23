"""The benchmark-ready, per-fold data contract: ``iter_folds(key, encoding)``.

Ties together ``afe.download.load`` (raw fetch/cache), ``afe.splits`` (frozen
per-dataset CV/seed-repeat protocol), and ``afe.encoders`` (per-model-family,
train-fold-only encoding) into the single entry point any AutoFE method or
model can use to get ready-to-train folds:

    from afe.eval_data import iter_folds
    for fold in iter_folds("nomao", encoding="tree"):
        model.fit(fold.X_train, fold.y_train)
        score(model, fold.X_test, fold.y_test)

Leak-safety by construction: rows are subset by fold index *before* any
encoder is touched, a fresh encoder instance is created per fold, and it is
fit on the training subset only -- so no fold's fit state, and no test row,
can ever influence another fold's encoding.
"""

from __future__ import annotations

import dataclasses
from typing import Iterator, Literal

import numpy as np
import pandas as pd

from ..encoders import ENCODER_PROFILES
from .download import load
from .registry import BENCHMARK, DatasetSpec
from .splits import iter_split_indices, protocol_for

_BY_KEY: dict[str, DatasetSpec] = {s.key: s for s in BENCHMARK}


@dataclasses.dataclass(frozen=True)
class Fold:
    key: str
    fold_id: str
    protocol: str
    encoding: str
    X_train: pd.DataFrame | np.ndarray
    y_train: pd.Series
    X_test: pd.DataFrame | np.ndarray
    y_test: pd.Series
    categorical_columns: tuple[str, ...]


def list_available_keys() -> list[str]:
    return [s.key for s in BENCHMARK]


def iter_folds(
    key: str,
    encoding: Literal["tree", "linear"] = "tree",
    categorical_encoding: str | None = None,
    use_cache: bool = True,
) -> Iterator[Fold]:
    """Yield leak-free ``Fold``s for ``key`` under its frozen split protocol.

    ``categorical_encoding`` overrides the encoder profile's default (e.g.
    pass ``"target"`` to use target encoding instead of ordinal/one-hot).
    """
    if key not in _BY_KEY:
        raise KeyError(f"unknown benchmark dataset key: {key!r}")
    spec = _BY_KEY[key]
    frame, meta = load(spec, use_cache=use_cache)
    target = meta["target"]
    y_full = frame[target]
    X_full = frame.drop(columns=[target])

    encoder_cls = ENCODER_PROFILES[encoding]
    encoder_kwargs = ({"categorical_encoding": categorical_encoding}
                      if categorical_encoding else {})
    protocol = protocol_for(spec)

    for fold_id, train_idx, test_idx in iter_split_indices(
        spec, n=len(X_full), y=y_full.to_numpy() if spec.task != "regression" else None,
    ):
        X_train = X_full.iloc[train_idx]
        y_train = y_full.iloc[train_idx]
        X_test = X_full.iloc[test_idx]
        y_test = y_full.iloc[test_idx]

        encoder = encoder_cls(**encoder_kwargs)
        X_train_enc = encoder.fit_transform(X_train, y_train)
        X_test_enc = encoder.transform(X_test)

        yield Fold(
            key=key, fold_id=fold_id, protocol=protocol, encoding=encoding,
            X_train=X_train_enc, y_train=y_train,
            X_test=X_test_enc, y_test=y_test,
            categorical_columns=encoder.categorical_columns,
        )
