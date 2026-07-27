"""Per-model-family encoders, fit on the training fold only.

Two profiles:

* ``TreeEncoder``  -- for boosted-tree / RF models. Default is ordinal-coding
  of categoricals with missing values passed through as NaN (LightGBM/XGBoost/
  CatBoost all handle NaN natively). ``categorical_encoding="target"`` swaps in
  leak-safe target encoding instead.
* ``LinearEncoder`` -- for linear/logistic/SVM/kNN models. Default is one-hot
  categoricals + standardized/imputed numerics. ``categorical_encoding="target"``
  swaps one-hot for target encoding (this profile's answer to Sec. 4's "one-hot
  or WoE" allowance -- see note below).

Target encoding (both profiles) uses ``sklearn.preprocessing.TargetEncoder``,
which internally cross-fits across folds of the *training* data during
``fit_transform`` -- i.e. leak-safe out-of-fold encoding without hand-rolled
nested-CV logic. This is used in place of hand-rolled Weight-of-Evidence:
for a binary target, TargetEncoder's per-category class-probability estimate
carries the same information WoE does (a category's relationship to the
target rate), with the leak-safety already built in.

Hard rule (Sec. 1/4/7): every encoder here is fit on the training fold only
and merely applied (``transform``) to the held-out fold -- never refit on it.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    OneHotEncoder,
    OrdinalEncoder,
    StandardScaler,
    TargetEncoder,
)

CATEGORICAL_DTYPES = ("object", "category", "bool")


def _split_columns(X: pd.DataFrame) -> tuple[list[str], list[str]]:
    """(categorical_cols, numeric_cols) by dtype -- mirrors download.py's
    ``_meta()`` categorical-count logic, so "categorical" means the same thing
    everywhere in the codebase."""
    cat = list(X.select_dtypes(include=list(CATEGORICAL_DTYPES)).columns)
    num = [c for c in X.columns if c not in cat]
    return cat, num


class TreeEncoder:
    """Ordinal (default) or target encoding for categoricals; numerics and
    missing values pass through unchanged (NaN-native boosters)."""

    def __init__(self, categorical_encoding: Literal["ordinal", "target"] = "ordinal"):
        self.categorical_encoding = categorical_encoding
        self._cat_cols: list[str] = []
        self._num_cols: list[str] = []
        self._encoder: OrdinalEncoder | TargetEncoder | None = None

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> "TreeEncoder":
        if self.categorical_encoding == "target" and y is None:
            raise ValueError("TreeEncoder(categorical_encoding='target') needs y")
        self._cat_cols, self._num_cols = _split_columns(X)
        if not self._cat_cols:
            return self
        if self.categorical_encoding == "ordinal":
            self._encoder = OrdinalEncoder(
                handle_unknown="use_encoded_value", unknown_value=-1,
                encoded_missing_value=-2)
            self._encoder.fit(X[self._cat_cols])
        else:
            self._encoder = TargetEncoder(random_state=0)
            self._encoder.fit(X[self._cat_cols], y)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        out = X.copy()
        if self._cat_cols:
            encoded = self._encoder.transform(X[self._cat_cols])
            out[self._cat_cols] = encoded
        return out

    def fit_transform(self, X: pd.DataFrame, y: pd.Series | None = None) -> pd.DataFrame:
        return self.fit(X, y).transform(X)

    @property
    def categorical_columns(self) -> tuple[str, ...]:
        return tuple(self._cat_cols)


class LinearEncoder:
    """One-hot (default) or target encoding for categoricals, standard-scaled
    + median/most-frequent-imputed numerics, all fit on train only."""

    def __init__(self, categorical_encoding: Literal["onehot", "target"] = "onehot"):
        self.categorical_encoding = categorical_encoding
        self._cat_cols: list[str] = []
        self._num_cols: list[str] = []
        self._pipeline: ColumnTransformer | None = None

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> "LinearEncoder":
        if self.categorical_encoding == "target" and y is None:
            raise ValueError("LinearEncoder(categorical_encoding='target') needs y")
        self._cat_cols, self._num_cols = _split_columns(X)

        cat_steps = [("impute", SimpleImputer(strategy="most_frequent"))]
        if self.categorical_encoding == "onehot":
            cat_steps.append(("encode", OneHotEncoder(handle_unknown="ignore",
                                                        sparse_output=False)))
        else:
            cat_steps.append(("encode", TargetEncoder(random_state=0)))
        num_steps = [("impute", SimpleImputer(strategy="median")),
                     ("scale", StandardScaler())]

        transformers = []
        if self._num_cols:
            transformers.append(("num", Pipeline(num_steps), self._num_cols))
        if self._cat_cols:
            transformers.append(("cat", Pipeline(cat_steps), self._cat_cols))
        self._pipeline = ColumnTransformer(transformers)
        self._pipeline.fit(X, y)
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        return self._pipeline.transform(X)

    def fit_transform(self, X: pd.DataFrame, y: pd.Series | None = None) -> np.ndarray:
        return self.fit(X, y).transform(X)

    @property
    def categorical_columns(self) -> tuple[str, ...]:
        return tuple(self._cat_cols)


ENCODER_PROFILES: dict[str, type] = {"tree": TreeEncoder, "linear": LinearEncoder}
