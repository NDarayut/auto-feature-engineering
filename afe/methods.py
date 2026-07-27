"""The AutoFE method contract, plus the one method this harness ships.

No third-party AutoFE library is bundled here. ``BaselineMethod`` (no-op) is
the reference arm every comparison needs; every other method is supplied by
you, as a plain function or an ``AutoFEMethod``-shaped object -- see
"Plugging in a method" in the repo-root README.

A method consumes an already-numeric, NaN-free matrix (categoricals
ordinal-coded, numerics median-imputed -- see ``prep_for_generation``) and
returns an expanded numeric matrix (original columns + engineered columns).
That keeps differing native-dtype support out of the benchmark loop: whatever
a method can or can't do with raw categoricals/NaN is normalized away before
it ever sees the data, and any model-family-specific encoding of the *result*
(scaling, NaN-from-generated-features cleanup) is applied afterwards by the
benchmark runner, not here.

``fit_transform`` is only ever called on the training fold; ``transform`` is
only ever called on the held-out fold using state fit during ``fit_transform``
-- this is the leak-safety contract every adapter must uphold.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder


def prep_for_generation(
    X_train: pd.DataFrame, X_test: pd.DataFrame, categorical_cols: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Ordinal-encode categoricals + median-impute numerics, fit on train only.

    Produces the fully-numeric, NaN-free input every AutoFE method here
    expects, without embedding encoding/imputation choices in each adapter.
    """
    num_cols = [c for c in X_train.columns if c not in categorical_cols]
    Xtr, Xte = X_train.copy(), X_test.copy()

    if categorical_cols:
        enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1,
                              encoded_missing_value=-2)
        enc.fit(Xtr[categorical_cols])
        Xtr[categorical_cols] = enc.transform(Xtr[categorical_cols])
        Xte[categorical_cols] = enc.transform(Xte[categorical_cols])
    if num_cols:
        imp = SimpleImputer(strategy="median")
        imp.fit(Xtr[num_cols])
        Xtr[num_cols] = imp.transform(Xtr[num_cols])
        Xte[num_cols] = imp.transform(Xte[num_cols])
    return Xtr.astype(float), Xte.astype(float)


class AutoFEMethod(Protocol):
    name: str

    def fit_transform(self, X_train: pd.DataFrame, y_train: pd.Series,
                       task: str) -> pd.DataFrame: ...

    def transform(self, X_test: pd.DataFrame) -> pd.DataFrame: ...


class BaselineMethod:
    """No feature engineering -- the raw (prepped) features, unchanged."""

    name = "baseline"

    def fit_transform(self, X_train, y_train, task):
        return X_train

    def transform(self, X_test):
        return X_test


METHODS: dict[str, type] = {
    "baseline": BaselineMethod,
}


def resolve_method(name: str):
    """Resolve a method name to a callable that constructs the method.

    Accepts a built-in key (``"baseline"``) or an import path to your own
    method -- ``"mypkg.methods.MyMethod"`` or ``"mypkg.methods:MyMethod"``.
    The target may be a class (instantiated with no arguments) or a plain
    ``(X_train, y_train, X_test, task)`` function, which is wrapped to
    satisfy the ``AutoFEMethod`` contract.

    Raises ``KeyError`` with the available built-ins if ``name`` is neither a
    built-in nor an importable path.
    """
    if name in METHODS:
        return METHODS[name]
    if "." not in name and ":" not in name:
        raise KeyError(
            f"unknown method {name!r}; built-ins are {sorted(METHODS)}. "
            "To benchmark your own method, pass an import path such as "
            "'mypkg.methods:MyMethod'.")

    import importlib

    module_path, _, attr = name.rpartition(":") if ":" in name else name.rpartition(".")
    try:
        obj = getattr(importlib.import_module(module_path), attr)
    except (ImportError, AttributeError) as exc:
        raise KeyError(f"could not import method {name!r}: {exc}") from exc

    if isinstance(obj, type):
        return obj
    # A plain function: adapt it to the fit_transform/transform contract by
    # deferring generation until transform() has both folds available.
    class _FunctionMethod:
        name = getattr(obj, "__name__", attr)

        def fit_transform(self, X_train, y_train, task):
            self._train, self._y, self._task = X_train, y_train, task
            out, _ = obj(X_train, y_train, X_train, task)
            return out

        def transform(self, X_test):
            _, out = obj(self._train, self._y, X_test, self._task)
            return out

    return _FunctionMethod
