"""AutoFE method adapters: a common fit_transform/transform contract wrapping
the baseline (no-op) and three external AutoFE libraries under benchmark.

Every adapter consumes an already-numeric, NaN-free matrix (categoricals
ordinal-coded, numerics median-imputed -- see ``prep_for_generation``) and
returns an expanded numeric matrix (original columns + engineered columns).
This keeps the three libraries' differing native-dtype support out of the
benchmark loop: whatever a method can or can't do with raw categoricals/NaN
is normalized away before it ever sees the data, and any model-family-specific
encoding of the *result* (scaling, NaN-from-generated-features cleanup) is
applied afterwards by the benchmark runner, not here.

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


class _IsolatedCwd:
    """chdir into a fresh temp directory for the duration of the block.

    openfe's ``fit()``/``transform()`` write to *hardcoded, relative* temp
    filenames (``./openfe_tmp_data.feather`` in ``transform()``, with no
    override parameter at all). A leftover file from a previous call in the
    same cwd would otherwise collide with a later one. Each call gets its
    own fresh directory, so the relative path always resolves to a new file.
    """

    def __enter__(self):
        import os
        import tempfile

        self._prev_cwd = os.getcwd()
        self._tmpdir = tempfile.mkdtemp(prefix="openfe_")
        os.chdir(self._tmpdir)
        return self

    def __exit__(self, *exc_info):
        import os
        import shutil

        os.chdir(self._prev_cwd)
        shutil.rmtree(self._tmpdir, ignore_errors=True)


class OpenFEMethod:
    name = "openfe"

    def __init__(self, n_new_features: int = 10, n_jobs: int = 1):
        self.n_new_features = n_new_features
        self.n_jobs = n_jobs
        self._features = None
        self._X_train = None

    def fit_transform(self, X_train, y_train, task):
        from openfe import OpenFE, transform

        self._X_train = X_train
        task_str = "regression" if task == "regression" else "classification"
        with _IsolatedCwd():
            ofe = OpenFE()
            self._features = ofe.fit(data=X_train, label=y_train, task=task_str,
                                      n_jobs=self.n_jobs, seed=0, verbose=False)
            selected = self._features[: self.n_new_features]
            X_train_new, _ = transform(X_train, X_train, selected, n_jobs=self.n_jobs)
        self._selected = selected
        return X_train_new

    def transform(self, X_test):
        from openfe import transform

        with _IsolatedCwd():
            _, X_test_new = transform(self._X_train, X_test, self._selected,
                                       n_jobs=self.n_jobs)
        return X_test_new


class FeaturetoolsMethod:
    """Single-table deep feature synthesis (no relational structure available,
    so this is effectively automated pairwise numeric transforms)."""

    name = "featuretools"

    def __init__(self, max_depth: int = 1):
        self.max_depth = max_depth
        self._feature_defs = None

    def _entityset(self, X: pd.DataFrame, name: str):
        import featuretools as ft

        df = X.reset_index(drop=True).copy()
        df["__id__"] = df.index
        es = ft.EntitySet(id=name)
        es = es.add_dataframe(dataframe_name="data", dataframe=df, index="__id__")
        return es

    def fit_transform(self, X_train, y_train, task):
        import featuretools as ft

        es = self._entityset(X_train, "train")
        fm, feature_defs = ft.dfs(
            entityset=es, target_dataframe_name="data",
            trans_primitives=["add_numeric", "subtract_numeric",
                               "multiply_numeric", "divide_numeric"],
            max_depth=self.max_depth, n_jobs=1, verbose=False,
        )
        self._feature_defs = feature_defs
        return fm.reset_index(drop=True)

    def transform(self, X_test):
        import featuretools as ft

        es = self._entityset(X_test, "test")
        fm = ft.calculate_feature_matrix(features=self._feature_defs, entityset=es)
        return fm.reset_index(drop=True)


class AutofeatMethod:
    name = "autofeat"

    def __init__(self, feateng_steps: int = 1):
        self.feateng_steps = feateng_steps
        self._model = None

    def fit_transform(self, X_train, y_train, task):
        from autofeat import AutoFeatClassifier, AutoFeatRegressor

        cls = AutoFeatRegressor if task == "regression" else AutoFeatClassifier
        self._model = cls(feateng_steps=self.feateng_steps, n_jobs=1, verbose=0)
        out = self._model.fit_transform(X_train, y_train)
        return self._to_frame(out, X_train.index)

    def transform(self, X_test):
        out = self._model.transform(X_test)
        return self._to_frame(out, X_test.index)

    @staticmethod
    def _to_frame(out, index) -> pd.DataFrame:
        # autofeat returns a DataFrame with its own (reset) index -- wrapping
        # it as `pd.DataFrame(out, index=index)` would silently re-align by
        # *label*, scrambling rows wherever the label sets differ. Go through
        # a bare ndarray so the caller's index is applied positionally.
        columns = out.columns if isinstance(out, pd.DataFrame) else None
        return pd.DataFrame(np.asarray(out), index=index,
                            columns=columns).replace([np.inf, -np.inf], np.nan)


METHODS: dict[str, type] = {
    "baseline": BaselineMethod,
    "openfe": OpenFEMethod,
    "featuretools": FeaturetoolsMethod,
    "autofeat": AutofeatMethod,
}
