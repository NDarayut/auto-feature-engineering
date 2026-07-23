"""Tests for afe.meta.online.MFOpenFE -- the public online AutoFE path.

Uses the real trained models/meta_model.pkl (needs to exist -- produced by
scripts.train_meta_model) and the real openfe/lightgbm libraries, but no
network: synthetic in-memory data only.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import train_test_split

from afe import MFOpenFE
from afe.meta.online import COVERED_OPERATORS, _infer_task
from afe.meta.stage1 import DEFAULT_MODEL_PATH

pytestmark = pytest.mark.skipif(
    not DEFAULT_MODEL_PATH.exists(),
    reason="models/meta_model.pkl not present -- run scripts.train_meta_model first",
)


def _synthetic_regression(n: int = 400, seed: int = 0):
    rng = np.random.RandomState(seed)
    x1 = rng.uniform(1, 10, n)
    x2 = rng.uniform(1, 10, n)
    x3 = rng.uniform(-3, 3, n)
    noise = rng.randn(n) * 0.1
    # log(x1) and square(x3) are both in COVERED_OPERATORS -- a real, learnable
    # nonlinear signal the raw features alone under-explain.
    y = 3 * np.log(x1) + 2 * (x3 ** 2) + noise
    X = pd.DataFrame({"x1": x1, "x2": x2, "x3": x3})
    return train_test_split(X, y, test_size=0.3, random_state=seed)


def test_infer_task():
    assert _infer_task(pd.Series([0, 1, 0, 1, 1])) == "classification"
    assert _infer_task(pd.Series(np.random.RandomState(0).randn(200))) == "regression"


def test_covered_operators_are_a_subset_of_trained_operators():
    from afe.meta.stage1 import MetaModel

    model = MetaModel.load(DEFAULT_MODEL_PATH)
    assert COVERED_OPERATORS <= set(model.per_operator)


def test_fit_transform_finds_useful_engineered_features():
    X_train, X_test, y_train, y_test = _synthetic_regression()
    mfe = MFOpenFE(task="regression", progress=False, max_candidates=30,
                   filter_threshold=0.3)
    X_train_fe = mfe.fit_transform(X_train, y_train)

    assert mfe.n_candidates_generated_ > 0
    assert mfe.n_candidates_after_filter_ <= 30
    assert mfe.n_features_kept_ > 0
    assert X_train_fe.shape[0] == X_train.shape[0]
    assert X_train_fe.shape[1] == X_train.shape[1] + mfe.n_features_kept_


def test_transform_replays_same_kept_features_without_refitting():
    X_train, X_test, y_train, y_test = _synthetic_regression()
    mfe = MFOpenFE(task="regression", progress=False, max_candidates=30,
                   filter_threshold=0.3)
    X_train_fe = mfe.fit_transform(X_train, y_train)
    X_test_fe = mfe.transform(X_test)

    assert X_test_fe.shape[1] == X_train_fe.shape[1]
    assert X_test_fe.shape[0] == X_test.shape[0]


def test_transform_before_fit_raises():
    mfe = MFOpenFE(task="regression", progress=False)
    with pytest.raises(RuntimeError):
        mfe.transform(pd.DataFrame({"x1": [1.0, 2.0]}))
