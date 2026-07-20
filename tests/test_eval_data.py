"""iter_folds() end-to-end guards, using a synthetic in-memory frame
(monkeypatched afe.download.load) so no network/cache is needed."""

import numpy as np
import pandas as pd
import pytest

import afe.eval_data as eval_data
from afe.registry import BENCHMARK

_KEY = next(s.key for s in BENCHMARK if s.scale == "small" and s.task != "regression")


def _synthetic_frame(n=200, seed=0):
    rng = np.random.RandomState(seed)
    frame = pd.DataFrame({
        "num_a": rng.randn(n),
        "cat_a": rng.choice(["a", "b", "c"], size=n),
        "target": rng.randint(0, 2, size=n),
    })
    meta = {"target": "target"}
    return frame, meta


@pytest.fixture(autouse=True)
def _patch_load(monkeypatch):
    frame, meta = _synthetic_frame()

    def fake_load(spec, use_cache=True):
        return frame.copy(), meta

    monkeypatch.setattr(eval_data, "load", fake_load)
    yield


def test_iter_folds_shapes_and_disjointness():
    for fold in eval_data.iter_folds(_KEY, encoding="tree"):
        n_train = len(fold.y_train)
        n_test = len(fold.y_test)
        assert n_train + n_test == 200
        assert set(fold.y_train.index).isdisjoint(fold.y_test.index)


def test_iter_folds_encoder_never_sees_test_rows(monkeypatch):
    seen_fit_sizes = []
    from afe.encoders import TreeEncoder

    orig_fit = TreeEncoder.fit

    def spy_fit(self, X, y=None):
        seen_fit_sizes.append(len(X))
        return orig_fit(self, X, y)

    monkeypatch.setattr(TreeEncoder, "fit", spy_fit)
    folds = list(eval_data.iter_folds(_KEY, encoding="tree"))
    for fold, fit_size in zip(folds, seen_fit_sizes):
        assert fit_size == len(fold.y_train)
        assert fit_size < 200  # never the full frame


def test_iter_folds_deterministic_across_calls():
    first = [(f.fold_id, f.y_train.index.tolist(), f.y_test.index.tolist())
             for f in eval_data.iter_folds(_KEY, encoding="tree")]
    second = [(f.fold_id, f.y_train.index.tolist(), f.y_test.index.tolist())
              for f in eval_data.iter_folds(_KEY, encoding="tree")]
    assert first == second


def test_iter_folds_linear_encoding_produces_ndarray():
    fold = next(iter(eval_data.iter_folds(_KEY, encoding="linear")))
    assert isinstance(fold.X_train, np.ndarray)


def test_iter_folds_unknown_key_raises():
    with pytest.raises(KeyError):
        list(eval_data.iter_folds("not-a-real-dataset"))


def test_list_available_keys_matches_registry():
    assert set(eval_data.list_available_keys()) == {s.key for s in BENCHMARK}
