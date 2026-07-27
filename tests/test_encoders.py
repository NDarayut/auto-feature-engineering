"""Encoder guards: per-model-family profiles, fit on train
fold only) -- no network needed, synthetic frames only."""

import numpy as np
import pandas as pd
import pytest

from afe.encoders import LinearEncoder, TreeEncoder, _split_columns


def _toy_frame(n=40, seed=0):
    rng = np.random.RandomState(seed)
    return pd.DataFrame({
        "num_a": rng.randn(n),
        "num_b": rng.randn(n) * 10 + 5,
        "cat_a": rng.choice(["red", "green", "blue"], size=n),
        "cat_b": rng.choice(["x", "y"], size=n),
    })


def test_split_columns_by_dtype():
    X = _toy_frame()
    cat, num = _split_columns(X)
    assert set(cat) == {"cat_a", "cat_b"}
    assert set(num) == {"num_a", "num_b"}


def test_tree_encoder_unseen_category_in_test_does_not_crash():
    X_train = _toy_frame(seed=1)
    X_test = pd.DataFrame({
        "num_a": [0.1], "num_b": [1.0], "cat_a": ["purple"], "cat_b": ["x"],
    })
    enc = TreeEncoder().fit(X_train)
    out = enc.transform(X_test)
    assert out["cat_a"].iloc[0] == -1  # reserved unknown code


def test_tree_encoder_state_derives_only_from_train():
    X_train = _toy_frame(seed=2)
    enc = TreeEncoder().fit(X_train)
    seen_categories = set(enc._encoder.categories_[0])
    assert seen_categories <= set(X_train["cat_a"].unique())


def test_tree_encoder_missing_values_pass_through():
    X = _toy_frame(seed=3)
    X.loc[0, "num_a"] = np.nan
    enc = TreeEncoder().fit(X)
    out = enc.transform(X)
    assert pd.isna(out.loc[0, "num_a"])


def test_linear_encoder_train_output_is_standardized():
    X = _toy_frame(seed=4, n=200)
    enc = LinearEncoder()
    out = enc.fit_transform(X)
    n_num = len(enc._num_cols)
    numeric_block = out[:, :n_num]
    assert np.allclose(numeric_block.mean(axis=0), 0, atol=1e-6)
    assert np.allclose(numeric_block.std(axis=0), 1, atol=1e-6)


def test_linear_encoder_imputes_missing_values():
    X = _toy_frame(seed=5)
    X.loc[0, "num_a"] = np.nan
    X.loc[1, "cat_a"] = None
    enc = LinearEncoder()
    out = enc.fit_transform(X)
    assert not np.isnan(out).any()


def test_target_encoding_requires_y():
    X = _toy_frame(seed=6)
    with pytest.raises(ValueError):
        TreeEncoder(categorical_encoding="target").fit(X)
    with pytest.raises(ValueError):
        LinearEncoder(categorical_encoding="target").fit(X)


def test_tree_target_encoding_fits_and_transforms():
    X = _toy_frame(seed=7, n=100)
    y = pd.Series(np.random.RandomState(7).randint(0, 2, size=100))
    enc = TreeEncoder(categorical_encoding="target").fit(X, y)
    out = enc.transform(X)
    assert out["cat_a"].dtype.kind == "f"  # target-encoded to a float rate


def test_categorical_columns_metadata_matches_dtypes():
    X = _toy_frame()
    enc = TreeEncoder().fit(X)
    assert set(enc.categorical_columns) == {"cat_a", "cat_b"}
