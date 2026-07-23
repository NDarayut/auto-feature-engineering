"""The 3-model panel (draft_plan Sec. 5.1): one boosted-tree model, one linear
model, one non-tree/non-linear (distance-based) model, per task type."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import mean_absolute_error, r2_score, roc_auc_score
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

MODEL_FAMILIES = ("tree", "linear", "knn")
# Which encoding profile (afe.encoders.ENCODER_PROFILES) each family needs.
ENCODING_FOR_FAMILY = {"tree": "tree", "linear": "linear", "knn": "linear"}


def _clean_for_tree(X):
    """Generated features can contain +/-inf (e.g. 1/0 from OpenFE/autofeat);
    LightGBM handles NaN natively but not inf."""
    if isinstance(X, pd.DataFrame):
        return X.replace([np.inf, -np.inf], np.nan)
    return np.where(np.isinf(X), np.nan, X)


def prepare_family_input(family: str, X_train_gen, X_test_gen):
    """Post-generation cleanup for a model family, fit on train only.

    ``X_train_gen``/``X_test_gen`` are the (already-numeric) matrices an
    AutoFE method produced. Generated features (ratios, logs, ...) can
    contain +/-inf or NaN; trees pass these through natively, but the
    linear/kNN family needs a finite, imputed, scaled input.
    """
    if family == "tree":
        return _clean_for_tree(X_train_gen), _clean_for_tree(X_test_gen)
    pipe = make_pipeline(SimpleImputer(strategy="median"), StandardScaler())
    Xtr = pipe.fit_transform(_clean_for_tree(X_train_gen))
    Xte = pipe.transform(_clean_for_tree(X_test_gen))
    return Xtr, Xte


def fit_and_score(family: str, task: str, X_train, y_train, X_test, y_test) -> dict:
    """Fit ``family``'s model on (X_train, y_train), score on (X_test, y_test).

    Returns {"metric": name, "value": float}. Classification uses AUC
    (binary or macro one-vs-rest); regression uses R2 (+ MAE as a secondary
    scale-aware metric).
    """
    if family == "tree":
        import lightgbm as lgb

        X_train, X_test = _clean_for_tree(X_train), _clean_for_tree(X_test)
        # n_jobs=1: kept conservative/deterministic rather than defaulting
        # to every core, since the feature-generation subprocess for the
        # same (dataset, method) may still be warming up native thread
        # pools around this call.
        if task == "regression":
            model = lgb.LGBMRegressor(n_estimators=200, verbose=-1, n_jobs=1)
        else:
            model = lgb.LGBMClassifier(n_estimators=200, verbose=-1, n_jobs=1)
    elif family == "linear":
        model = Ridge() if task == "regression" else LogisticRegression(
            max_iter=1000)
    elif family == "knn":
        model = (KNeighborsRegressor(n_neighbors=10) if task == "regression"
                 else KNeighborsClassifier(n_neighbors=10))
    else:
        raise ValueError(f"unknown model family: {family!r}")

    if task == "regression":
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        return {"metric": "r2", "value": float(r2_score(y_test, pred)),
                "mae": float(mean_absolute_error(y_test, pred))}

    y_train_c = pd.Series(y_train).astype("category").cat.codes
    y_test_c = pd.Series(y_test).astype("category").cat.codes
    model.fit(X_train, y_train_c)
    if y_train_c.nunique() <= 2:
        p = model.predict_proba(X_test)[:, 1]
        return {"metric": "auc", "value": float(roc_auc_score(y_test_c, p))}
    p = model.predict_proba(X_test)
    return {"metric": "auc_ovr", "value": float(
        roc_auc_score(y_test_c, p, multi_class="ovr"))}
