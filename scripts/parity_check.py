"""Raw-feature LightGBM baseline on a few benchmark datasets.

Validates the load/split path end-to-end before any AutoFE: numbers should land
near OpenFE Table-3 "Base" territory (regression R^2 high on California Housing,
classification AUC well above 0.5 on Nomao). This is a sanity gate, not a full
benchmark run (single split, default params).

    python -m scripts.parity_check
"""

from __future__ import annotations

import warnings

from sklearn.metrics import r2_score, roc_auc_score
from sklearn.model_selection import train_test_split

from afe.download import load
from afe.registry import BENCHMARK

warnings.filterwarnings("ignore")
CHECK = ["california-housing", "nomao", "concrete-strength"]


def _prep(frame, target):
    y = frame[target]
    X = frame.drop(columns=[target])
    # LightGBM wants numeric/categorical dtypes; cheap encode of object cols.
    for c in X.select_dtypes(include=["object", "category", "bool"]).columns:
        X[c] = X[c].astype("category").cat.codes
    return X, y


def main() -> int:
    import lightgbm as lgb

    specs = {s.key: s for s in BENCHMARK}
    for key in CHECK:
        spec = specs[key]
        frame, meta = load(spec)
        X, y = _prep(frame, meta["target"])
        reg = spec.task == "regression"
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=0)
        if reg:
            model = lgb.LGBMRegressor(n_estimators=200, verbose=-1)
            model.fit(Xtr, ytr)
            score = r2_score(yte, model.predict(Xte))
            print(f"{key:22s} regression  R2 = {score:.3f}")
        else:
            yc = y.astype("category").cat.codes
            ytr = yc.loc[ytr.index]
            yte = yc.loc[yte.index]
            model = lgb.LGBMClassifier(n_estimators=200, verbose=-1)
            model.fit(Xtr, ytr)
            if yc.nunique() == 2:
                p = model.predict_proba(Xte)[:, 1]
                score = roc_auc_score(yte, p)
                print(f"{key:22s} binary-clf  AUC = {score:.3f}")
            else:
                p = model.predict_proba(Xte)
                score = roc_auc_score(yte, p, multi_class="ovr")
                print(f"{key:22s} multiclass  AUC = {score:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
