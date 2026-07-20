"""Raw-feature LightGBM baseline on a few benchmark datasets.

Validates the full split -> encode -> fold-iterate -> train -> score path end
to end before any AutoFE: numbers should land near OpenFE Table-3 "Base"
territory (regression R^2 high on California Housing, classification AUC well
above 0.5 on Nomao). This is a sanity gate, not a full benchmark run (default
hyperparameters, no compute-budget enforcement, no model panel) -- but it now
exercises the real per-dataset split protocol (draft_plan Sec. 5.3) and
reports mean +/- std across folds/seeds rather than a single split's score.

    python -m scripts.parity_check
"""

from __future__ import annotations

import warnings

import numpy as np
from sklearn.metrics import r2_score, roc_auc_score

from afe.eval_data import iter_folds
from afe.registry import BENCHMARK

warnings.filterwarnings("ignore")
CHECK = ["california-housing", "nomao", "concrete-strength"]


def _score_fold(spec, fold) -> float:
    import lightgbm as lgb

    if spec.task == "regression":
        model = lgb.LGBMRegressor(n_estimators=200, verbose=-1)
        model.fit(fold.X_train, fold.y_train)
        return r2_score(fold.y_test, model.predict(fold.X_test))

    y_train = fold.y_train.astype("category").cat.codes
    y_test = fold.y_test.astype("category").cat.codes
    model = lgb.LGBMClassifier(n_estimators=200, verbose=-1)
    model.fit(fold.X_train, y_train)
    if y_train.nunique() == 2:
        p = model.predict_proba(fold.X_test)[:, 1]
        return roc_auc_score(y_test, p)
    p = model.predict_proba(fold.X_test)
    return roc_auc_score(y_test, p, multi_class="ovr")


def main() -> int:
    specs = {s.key: s for s in BENCHMARK}
    for key in CHECK:
        spec = specs[key]
        scores = [_score_fold(spec, fold) for fold in iter_folds(key, encoding="tree")]
        metric = "R2" if spec.task == "regression" else "AUC"
        print(f"{key:22s} {spec.task:14s} {metric} = "
              f"{np.mean(scores):.3f} +/- {np.std(scores):.3f}  (n={len(scores)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
