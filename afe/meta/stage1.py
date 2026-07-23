"""Stage 1 -- train the supervised meta-model (algorithm_plan Sec. 2, Stage 1).

Consumes Stage 0's pooled ``(QSA sketch, operator, useful)`` tuples and trains
the artifact used online: a **per-operator** classifier (LFE, IJCAI 2017) that,
given a feature's QSA sketch, predicts whether applying that operator will yield
a useful new feature. One small RandomForest per operator, bundled into a single
``MetaModel`` -- small, fast, inspectable, and the only meta-learning artifact
the online path (Stage 4 filter) ever loads.

Evaluation is grouped by dataset (``did``): we report metrics with datasets held
out, because the real online use is "a dataset the meta-model never saw," so an
in-dataset split would be optimistic. Operators with too few / single-class
labels fall back to their global base rate rather than a fitted model.
"""

from __future__ import annotations

import json
import pickle
import time
from pathlib import Path

import numpy as np

from .meta_features import SKETCH_DIM
from .operators import OPERATOR_NAMES
from .stage0 import DEFAULT_TUPLES_PATH

MODEL_DIR = Path(__file__).resolve().parent.parent.parent / "models"
DEFAULT_MODEL_PATH = MODEL_DIR / "meta_model.pkl"
MIN_PER_OPERATOR = 20  # below this, use the base rate instead of a fitted model


class _BaseRate:
    """Degenerate 'model' for operators with too little/one-class data."""

    def __init__(self, p: float):
        self.p = float(p)

    def predict_proba_useful(self, X: np.ndarray) -> np.ndarray:
        return np.full(len(X), self.p, dtype="float64")


class _RFModel:
    def __init__(self, clf):
        self.clf = clf

    def predict_proba_useful(self, X: np.ndarray) -> np.ndarray:
        proba = self.clf.predict_proba(X)
        # index of the positive (useful=True) class
        classes = list(self.clf.classes_)
        if True in classes:
            return proba[:, classes.index(True)]
        return np.zeros(len(X), dtype="float64")


class MetaModel:
    """Bundle of per-operator usefulness predictors (the Stage-1 artifact)."""

    def __init__(self, per_operator: dict, sketch_dim: int):
        self.per_operator = per_operator
        self.sketch_dim = sketch_dim

    def score(self, operator: str, sketch: np.ndarray) -> float:
        """Predicted usefulness probability for one (operator, feature) pair."""
        model = self.per_operator.get(operator)
        if model is None:
            return 0.0
        x = np.asarray(sketch, dtype="float64").reshape(1, -1)
        return float(model.predict_proba_useful(x)[0])

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as fh:
            pickle.dump(self, fh)
        return path

    @staticmethod
    def load(path: str | Path) -> "MetaModel":
        with Path(path).open("rb") as fh:
            return pickle.load(fh)


def load_tuples(path: str | Path) -> list[dict]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"no Stage-0 tuples at {path}; run stage0 first")
    rows = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def _to_arrays(rows: list[dict]):
    X = np.array([r["sketch"] for r in rows], dtype="float64")
    y = np.array([bool(r["useful"]) for r in rows], dtype=bool)
    ops = np.array([r["operator"] for r in rows])
    dids = np.array([int(r["did"]) for r in rows])
    return X, y, ops, dids


def _fit_operator(X: np.ndarray, y: np.ndarray, seed: int):
    """Fit one operator's predictor, or fall back to its base rate."""
    if len(y) < MIN_PER_OPERATOR or y.sum() == 0 or y.sum() == len(y):
        return _BaseRate(float(y.mean()) if len(y) else 0.0)
    from sklearn.ensemble import RandomForestClassifier

    clf = RandomForestClassifier(
        n_estimators=200, max_depth=8, min_samples_leaf=5,
        class_weight="balanced", random_state=seed, n_jobs=1)
    clf.fit(X, y)
    return _RFModel(clf)


def _grouped_holdout_metrics(rows: list[dict], seed: int) -> dict:
    """Leave-datasets-out AUC/accuracy, pooled across operators."""
    from sklearn.metrics import accuracy_score, roc_auc_score

    X, y, ops, dids = _to_arrays(rows)
    unique_dids = np.unique(dids)
    if len(unique_dids) < 2:
        return {"note": "need >=2 datasets for grouped holdout; skipped"}

    rng = np.random.RandomState(seed)
    perm = rng.permutation(unique_dids)
    n_test = max(1, len(perm) // 5)
    test_dids = set(perm[:n_test].tolist())
    test_mask = np.array([d in test_dids for d in dids])
    if test_mask.all() or (~test_mask).all():
        return {"note": "degenerate dataset split; skipped"}

    preds = np.zeros(test_mask.sum(), dtype="float64")
    for op in OPERATOR_NAMES:
        tr = (~test_mask) & (ops == op)
        te = test_mask & (ops == op)
        if te.sum() == 0:
            continue
        model = _fit_operator(X[tr], y[tr], seed)
        preds[_local_index(test_mask, te)] = model.predict_proba_useful(X[te])

    y_test = y[test_mask]
    out = {"n_train": int((~test_mask).sum()), "n_test": int(test_mask.sum()),
           "test_datasets": sorted(test_dids),
           "base_rate_useful": float(y.mean())}
    if len(np.unique(y_test)) == 2:
        out["holdout_auc"] = float(roc_auc_score(y_test, preds))
    out["holdout_accuracy"] = float(accuracy_score(y_test, preds >= 0.5))
    return out


def _local_index(test_mask: np.ndarray, sub: np.ndarray) -> np.ndarray:
    """Positions of ``sub`` within the compacted test-only array."""
    test_positions = np.where(test_mask)[0]
    sub_positions = np.where(sub)[0]
    return np.searchsorted(test_positions, sub_positions)


def train_meta_model(
    tuples_path: str | Path | None = None, out_path: str | Path | None = None,
    seed: int = 0, verbose: bool = True,
) -> tuple[MetaModel, dict]:
    """Train + persist the per-operator meta-model; return ``(model, report)``."""
    tuples_path = Path(tuples_path) if tuples_path else DEFAULT_TUPLES_PATH
    out_path = Path(out_path) if out_path else DEFAULT_MODEL_PATH
    t0 = time.time()

    rows = load_tuples(tuples_path)
    X, y, ops, dids = _to_arrays(rows)
    if X.shape[1] != SKETCH_DIM:
        raise ValueError(
            f"tuple sketch dim {X.shape[1]} != current SKETCH_DIM {SKETCH_DIM}; "
            "regenerate Stage-0 tuples after changing meta_features")

    metrics = _grouped_holdout_metrics(rows, seed)

    # Final model: fit each operator on ALL tuples (holdout was for reporting).
    per_operator: dict = {}
    per_op_counts: dict = {}
    for op in OPERATOR_NAMES:
        m = ops == op
        per_operator[op] = _fit_operator(X[m], y[m], seed)
        per_op_counts[op] = {"n": int(m.sum()), "useful": int(y[m].sum()),
                             "fitted": bool(m.sum() >= MIN_PER_OPERATOR
                                            and 0 < y[m].sum() < m.sum())}

    model = MetaModel(per_operator, sketch_dim=SKETCH_DIM)
    saved = model.save(out_path)

    report = {
        "n_tuples": len(rows),
        "n_datasets": int(len(np.unique(dids))),
        "sketch_dim": SKETCH_DIM,
        "per_operator": per_op_counts,
        "holdout": metrics,
        "model_path": str(saved),
        "train_seconds": round(time.time() - t0, 2),
    }
    if verbose:
        print(json.dumps(report, indent=2))
    return model, report
