"""Feature Transformation Graph (FTG) environment for the Stage-0 RL agent.

CAFEM (PAKDD 2020) frames per-feature engineering as navigating a graph: a node
is a candidate feature, an action applies a transformation operator, and the
reward is the wrapper model's performance change from the resulting feature.
This module is that environment for a *single* corpus dataset.

Design choices that keep RL search affordable (algorithm_plan Sec. 3 caps
Stage-0 cost by corpus size, not per-dataset exhaustiveness):

* One fixed train/eval split per dataset drives every reward estimate, so
  rewards are comparable within an episode and across episodes.
* Reward = *marginal* wrapper improvement from adding the transformed feature
  to the raw feature set (``score(base + f') - score(base)``), minus a small
  per-step penalty. This isolates the new feature's standalone usefulness,
  which is exactly the label Stage 1 must learn to predict.
* Evaluations are cached by feature content, and rows are subsampled, so
  revisiting a feature costs nothing and each fit is cheap.

State = the LFE QSA sketch of the *current* feature (``meta_features``), so the
transitions logged here are directly reusable as Stage-1 training rows.
"""

from __future__ import annotations

import dataclasses
import hashlib

import numpy as np
from sklearn.metrics import accuracy_score, r2_score
from sklearn.model_selection import train_test_split

from .corpus_data import CorpusDataset
from .meta_features import SKETCH_DIM, _target_association, feature_sketch
from .operators import N_OPERATORS, OPERATOR_NAMES, apply_operator

USEFUL_THRESHOLD = 0.0  # a transform is "useful" if delta strictly beats this
STEP_PENALTY = 1e-3     # discourages long chains that don't add value
_EPS = 1e-9


@dataclasses.dataclass
class Transition:
    """One recorded (state, action, outcome) tuple -- a Stage-1 training row."""

    feature_sketch: np.ndarray  # state: QSA sketch of the input feature
    operator: str
    action: int
    delta: float                # wrapper improvement over base
    reward: float
    useful: bool
    depth: int                  # how many transforms deep this feature is


class FTGEnvironment:
    """RL environment over one dataset's Feature Transformation Graph."""

    def __init__(
        self, dataset: CorpusDataset, max_depth: int = 3,
        eval_rows: int = 2000, max_features: int = 50, seed: int = 0,
    ):
        self.task = dataset.task
        self.max_depth = max_depth
        self.state_dim = SKETCH_DIM
        self.n_actions = N_OPERATORS
        self._rng = np.random.RandomState(seed)
        self._cache: dict[str, float] = {}
        self.transitions: list[Transition] = []

        X = dataset.X.to_numpy(dtype="float64")
        y = dataset.y
        # Subsample rows once so every fit in this dataset is cheap + comparable.
        if len(X) > eval_rows:
            idx = self._rng.choice(len(X), size=eval_rows, replace=False)
            X, y = X[idx], y[idx]
        # Cap the base feature set: every wrapper fit carries all base columns,
        # so a wide table (e.g. 216 cols) makes each of the hundreds of fits
        # per dataset expensive. Keep the columns most associated with the
        # target -- that keeps the base score meaningful (so a candidate's
        # *marginal* gain is a fair signal) while bounding cost by max_features,
        # not by the dataset's native width (algorithm_plan Sec. 3).
        if X.shape[1] > max_features:
            self.selected_columns = _top_feature_columns(X, y, self.task, max_features)
            X = X[:, self.selected_columns]
        else:
            self.selected_columns = np.arange(X.shape[1])
        strat = y if (self.task != "regression" and _has_min_class(y)) else None
        self._tr, self._te, self._ytr, self._yte = train_test_split(
            X, y, test_size=0.33, random_state=seed, stratify=strat)
        self._n_base = X.shape[1]
        self._base_score = self._fit_score(self._tr, self._te)

        # Current-episode state.
        self._start_col: np.ndarray | None = None
        self._cur_tr: np.ndarray | None = None
        self._cur_te: np.ndarray | None = None
        self._depth = 0

    # -- wrapper evaluator ---------------------------------------------------
    def _new_model(self):
        import lightgbm as lgb

        common = dict(n_estimators=60, num_leaves=15, min_child_samples=20,
                      verbose=-1, n_jobs=1)
        if self.task == "regression":
            return lgb.LGBMRegressor(**common)
        return lgb.LGBMClassifier(**common)

    def _fit_score(self, Xtr: np.ndarray, Xte: np.ndarray) -> float:
        model = self._new_model()
        if self.task == "regression":
            model.fit(Xtr, self._ytr)
            return float(r2_score(self._yte, model.predict(Xte)))
        model.fit(Xtr, self._ytr)
        return float(accuracy_score(self._yte, model.predict(Xte)))

    def _augmented_score(self, new_tr: np.ndarray, new_te: np.ndarray) -> float:
        """Wrapper score of the raw feature set plus one candidate column."""
        key = _col_hash(new_tr)
        if key in self._cache:
            return self._cache[key]
        Xtr = np.column_stack([self._tr, new_tr])
        Xte = np.column_stack([self._te, new_te])
        score = self._fit_score(Xtr, Xte)
        self._cache[key] = score
        return score

    # -- RL interface --------------------------------------------------------
    def reset(self, feature_index: int | None = None) -> np.ndarray:
        """Start a new episode from one raw feature; return its state sketch."""
        j = (self._rng.randint(self._n_base) if feature_index is None
             else feature_index)
        self._start_col = j
        self._cur_tr = self._tr[:, j].copy()
        self._cur_te = self._te[:, j].copy()
        self._depth = 0
        return self._state(self._cur_tr)

    def _state(self, col_tr: np.ndarray) -> np.ndarray:
        return feature_sketch(col_tr, self._ytr, self.task)

    def step(self, action: int) -> tuple[np.ndarray, float, bool, dict]:
        """Apply operator ``action`` to the current feature.

        Returns ``(next_state, reward, done, info)``. A degenerate result
        (constant/all-NaN feature) ends the episode with the step penalty.
        """
        op = OPERATOR_NAMES[action]
        state = self._state(self._cur_tr)
        new_tr = apply_operator(op, self._cur_tr)
        new_te = apply_operator(op, self._cur_te)

        if _degenerate(new_tr):
            reward = -STEP_PENALTY
            self.transitions.append(Transition(
                state, op, action, delta=0.0, reward=reward, useful=False,
                depth=self._depth))
            return state, reward, True, {"operator": op, "degenerate": True}

        # NaNs from an operator are median-filled so the wrapper can fit.
        new_tr_f = _fill(new_tr)
        new_te_f = _fill(new_te, ref=new_tr)
        delta = self._augmented_score(new_tr_f, new_te_f) - self._base_score
        reward = delta - STEP_PENALTY
        useful = delta > USEFUL_THRESHOLD
        self.transitions.append(Transition(
            state, op, action, delta=float(delta), reward=float(reward),
            useful=bool(useful), depth=self._depth))

        self._cur_tr, self._cur_te = new_tr_f, new_te_f
        self._depth += 1
        done = self._depth >= self.max_depth
        return self._state(self._cur_tr), float(reward), done, {"operator": op}


def _top_feature_columns(X: np.ndarray, y: np.ndarray, task: str, k: int) -> np.ndarray:
    """Indices of the ``k`` base columns most associated with the target.

    Uses the same scale-invariant association the QSA sketch does, so the
    selected base set is consistent with how features are represented
    elsewhere. Ties/degenerate columns fall to the back.
    """
    scores = np.array([_target_association(X[:, j], y, task)
                       for j in range(X.shape[1])])
    return np.argsort(scores)[::-1][:k]


def _has_min_class(y: np.ndarray) -> bool:
    _, counts = np.unique(y, return_counts=True)
    return counts.min() >= 2


def _degenerate(col: np.ndarray) -> bool:
    finite = col[np.isfinite(col)]
    return finite.size < 2 or np.nanstd(finite) < _EPS


def _fill(col: np.ndarray, ref: np.ndarray | None = None) -> np.ndarray:
    src = col if ref is None else ref
    med = np.nanmedian(src[np.isfinite(src)]) if np.isfinite(src).any() else 0.0
    out = np.where(np.isfinite(col), col, med)
    return out


def _col_hash(col: np.ndarray) -> str:
    return hashlib.sha1(np.round(col, 6).tobytes()).hexdigest()
