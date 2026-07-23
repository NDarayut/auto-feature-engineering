"""MF-OpenFE's online path (algorithm_plan.md Stages 3-6) -- the public,
per-dataset AutoFE entrypoint. Everything here runs live, every time, on
whatever dataframe the caller passes in; nothing here trains anything (that's
Stage 0/1, ``stage0.py``/``stage1.py``, run once offline on the meta-training
corpus to produce the ``MetaModel`` this module loads).

Pipeline (each stage matches its ``algorithm_plan.md`` section):

* Stage 3 (generate)  -- ``openfe.get_candidate_features()``: OpenFE's own
  candidate operator library (arithmetic, groupby aggregations, log/sqrt/
  square/sigmoid/..., not our own simplified Stage-0 operator set).
* Stage 4 (meta-filter) -- score each candidate with the trained
  ``MetaModel``. **Coverage is partial and this is a deliberate, documented
  limitation, not a bug**: the meta-model was trained on labels generated
  from our own 8-operator library (``afe.meta.operators``), of which only
  ``log/sqrt/square/sigmoid`` (``COVERED_OPERATORS``) also appear as root
  operators in OpenFE's native candidates. Every other candidate (the
  majority -- arithmetic combinations, groupby aggregations, ``abs``,
  ``freq``, ``round``, ...) has no trained classifier and passes through
  this stage unfiltered rather than being silently dropped or wrongly
  scored. Closing this gap means regenerating Stage-0 labels from OpenFE's
  own candidate generator instead of our simplified operator list -- future
  work, not done here.
* Stage 5+6 (verify + select, merged) -- rather than reimplementing OpenFE's
  internal FeatureBoost residual-fitting + successive-halving machinery (deep
  undocumented internals), fit one LightGBM model on raw + surviving
  candidates and keep those whose feature importance clears a threshold --
  still an MDI-style attribution (Stage 6's own method), just in one fit.
* Stage 2 (gatekeeper) is not implemented: there is no offline-trained
  "will this help at all" model yet (it needs its own historical-outcome
  training data across the corpus, which doesn't exist). ``MFOpenFE`` always
  attempts generation.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ..encoders import _split_columns
from ..methods import prep_for_generation
from ..progress import ProgressReporter
from ..task import Task, infer_task as _infer_task
from .meta_features import feature_sketch
from .stage1 import DEFAULT_MODEL_PATH, MetaModel

# Root operators the trained meta-model has a classifier for AND that also
# appear as root operators among OpenFE's native candidates (verified against
# the installed openfe==0.0.12 package's get_candidate_features() output).
COVERED_OPERATORS = frozenset({"log", "sqrt", "square", "sigmoid"})


class MFOpenFE:
    """Meta-filtered OpenFE: generate candidates with OpenFE's own operator
    library, prune with a meta-model trained offline on ~100 historical
    datasets, verify + select by feature importance.

    ``fit_transform`` is only ever called on the training split; ``transform``
    replays the same kept candidate features on new data using state fit
    during ``fit_transform`` -- the same leak-safety contract as the
    benchmark's ``OpenFEMethod`` adapter.

    Parameters
    ----------
    task: "classification" | "regression", optional
        Inferred from ``y`` if not given.
    model_path: path to a trained ``MetaModel`` pickle (default:
        ``models/meta_model.pkl``, produced by ``scripts.train_meta_model``).
    order: candidate feature order passed to OpenFE's generator (default 1).
    filter_threshold: minimum meta-model usefulness score to survive Stage 4
        for a covered-operator candidate (default 0.5).
    max_candidates: cap on how many candidates proceed to verify/select,
        highest meta-filter score first (default 100).
    importance_threshold: minimum LightGBM feature importance to survive
        Stage 6 (default 0.0 -- drop only candidates used in zero splits).
    progress: show tqdm bars + stage markers while running (default True).
    """

    def __init__(
        self,
        task: Task | None = None,
        model_path: str | Path | None = None,
        order: int = 1,
        filter_threshold: float = 0.5,
        max_candidates: int = 100,
        importance_threshold: float = 0.0,
        n_jobs: int = 1,
        seed: int = 0,
        progress: bool = True,
    ):
        self.task = task
        self.model_path = Path(model_path) if model_path else DEFAULT_MODEL_PATH
        self.order = order
        self.filter_threshold = filter_threshold
        self.max_candidates = max_candidates
        self.importance_threshold = importance_threshold
        self.n_jobs = n_jobs
        self.seed = seed
        self.progress = progress

        self._task_resolved: Task | None = None
        self._categorical_cols: list[str] = []
        self._numeric_cols: list[str] = []
        self._X_train_raw: pd.DataFrame | None = None
        self._X_train_prepped: pd.DataFrame | None = None
        self._meta_model: MetaModel | None = None
        self._kept_nodes: list = []

        # Introspection, set after fit_transform().
        self.n_candidates_generated_: int = 0
        self.n_candidates_after_filter_: int = 0
        self.n_features_kept_: int = 0

    def fit_transform(self, X_train: pd.DataFrame, y_train) -> pd.DataFrame:
        progress = ProgressReporter(enabled=self.progress)
        y_train = pd.Series(y_train).reset_index(drop=True)
        X_train = X_train.reset_index(drop=True)
        self._task_resolved = self.task or _infer_task(y_train)
        self._X_train_raw = X_train

        progress.stage("Preparing data (encoding categoricals, imputing)...")
        self._categorical_cols, self._numeric_cols = _split_columns(X_train)
        X_prepped, _ = prep_for_generation(X_train, X_train, self._categorical_cols)
        self._X_train_prepped = X_prepped

        progress.stage(f"Loading meta-model ({self.model_path})...")
        self._meta_model = MetaModel.load(self.model_path)

        progress.stage("Generating candidate features (OpenFE operator library)...")
        candidates = self._generate(X_prepped)
        self.n_candidates_generated_ = len(candidates)
        progress.stage(f"Generated {len(candidates)} raw candidates")

        survivors = self._meta_filter(candidates, X_prepped, y_train, progress)
        self.n_candidates_after_filter_ = len(survivors)
        progress.stage(f"{len(survivors)} candidates survive the meta-filter "
                        f"(threshold={self.filter_threshold}, cap={self.max_candidates})")

        progress.stage(f"Verifying + selecting from {len(survivors)} candidates...")
        kept_nodes, X_train_fe = self._verify_and_select(survivors, X_prepped, y_train)
        self._kept_nodes = kept_nodes
        self.n_features_kept_ = len(kept_nodes)
        progress.done(f"Done: kept {len(kept_nodes)} engineered features "
                       f"(of {len(candidates)} generated)")
        return X_train_fe

    def transform(self, X_test: pd.DataFrame) -> pd.DataFrame:
        if self._X_train_prepped is None:
            raise RuntimeError("call fit_transform(...) before transform(...)")
        import openfe as ofe_pkg

        X_test = X_test.reset_index(drop=True)
        _, X_test_prepped = prep_for_generation(
            self._X_train_raw, X_test, self._categorical_cols)
        if not self._kept_nodes:
            return X_test_prepped
        _, X_test_fe = ofe_pkg.transform(
            self._X_train_prepped, X_test_prepped, self._kept_nodes, n_jobs=self.n_jobs)
        return X_test_fe

    # -- stages ---------------------------------------------------------
    def _generate(self, X_prepped: pd.DataFrame) -> list:
        """Stage 3: OpenFE's own candidate generator."""
        import openfe as ofe_pkg

        return ofe_pkg.get_candidate_features(
            numerical_features=self._numeric_cols,
            categorical_features=self._categorical_cols,
            order=self.order,
        )

    def _meta_filter(
        self, candidates: list, X_prepped: pd.DataFrame, y_train: pd.Series,
        progress: ProgressReporter,
    ) -> list:
        """Stage 4: score covered-operator candidates, pass the rest through."""
        import openfe as ofe_pkg

        covered = [n for n in candidates if n.name in self._meta_model.per_operator]
        uncovered = [n for n in candidates if n.name not in self._meta_model.per_operator]

        scored: list[tuple[float, object]] = []
        if covered:
            X_cov, _ = ofe_pkg.transform(X_prepped, X_prepped, covered, n_jobs=self.n_jobs)
            new_cols = X_cov.iloc[:, X_prepped.shape[1]:]
            y_arr = y_train.to_numpy()
            for i, node in enumerate(
                progress.iter(covered, desc="Meta-filtering", total=len(covered))
            ):
                values = new_cols.iloc[:, i].to_numpy(dtype="float64")
                sketch = feature_sketch(values, y_arr, self._task_resolved)
                score = self._meta_model.score(node.name, sketch)
                if score >= self.filter_threshold:
                    scored.append((score, node))
        scored.sort(key=lambda t: t[0], reverse=True)
        survivors = [node for _, node in scored] + uncovered
        return survivors[: self.max_candidates]

    def _verify_and_select(
        self, survivors: list, X_prepped: pd.DataFrame, y_train: pd.Series,
    ) -> tuple[list, pd.DataFrame]:
        """Stage 5+6 merged: fit once, keep candidates above an importance bar."""
        import openfe as ofe_pkg

        if not survivors:
            return [], X_prepped.copy()

        X_full, _ = ofe_pkg.transform(X_prepped, X_prepped, survivors, n_jobs=self.n_jobs)
        X_full = X_full.replace([np.inf, -np.inf], np.nan)

        model = self._fit_verifier(X_full, y_train)
        n_raw = X_prepped.shape[1]
        importances = model.feature_importances_[n_raw:]
        keep_mask = importances > self.importance_threshold

        kept_nodes = [node for node, keep in zip(survivors, keep_mask) if keep]
        kept_cols = list(X_full.columns[n_raw:][keep_mask])
        X_result = pd.concat(
            [X_prepped.reset_index(drop=True), X_full[kept_cols].reset_index(drop=True)],
            axis=1,
        )
        return kept_nodes, X_result

    def _fit_verifier(self, X_full: pd.DataFrame, y_train: pd.Series):
        import lightgbm as lgb

        common = dict(n_estimators=200, num_leaves=31, min_child_samples=20,
                      importance_type="gain", verbose=-1, n_jobs=self.n_jobs,
                      random_state=self.seed)
        if self._task_resolved == "regression":
            model = lgb.LGBMRegressor(**common)
            model.fit(X_full, y_train)
        else:
            y_codes = y_train.astype("category").cat.codes
            model = lgb.LGBMClassifier(**common)
            model.fit(X_full, y_codes)
        return model
