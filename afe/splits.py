"""Freeze the evaluation split *protocol* for every benchmark dataset.

draft_plan.md Sec. 2 requires that, for each dataset, the train/test split (or
CV protocol) is "fixed and recorded ... decided once and reused across all
methods." Sec. 5.3 picks the protocol by dataset size: 5-fold CV for datasets
small enough for it to be cheap, 5-seed-repeat holdout otherwise.

We freeze the *recipe* (protocol name + seed + params), not materialized row
indices: regenerating indices from a fixed seed is a cheap, pure function, and
keeps ``afe/manifests/`` metadata-sized like ``benchmark.json``/``corpus.json``
rather than growing with dataset size. Shuffling uses ``numpy.random.RandomState``
(NumPy's legacy RNG, whose bit-stream is stability-guaranteed across versions)
rather than sklearn's splitters, so "same seed" really means "same split"
regardless of which library versions a given machine has installed.

Run ``python -m afe.splits`` to (re)generate ``afe/manifests/splits.json``.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from pathlib import Path
from typing import Iterator, Literal

import numpy as np
import pandas as pd

from .registry import BENCHMARK, DatasetSpec

CV_FOLDS = 5
N_SEED_REPEATS = 5
HOLDOUT_TEST_SIZE = 0.2
SPLIT_SEED_BASE = 20260101

MANIFEST_DIR = Path(__file__).resolve().parent / "manifests"
SPLITS_MANIFEST = MANIFEST_DIR / "splits.json"

Protocol = Literal["cv5", "seed_repeat5"]


@dataclasses.dataclass(frozen=True)
class SplitPlan:
    key: str
    protocol: Protocol
    seed: int
    n_folds: int | None  # set for cv5, else None
    n_repeats: int | None  # set for seed_repeat5, else None
    test_size: float | None  # set for seed_repeat5, else None
    stratify: bool


def _seed_for(key: str) -> int:
    """Stable per-dataset seed derived from the base constant + the key.

    Using a hash (rather than e.g. list index) means adding/reordering
    datasets in the registry never perturbs an existing dataset's seed.
    """
    digest = hashlib.sha256(key.encode()).hexdigest()
    return (SPLIT_SEED_BASE + int(digest[:8], 16)) % (2**32 - 1)


def protocol_for(spec: DatasetSpec) -> Protocol:
    return "cv5" if spec.scale == "small" else "seed_repeat5"


def plan_for(spec: DatasetSpec) -> SplitPlan:
    protocol = protocol_for(spec)
    stratify = spec.task != "regression"
    if protocol == "cv5":
        return SplitPlan(spec.key, protocol, _seed_for(spec.key),
                          n_folds=CV_FOLDS, n_repeats=None, test_size=None,
                          stratify=stratify)
    return SplitPlan(spec.key, protocol, _seed_for(spec.key),
                      n_folds=None, n_repeats=N_SEED_REPEATS,
                      test_size=HOLDOUT_TEST_SIZE, stratify=stratify)


def build_manifest() -> dict[str, dict]:
    return {spec.key: dataclasses.asdict(plan_for(spec)) for spec in BENCHMARK}


def write_manifest() -> None:
    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
    SPLITS_MANIFEST.write_text(json.dumps(build_manifest(), indent=2))
    print(f"wrote {SPLITS_MANIFEST}")


def load_manifest() -> dict[str, SplitPlan]:
    """Read the frozen manifest; fall back to an in-memory build if absent.

    The fallback means tests and callers never need ``python -m afe.splits``
    to have been run first -- same offline-safety pattern as
    ``afe.manifests.build_corpus_manifest``'s per-suite stub fallback.
    """
    if SPLITS_MANIFEST.exists():
        raw = json.loads(SPLITS_MANIFEST.read_text())
        return {k: SplitPlan(**v) for k, v in raw.items()}
    return {spec.key: plan_for(spec) for spec in BENCHMARK}


# --------------------------------------------------------------------------- #
# Deterministic index generation
# --------------------------------------------------------------------------- #
def _fold_ids(n: int, k: int, seed: int) -> np.ndarray:
    """Unstratified fold assignment: shuffle row order, cut into k chunks."""
    rng = np.random.RandomState(seed)
    order = rng.permutation(n)
    fold_of_row = np.empty(n, dtype=int)
    fold_of_row[order] = np.arange(n) % k
    return fold_of_row


def _stratified_fold_ids(y: np.ndarray, k: int, seed: int) -> np.ndarray:
    """Per-class shuffle + round-robin fold cut, then merged back by row."""
    rng = np.random.RandomState(seed)
    fold_of_row = np.empty(len(y), dtype=int)
    for cls in np.unique(y):
        idx = np.flatnonzero(y == cls)
        order = rng.permutation(idx)
        fold_of_row[order] = np.arange(len(order)) % k
    return fold_of_row


def _holdout_test_mask(n: int, test_size: float, seed: int) -> np.ndarray:
    rng = np.random.RandomState(seed)
    order = rng.permutation(n)
    n_test = int(round(n * test_size))
    mask = np.zeros(n, dtype=bool)
    mask[order[:n_test]] = True
    return mask


def _stratified_holdout_test_mask(y: np.ndarray, test_size: float,
                                   seed: int) -> np.ndarray:
    rng = np.random.RandomState(seed)
    mask = np.zeros(len(y), dtype=bool)
    for cls in np.unique(y):
        idx = np.flatnonzero(y == cls)
        order = rng.permutation(idx)
        n_test = int(round(len(idx) * test_size))
        mask[order[:n_test]] = True
    return mask


def iter_split_indices(
    spec: DatasetSpec, n: int, y: pd.Series | np.ndarray | None = None,
) -> Iterator[tuple[str, np.ndarray, np.ndarray]]:
    """Yield ``(fold_id, train_idx, test_idx)`` for ``spec``'s frozen plan.

    ``n`` is the row count of the dataset being split. ``y`` is required (and
    used for class-stratified splitting) whenever ``spec.task != "regression"``.
    """
    plan = load_manifest().get(spec.key) or plan_for(spec)
    if plan.stratify and y is None:
        raise ValueError(f"{spec.key}: stratified split requires y")
    y_arr = np.asarray(y) if y is not None else None
    all_idx = np.arange(n)

    if plan.protocol == "cv5":
        fold_of_row = (_stratified_fold_ids(y_arr, plan.n_folds, plan.seed)
                       if plan.stratify
                       else _fold_ids(n, plan.n_folds, plan.seed))
        for f in range(plan.n_folds):
            test_idx = all_idx[fold_of_row == f]
            train_idx = all_idx[fold_of_row != f]
            yield f"cv{f}", train_idx, test_idx
        return

    for r in range(plan.n_repeats):
        seed_r = plan.seed + r
        test_mask = (_stratified_holdout_test_mask(y_arr, plan.test_size, seed_r)
                    if plan.stratify
                    else _holdout_test_mask(n, plan.test_size, seed_r))
        yield f"seed{r}", all_idx[~test_mask], all_idx[test_mask]


if __name__ == "__main__":
    write_manifest()
