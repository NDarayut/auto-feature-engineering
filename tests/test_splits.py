"""Split-protocol guards: draft_plan Sec. 2 (fixed, reused split) and Sec. 5.3
(CV for small datasets, seed-repeat holdout otherwise) -- no network needed."""

import numpy as np

from afe.registry import BENCHMARK
from afe.splits import build_manifest, iter_split_indices, plan_for, protocol_for


def test_protocol_matches_scale():
    for spec in BENCHMARK:
        expected = "cv5" if spec.scale == "small" else "seed_repeat5"
        assert protocol_for(spec) == expected


def test_splits_manifest_covers_registry():
    manifest = build_manifest()
    assert set(manifest) == {s.key for s in BENCHMARK}


def _rng_y(n, n_classes, seed=0):
    rng = np.random.RandomState(seed)
    return rng.randint(0, n_classes, size=n)


def test_split_determinism():
    spec = next(s for s in BENCHMARK if s.task == "regression")
    n = 500
    first = list(iter_split_indices(spec, n))
    second = list(iter_split_indices(spec, n))
    assert len(first) == len(second)
    for (fid1, tr1, te1), (fid2, tr2, te2) in zip(first, second):
        assert fid1 == fid2
        assert np.array_equal(tr1, tr2)
        assert np.array_equal(te1, te2)


def test_no_train_test_overlap():
    spec = next(s for s in BENCHMARK if s.scale == "small")
    n = 300
    y = _rng_y(n, 2)
    for _fid, train_idx, test_idx in iter_split_indices(spec, n, y=y):
        assert set(train_idx).isdisjoint(test_idx)


def test_cv_folds_partition_all_rows():
    spec = next(s for s in BENCHMARK if s.scale == "small")
    n = 251  # not evenly divisible by 5, exercises remainder handling
    y = _rng_y(n, 2)
    test_sets = [set(test_idx) for _fid, _tr, test_idx in iter_split_indices(spec, n, y=y)]
    assert len(test_sets) == plan_for(spec).n_folds
    union = set().union(*test_sets)
    assert union == set(range(n))
    for i in range(len(test_sets)):
        for j in range(i + 1, len(test_sets)):
            assert test_sets[i].isdisjoint(test_sets[j])


def test_stratified_folds_preserve_class_balance():
    spec = next(s for s in BENCHMARK if s.scale == "small" and s.task != "regression")
    n = 1000
    rng = np.random.RandomState(1)
    y = (rng.rand(n) < 0.1).astype(int)  # imbalanced 10/90 split
    global_rate = y.mean()
    for _fid, train_idx, test_idx in iter_split_indices(spec, n, y=y):
        test_rate = y[test_idx].mean()
        assert abs(test_rate - global_rate) < 0.05


def test_seed_repeat_holdout_respects_test_size():
    spec = next(s for s in BENCHMARK if s.scale != "small")
    n = 1000
    y = _rng_y(n, 2) if spec.task != "regression" else None
    plan = plan_for(spec)
    for _fid, train_idx, test_idx in iter_split_indices(spec, n, y=y):
        assert abs(len(test_idx) / n - plan.test_size) < 0.02
