"""afe.benchmark guards: budget enforcement, failure handling, sequential
per-dataset execution, and JSONL output -- using the baseline method + a
synthetic in-memory frame (monkeypatch afe.benchmark.load), so no
network/cache or heavy AutoFE libraries are needed."""

import json
import time

import numpy as np
import pandas as pd
import pytest

import afe.benchmark.benchmark as benchmark
from afe.benchmark.registry import BENCHMARK

_KEY = next(s.key for s in BENCHMARK if s.scale == "small" and s.task != "regression")


def _synthetic_frame(n=80, seed=0):
    rng = np.random.RandomState(seed)
    frame = pd.DataFrame({
        "num_a": rng.randn(n),
        "cat_a": rng.choice(["a", "b", "c"], size=n),
        "target": rng.randint(0, 2, size=n),
    })
    return frame, {"target": "target"}


@pytest.fixture(autouse=True)
def _patch_load(monkeypatch):
    frame, meta = _synthetic_frame()
    monkeypatch.setattr(benchmark, "load", lambda spec, use_cache=True: (frame.copy(), meta))
    yield


def test_run_dataset_baseline_rows():
    rows = list(benchmark.run_dataset(_KEY, ["baseline"], budget_seconds=30))
    # 1 split * 3 model families
    assert len(rows) == 3
    assert all(r["status"] == "ok" for r in rows)
    assert all(r["method"] == "baseline" for r in rows)
    assert {r["model_family"] for r in rows} == {"tree", "linear", "knn"}
    assert all(r["n_features_generated"] == 0 for r in rows)


def test_run_dataset_reuses_split_across_methods():
    # baseline + baseline again (same method name is fine here -- just checks
    # both runs share the split's fold_id, i.e. one dataset load, one split,
    # reused across every method in the loop).
    rows = list(benchmark.run_dataset(_KEY, ["baseline", "baseline"], budget_seconds=30))
    assert len(rows) == 6
    assert {r["fold_id"] for r in rows} == {"split0"}


def test_run_dataset_respects_model_families_subset():
    rows = list(benchmark.run_dataset(_KEY, ["baseline"], model_families=["tree"],
                                      budget_seconds=30))
    assert len(rows) == 1
    assert rows[0]["model_family"] == "tree"


def _slow_method_worker(method_name, X_train, y_train, X_test, task, queue):
    time.sleep(5)
    queue.put(("ok", X_train, X_test, 5.0))


def test_budget_timeout_is_recorded(monkeypatch):
    monkeypatch.setattr(benchmark, "_generation_worker", _slow_method_worker)
    result = benchmark._run_method(
        "baseline", pd.DataFrame({"a": [1, 2]}), pd.Series([0, 1]),
        pd.DataFrame({"a": [1]}), "classification", budget_seconds=0.5)
    assert result["status"] == "timeout"


def test_unknown_method_raises_at_generation():
    rows = list(benchmark.run_dataset(_KEY, ["not-a-real-method"], budget_seconds=10))
    # METHODS[method_name] KeyError happens inside the worker -> reported, not crashed
    assert all(r["status"] == "error" for r in rows)


def test_run_benchmark_writes_valid_jsonl(tmp_path):
    out_path = tmp_path / "results.jsonl"
    n, path = benchmark.run_benchmark([_KEY], ["baseline"], budget_seconds=30,
                                      out_path=out_path)
    assert n == 3
    lines = path.read_text().strip().split("\n")
    assert len(lines) == 3
    for line in lines:
        row = json.loads(line)
        assert row["key"] == _KEY


def test_completed_pairs_skips_pairs_with_any_rows(tmp_path):
    out_path = tmp_path / "results.jsonl"
    out_path.write_text(
        json.dumps({"key": "a", "method": "baseline", "fold_id": "split0",
                   "model_family": "tree"}) + "\n" +
        json.dumps({"key": "a", "method": "openfe", "fold_id": "split0",
                   "model_family": None}) + "\n")
    done = benchmark._completed_pairs(out_path)
    assert done == {("a", "baseline"), ("a", "openfe")}
    assert benchmark._completed_pairs(tmp_path / "missing.jsonl") == set()


def test_run_benchmark_is_resumable(tmp_path):
    out_path = tmp_path / "results.jsonl"
    n1, _ = benchmark.run_benchmark([_KEY], ["baseline"], budget_seconds=30,
                                    out_path=out_path)
    assert n1 == 3
    n2, _ = benchmark.run_benchmark([_KEY], ["baseline"], budget_seconds=30,
                                    out_path=out_path)
    assert n2 == 0  # already-done pair skipped on resume
