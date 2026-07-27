"""afe.benchmark.compare: the plug-and-play, algorithm-agnostic benchmark API.

Synthetic in-memory data throughout (no network). Covers: plain-function
methods, class-based methods, a method that raises, custom vs. built-in
datasets, out_path resumability, and the opt-in subprocess+budget path.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import afe.benchmark._compare as compare_mod
from afe.benchmark import BaselineMethod, CompareResult, compare


def _synthetic_regression(n: int = 200, seed: int = 0):
    rng = np.random.RandomState(seed)
    x1 = rng.uniform(1, 10, n)
    x2 = rng.uniform(-3, 3, n)
    y = 3 * np.log(x1) + 2 * (x2 ** 2) + rng.randn(n) * 0.1
    return pd.DataFrame({"x1": x1, "x2": x2}), pd.Series(y)


def _noop_function(X_train, y_train, X_test, task):
    return X_train, X_test


class NoOpMethod:
    """Module-level class -- picklable, usable with budget_seconds=."""

    name = "noop_class"

    def fit_transform(self, X_train, y_train, task):
        return X_train

    def transform(self, X_test):
        return X_test


class RaisingMethod:
    name = "raiser"

    def fit_transform(self, X_train, y_train, task):
        raise RuntimeError("boom")

    def transform(self, X_test):
        return X_test


def test_requires_at_least_one_dataset_source():
    with pytest.raises(ValueError):
        compare(methods=[NoOpMethod])


def test_plain_function_and_class_method_both_work():
    X, y = _synthetic_regression()
    result = compare(methods=[_noop_function, NoOpMethod],
                     custom_datasets={"synthetic": (X, y)},
                     model_families=["tree"])
    frame = result.to_frame()
    assert set(frame["method"]) == {"_noop_function", "noop_class"}
    assert (frame["status"] == "ok").all()
    # both are no-ops on the same data/split -> identical scores
    assert frame.set_index("method")["value"].nunique() == 1


def test_method_tuple_overrides_display_name():
    X, y = _synthetic_regression()
    result = compare(methods=[("my-custom-name", _noop_function)],
                     custom_datasets={"synthetic": (X, y)}, model_families=["tree"])
    assert result.to_frame()["method"].iloc[0] == "my-custom-name"


def test_raising_method_recorded_as_error_others_still_run():
    X, y = _synthetic_regression()
    result = compare(methods=[NoOpMethod, RaisingMethod],
                     custom_datasets={"synthetic": (X, y)}, model_families=["tree"])
    frame = result.to_frame()
    ok_rows = frame[frame["method"] == "noop_class"]
    err_rows = frame[frame["method"] == "raiser"]
    assert (ok_rows["status"] == "ok").all()
    assert (err_rows["status"] == "error").all()
    assert "Failures" in result.summary()


def test_summary_has_no_privileged_baseline_column():
    X, y = _synthetic_regression()
    result = compare(methods=[NoOpMethod], custom_datasets={"synthetic": (X, y)},
                     model_families=["tree"])
    assert "baseline" not in result.summary().lower()


def test_builtin_dataset_path(monkeypatch):
    frame = pd.DataFrame({
        "num_a": np.random.RandomState(0).randn(80),
        "target": np.random.RandomState(0).randint(0, 2, size=80),
    })
    monkeypatch.setattr(compare_mod, "load",
                       lambda spec, use_cache=True: (frame.copy(), {"target": "target"}))
    key = compare_mod.BENCHMARK[0].key
    result = compare(methods=[NoOpMethod], datasets=[key], model_families=["tree"])
    assert result.to_frame()["key"].iloc[0] == key


def test_out_path_is_resumable(tmp_path):
    X, y = _synthetic_regression()
    out = tmp_path / "results.jsonl"
    r1 = compare(methods=[NoOpMethod], custom_datasets={"synthetic": (X, y)},
                out_path=out, model_families=["tree"])
    n_lines_1 = len(out.read_text().splitlines())
    r2 = compare(methods=[NoOpMethod], custom_datasets={"synthetic": (X, y)},
                out_path=out, model_families=["tree"])
    n_lines_2 = len(out.read_text().splitlines())
    assert n_lines_1 == n_lines_2  # second call did no new work
    assert len(r1.rows) == len(r2.rows)


def test_budget_seconds_subprocess_path_with_module_level_class():
    X, y = _synthetic_regression()
    result = compare(methods=[NoOpMethod], custom_datasets={"synthetic": (X, y)},
                     model_families=["tree"], budget_seconds=30)
    assert (result.to_frame()["status"] == "ok").all()


def test_compare_result_repr_is_summary():
    X, y = _synthetic_regression()
    result = compare(methods=[NoOpMethod], custom_datasets={"synthetic": (X, y)},
                     model_families=["tree"])
    assert repr(result) == result.summary()
    assert isinstance(result, CompareResult)


def test_compare_writes_report_by_default(tmp_path, monkeypatch):
    """A report is produced without asking, like the CLI."""
    monkeypatch.setattr(compare_mod, "RESULTS_DIR", tmp_path)
    X = pd.DataFrame({"a": [0.0, 1.0, 2.0, 3.0] * 8, "b": [1.0, 0.0] * 16})
    y = pd.Series([0, 1] * 16)
    compare(methods=[BaselineMethod], custom_datasets={"d": (X, y)},
            model_families=["tree"], progress=False)
    default = tmp_path / "compare_report.md"
    assert default.exists(), "compare() should write a report by default"
    assert default.read_text().startswith("# Benchmark Report")


def test_compare_report_can_be_disabled(tmp_path, monkeypatch):
    monkeypatch.setattr(compare_mod, "RESULTS_DIR", tmp_path)
    X = pd.DataFrame({"a": [0.0, 1.0, 2.0, 3.0] * 8, "b": [1.0, 0.0] * 16})
    y = pd.Series([0, 1] * 16)
    compare(methods=[BaselineMethod], custom_datasets={"d": (X, y)},
            model_families=["tree"], report_path=None, progress=False)
    assert list(tmp_path.glob("*.md")) == [], "report_path=None must skip the report"


def test_compare_writes_report_when_asked(tmp_path):
    X = pd.DataFrame({"a": [0.0, 1.0, 2.0, 3.0] * 8, "b": [1.0, 0.0] * 16})
    y = pd.Series([0, 1] * 16)
    report = tmp_path / "r.md"
    compare(methods=[BaselineMethod], custom_datasets={"d": (X, y)},
            model_families=["tree"], report_path=report, progress=False)
    assert report.read_text().startswith("# Benchmark Report")
