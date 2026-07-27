"""``compare()`` -- benchmark any AutoFE algorithm, plug and play.

Standalone and algorithm-agnostic: this module has no built-in notion of
"baseline"/"openfe"/"mf-openfe" or any other named method. A caller passes
whichever methods they want compared -- their own, ours, or a mix -- as a
plain list. Every entry is either:

* an object/class implementing ``afe.methods.AutoFEMethod``
  (``fit_transform(X_train, y_train, task)`` / ``transform(X_test)``), or
* a plain function ``(X_train, y_train, X_test, task) -> (X_train_new,
  X_test_new)`` -- no class required.

Datasets are either built-in benchmark keys (``afe.benchmark.registry``) or
a caller's own ``(X, y)`` data, in the same call. Each dataset gets a single
fixed-seed 80/20 train/test split (the same protocol and helpers the rest of
``afe.benchmark`` uses -- see ``splits.py``), reused identically across every
method being compared, so differences reflect the method, not the split.

Runs **in-process by default** (works no matter where a method is defined --
script, notebook, REPL). Pass ``budget_seconds=`` to opt into the same
spawn-subprocess + wall-clock-budget isolation ``afe.benchmark.benchmark``
uses; that path requires the method to be picklable (defined in a real
importable module), the accepted tradeoff for the stronger isolation.
"""

from __future__ import annotations

import dataclasses
import gc
import json
import multiprocessing as mp
import time
from pathlib import Path
from queue import Empty as QueueEmpty
from typing import Callable, Iterator, Sequence, Union

import numpy as np
import pandas as pd

from ..encoders import _split_columns
from ..methods import AutoFEMethod, prep_for_generation
from ..methods import isolated_cwd, method_task, quiet_method_warnings
from .progress import ProgressReporter
from .report import build_report
from ..task import infer_task
from .benchmark import _completed_pairs
from .download import load
from .models import (MODEL_FAMILIES, feature_efficiency, fit_and_score,
                     prepare_family_input)
from .registry import BENCHMARK
from .splits import HOLDOUT_TEST_SIZE, _holdout_test_mask, _seed_for, _stratified_holdout_test_mask

MethodLike = Union[AutoFEMethod, type, Callable]
MethodSpec = Union[MethodLike, "tuple[str, MethodLike]"]

_BY_KEY = {s.key: s for s in BENCHMARK}


# --------------------------------------------------------------------------- #
# Method resolution + dispatch
# --------------------------------------------------------------------------- #
def _method_name(spec: MethodSpec, index: int) -> tuple[str, MethodLike]:
    """Resolve a ``methods=`` entry to ``(display_name, method)``.

    ``.name`` (a class attribute or instance attribute -- matches
    ``afe.methods``' adapter convention, e.g. ``BaselineMethod.name =
    "baseline"``) wins over ``__name__`` for both classes and instances, so a
    class's declared ``name`` isn't shadowed by its Python class name.
    """
    if isinstance(spec, tuple) and len(spec) == 2 and isinstance(spec[0], str):
        return spec
    method = spec
    name = getattr(method, "name", None) or getattr(method, "__name__", None)
    if name is None:
        name = f"method_{index}"
    return name, method


def _generate_features(method: MethodLike, X_train, y_train, X_test, task):
    """Dispatch one method to ``(X_train_gen, X_test_gen)``, either calling
    style (class/instance with fit_transform+transform, or a plain function)."""
    obj = method() if isinstance(method, type) else method
    # isolated_cwd: methods may write scratch files to hardcoded relative
    # paths; method_task: methods only ever see classification/regression.
    task = method_task(task)
    if hasattr(obj, "fit_transform") and hasattr(obj, "transform"):
        with isolated_cwd(), quiet_method_warnings():
            X_train_gen = obj.fit_transform(X_train, y_train, task)
            X_test_gen = obj.transform(X_test)
        return X_train_gen, X_test_gen
    if callable(obj):
        with isolated_cwd(), quiet_method_warnings():
            result = obj(X_train, y_train, X_test, task)
        if not (isinstance(result, tuple) and len(result) == 2):
            raise TypeError(
                "a plain-function method must return (X_train_new, X_test_new); "
                f"got {type(result)!r} from {obj!r}")
        return result
    raise TypeError(
        f"{obj!r} must be callable, or implement fit_transform(X_train, y_train, "
        "task) / transform(X_test)")


def _generation_worker(method, X_train, y_train, X_test, task, queue) -> None:
    try:
        t0 = time.time()
        X_train_gen, X_test_gen = _generate_features(method, X_train, y_train, X_test, task)
        queue.put(("ok", X_train_gen, X_test_gen, time.time() - t0))
    except Exception as exc:  # noqa: BLE001 -- report to parent, don't crash the run
        queue.put(("error", str(exc), None, None))


def _run_with_budget(method, X_train, y_train, X_test, task, budget_seconds: float) -> dict:
    """Subprocess-isolated, wall-clock-budgeted generation (opt-in).

    Same spawn+Queue-before-join pattern as ``afe.benchmark.benchmark``'s
    ``_run_method`` -- see that module's docstring for why spawn (not fork)
    and why the queue is read before ``proc.join()``. The method object
    itself travels through ``args=``, so it must be picklable.
    """
    ctx = mp.get_context("spawn")
    queue = ctx.Queue()
    proc = ctx.Process(target=_generation_worker,
                        args=(method, X_train, y_train, X_test, task, queue))
    proc.start()
    try:
        status, a, b, elapsed = queue.get(timeout=budget_seconds)
    except QueueEmpty:
        if proc.is_alive():
            proc.terminate()
            proc.join()
            return {"status": "timeout", "elapsed_s": budget_seconds}
        return {"status": "crashed", "elapsed_s": None}
    proc.join()
    if status == "error":
        return {"status": "error", "error": a, "elapsed_s": None}
    return {"status": "ok", "X_train_gen": a, "X_test_gen": b, "elapsed_s": elapsed}


def _run_inprocess(method, X_train, y_train, X_test, task) -> dict:
    t0 = time.time()
    try:
        X_train_gen, X_test_gen = _generate_features(method, X_train, y_train, X_test, task)
        return {"status": "ok", "X_train_gen": X_train_gen, "X_test_gen": X_test_gen,
                "elapsed_s": time.time() - t0}
    except Exception as exc:  # noqa: BLE001 -- record, keep the run going
        return {"status": "error", "error": str(exc), "elapsed_s": None}


# --------------------------------------------------------------------------- #
# Dataset loading + splitting (built-in registry keys or user-supplied data)
# --------------------------------------------------------------------------- #
def _load_builtin(key: str, use_cache: bool = True):
    spec = _BY_KEY[key]
    frame, meta = load(spec, use_cache=use_cache)
    target = meta["target"]
    y_full = frame[target]
    X_full = frame.drop(columns=[target])
    cat_cols, _ = _split_columns(X_full)
    return X_full, y_full, spec.task, cat_cols


def _prepare_custom(X: pd.DataFrame, y, task_override: str | None):
    X_full = X.reset_index(drop=True)
    y_full = pd.Series(y).reset_index(drop=True)
    task = task_override or infer_task(y_full)
    cat_cols, _ = _split_columns(X_full)
    return X_full, y_full, task, cat_cols


def _split(dataset_name: str, X_full: pd.DataFrame, y_full: pd.Series, task: str):
    seed = _seed_for(dataset_name)
    if task != "regression":
        mask = _stratified_holdout_test_mask(y_full.to_numpy(), HOLDOUT_TEST_SIZE, seed)
    else:
        mask = _holdout_test_mask(len(X_full), HOLDOUT_TEST_SIZE, seed)
    train_idx, test_idx = np.flatnonzero(~mask), np.flatnonzero(mask)
    return (X_full.iloc[train_idx], y_full.iloc[train_idx],
            X_full.iloc[test_idx], y_full.iloc[test_idx])


# --------------------------------------------------------------------------- #
# Per-dataset run
# --------------------------------------------------------------------------- #
def _run_dataset(
    dataset_name: str, X_full: pd.DataFrame, y_full: pd.Series, task: str,
    cat_cols: list[str], methods: list[tuple[str, MethodLike]],
    model_families: Sequence[str], budget_seconds: float | None,
) -> Iterator[dict]:
    X_train_raw, y_train, X_test_raw, y_test = _split(dataset_name, X_full, y_full, task)
    X_train_num, X_test_num = prep_for_generation(X_train_raw, X_test_raw, cat_cols)
    try:
        for name, method in methods:
            gen = (_run_inprocess(method, X_train_num, y_train, X_test_num, task)
                   if budget_seconds is None
                   else _run_with_budget(method, X_train_num, y_train, X_test_num,
                                        task, budget_seconds))
            base = dict(key=dataset_name, method=name, task=task, status=gen["status"],
                        gen_elapsed_s=gen.get("elapsed_s"))
            if gen["status"] != "ok":
                yield dict(base, model_family=None, metric=None, value=None,
                           error=gen.get("error"))
                continue

            X_train_gen, X_test_gen = gen["X_train_gen"], gen["X_test_gen"]
            n_generated = X_train_gen.shape[1] - X_train_num.shape[1]
            base["feature_efficiency"] = feature_efficiency(
                X_train_num, X_train_gen, y_train, task)
            for family in model_families:
                Xtr_f, Xte_f = prepare_family_input(family, X_train_gen, X_test_gen)
                try:
                    score = fit_and_score(family, task, Xtr_f, y_train, Xte_f, y_test)
                    yield dict(base, model_family=family, n_features_generated=n_generated,
                              n_features_final=X_train_gen.shape[1], **score)
                except Exception as exc:  # noqa: BLE001 -- record, keep the run going
                    yield dict(base, model_family=family, status="model_error",
                              error=str(exc), metric=None, value=None)
            del X_train_gen, X_test_gen
    finally:
        del X_train_raw, X_test_raw, X_train_num, X_test_num
        gc.collect()


def _load_all_rows(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


# --------------------------------------------------------------------------- #
# Result
# --------------------------------------------------------------------------- #
@dataclasses.dataclass
class CompareResult:
    """``compare()``'s return value: raw rows, a DataFrame, or a printable table."""

    rows: list[dict]

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.rows)

    def summary(self) -> str:
        """A plain per-model-family score table (dataset x method) + a mean
        row per method -- no method is treated as a privileged baseline."""
        ok = [r for r in self.rows if r.get("status") == "ok"]
        if not ok:
            return "No successful (dataset, method, model_family) results."

        lines: list[str] = []
        for family in sorted({r["model_family"] for r in ok}):
            sub = [r for r in ok if r["model_family"] == family]
            datasets_ = sorted({r["key"] for r in sub})
            methods_ = sorted({r["method"] for r in sub})
            value_by = {(r["key"], r["method"]): r["value"] for r in sub}

            lines.append(f"## {family}")
            lines.append("| dataset | " + " | ".join(methods_) + " |")
            lines.append("|---|" + "|".join(["---"] * len(methods_)) + "|")
            for ds in datasets_:
                cells = [f"{value_by[(ds, m)]:.3f}" if (ds, m) in value_by else "—"
                        for m in methods_]
                lines.append(f"| {ds} | " + " | ".join(cells) + " |")
            means = []
            for m in methods_:
                vals = [value_by[(ds, m)] for ds in datasets_ if (ds, m) in value_by]
                means.append(f"{sum(vals) / len(vals):.3f}" if vals else "—")
            lines.append("| **mean** | " + " | ".join(means) + " |")
            lines.append("")

        failed = [r for r in self.rows if r.get("status") != "ok"]
        if failed:
            counts: dict[tuple[str, str, str], int] = {}
            for r in failed:
                k = (r["key"], r["method"], r["status"])
                counts[k] = counts.get(k, 0) + 1
            lines.append("## Failures")
            for (ds, m, status), n in sorted(counts.items()):
                lines.append(f"- {ds} / {m}: {status} x{n}")
        return "\n".join(lines)

    def __repr__(self) -> str:
        return self.summary()


# --------------------------------------------------------------------------- #
# Public entrypoint
# --------------------------------------------------------------------------- #
def compare(
    methods: Sequence[MethodSpec],
    datasets: Sequence[str] | None = None,
    custom_datasets: dict[str, tuple[pd.DataFrame, "pd.Series | np.ndarray"]] | None = None,
    custom_tasks: dict[str, str] | None = None,
    model_families: Sequence[str] = MODEL_FAMILIES,
    budget_seconds: float | None = None,
    out_path: str | Path | None = None,
    resume: bool = True,
    use_cache: bool = True,
    progress: bool = True,
    report_path: str | Path | None = None,
) -> CompareResult:
    """Benchmark ``methods`` against each other on ``datasets``/``custom_datasets``.

    ``methods``: each entry is an ``AutoFEMethod``-style object/class, a plain
    ``(X_train, y_train, X_test, task) -> (X_train_new, X_test_new)``
    function, or a ``(display_name, method)`` tuple to control its label.

    ``datasets``: built-in benchmark keys (see ``afe.benchmark.registry.BENCHMARK``).
    ``custom_datasets``: ``{name: (X, y)}`` -- your own data, task inferred
    unless given in ``custom_tasks``. At least one of ``datasets``/
    ``custom_datasets`` is required.

    ``budget_seconds=None`` (default) runs every method in-process, no
    timeout -- always works. Set it to opt into subprocess isolation + a hard
    wall-clock budget per (dataset, method) pair, which requires the method
    to be importable/picklable (won't work for a class/function defined in a
    notebook or REPL).

    ``out_path``, if given, persists rows as JSONL and makes the run
    resumable (skips ``(dataset, method)`` pairs already present, same
    contract as ``afe.benchmark.run_benchmark``).

    ``progress`` prints one line per completed (dataset, method) pair to
    stderr -- stdout stays clean for the result table. ``report_path``
    writes a markdown report there when the run finishes.
    """
    if not datasets and not custom_datasets:
        raise ValueError("pass datasets=[...] and/or custom_datasets={...}")

    methods_resolved = [_method_name(m, i) for i, m in enumerate(methods)]

    out = Path(out_path) if out_path else None
    done = _completed_pairs(out) if (out and resume) else set()
    if out:
        out.parent.mkdir(parents=True, exist_ok=True)

    dataset_specs: list[tuple[str, pd.DataFrame, pd.Series, str, list[str]]] = []
    for key in (datasets or []):
        dataset_specs.append((key, *_load_builtin(key, use_cache=use_cache)))
    for name, (X, y) in (custom_datasets or {}).items():
        task_override = (custom_tasks or {}).get(name)
        dataset_specs.append((name, *_prepare_custom(X, y, task_override)))

    n_pending = sum(1 for name, _, _, _, _ in dataset_specs
                    for m, _ in methods_resolved if (name, m) not in done)
    reporter = ProgressReporter(total=n_pending, enabled=progress)
    reporter.start([m for m, _ in methods_resolved],
                   [name for name, *_ in dataset_specs])

    rows: list[dict] = []
    fh = out.open("a") if out else None
    try:
        for dataset_name, X_full, y_full, task, cat_cols in dataset_specs:
            pending = [(n, m) for n, m in methods_resolved if (dataset_name, n) not in done]
            if not pending:
                continue
            # One pair yields one row per model family; group them so the
            # progress line can show every family's score on one line.
            per_pair: dict[str, list[dict]] = {}
            for row in _run_dataset(dataset_name, X_full, y_full, task, cat_cols,
                                    pending, model_families, budget_seconds):
                rows.append(row)
                per_pair.setdefault(row["method"], []).append(row)
                if fh:
                    fh.write(json.dumps(row, default=str) + "\n")
                    fh.flush()
            for name, _ in pending:
                pair_rows = per_pair.get(name)
                if not pair_rows:
                    continue
                first = pair_rows[0]
                reporter.pair_done(
                    dataset_name, name, first.get("status", "?"),
                    gen_elapsed_s=first.get("gen_elapsed_s"),
                    scores={r["model_family"]: r["value"] for r in pair_rows
                            if r.get("model_family") and r.get("value") is not None},
                    error=first.get("error"))
    finally:
        if fh:
            fh.close()

    if out and out.exists():
        rows = _load_all_rows(out)

    report_out = Path(report_path) if report_path else None
    if report_out:
        report_out.parent.mkdir(parents=True, exist_ok=True)
        report_out.write_text(build_report(rows))
    reporter.finish(len(rows), out_path=out, report_path=report_out,
                    hint_when_unsaved=True)
    return CompareResult(rows=rows)
