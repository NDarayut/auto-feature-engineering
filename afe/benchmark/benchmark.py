"""Run AutoFE methods against the frozen single train/test split and the
model panel, under a fixed compute budget, and write results to JSONL.

One row per (dataset, method, model_family): task metric, generation time,
feature counts, and a ``status`` that is one of "ok", "timeout", "error", or
"model_error" -- failure cases, timeouts and crashes are included in the
results rather than silently dropped.

Feature *generation* (the expensive, potentially-runaway step) runs in a
spawned subprocess with a hard wall-clock budget: if a
method doesn't finish in time, the subprocess is killed and the row is
recorded as a timeout rather than blocking the run indefinitely. Running
generation in its own process also means whatever memory
featuretools/autofeat/OpenFE use internally is released the moment that
subprocess exits, rather than accumulating in the main process.

Datasets are processed strictly sequentially, one at a time: for each
dataset, load it once, run every method against the same split, then
explicitly drop the dataset's in-memory objects and collect garbage before
moving to the next dataset. This keeps peak memory bounded by one dataset's
generation step at a time instead of scaling with a worker count.

On top of that, every method (not just OpenFE/CAFEM -- this is a harness-
level guard, not a per-method setting) is protected from unbounded memory
use on large/wide datasets by four generic mechanisms, applied identically
regardless of which method is running:

* a column pre-cap (``max_cols``): datasets wider than this are reduced to
  their most target-associated columns once per dataset, before any method
  sees the data;
* a row-sampled fit (``fit_sample_rows``): a method's ``fit_transform`` is
  called on a bounded random sample of the training fold rather than the
  whole thing, since combinatorial candidate search/scoring is where a
  method's memory use is worst;
* a row-chunked transform (``transform_chunk_rows``): the final expanded
  train/test matrices are built by calling a fitted method's ``transform``
  over row chunks and concatenating, instead of on the whole fold at once;
* a hard subprocess memory ceiling (``max_mem_gb``, via ``RLIMIT_AS``) as a
  safety net for whatever still overshoots, with fast (~1s) detection of a
  killed child instead of waiting out the full time budget.

    python -m scripts.run_benchmark --datasets nomao concrete-strength \\
        --methods baseline openfe --budget 300
"""

from __future__ import annotations

import gc
import json
import multiprocessing as mp
import time
from queue import Empty as QueueEmpty
from pathlib import Path
from typing import Iterable, Iterator

import numpy as np
import pandas as pd

from ..encoders import _split_columns
from ..methods import (METHODS, isolated_cwd, method_task, prep_for_generation,
                       quiet_method_warnings, resolve_method)
from .progress import ProgressReporter
from .report import build_report, load_rows
from .download import load
from .models import (MODEL_FAMILIES, feature_efficiency, fit_and_score,
                     prepare_family_input)
from .registry import BENCHMARK
from .splits import iter_split_indices, protocol_for

RESULTS_DIR = Path(__file__).resolve().parent.parent.parent / "results"
DEFAULT_BUDGET_SECONDS = 300
DEFAULT_MAX_COLS = 200
DEFAULT_FIT_SAMPLE_ROWS = 20_000
DEFAULT_TRANSFORM_CHUNK_ROWS = 20_000
DEFAULT_MAX_MEM_GB = 16.0
_POLL_INTERVAL_S = 1.0

_BY_KEY = {s.key: s for s in BENCHMARK}


def _cap_columns(X: pd.DataFrame, y: pd.Series, task: str, max_cols: int) -> pd.DataFrame:
    """Reduce ``X`` to its ``max_cols`` most target-associated columns.

    A generic, method-agnostic guard: any AutoFE method's internal
    candidate/interaction search can scale combinatorially in column count,
    so capping width once per dataset (applied identically ahead of every
    method, baseline included) bounds that risk without knowing anything
    about a particular method's algorithm.
    """
    if X.shape[1] <= max_cols:
        return X
    y_num = (np.asarray(y, dtype="float64") if task == "regression"
             else pd.factorize(pd.Series(y))[0].astype("float64"))
    corr = X.apply(lambda col: abs(np.corrcoef(col.to_numpy(dtype="float64"), y_num)[0, 1])
                   if col.nunique() > 1 else 0.0)
    keep = corr.sort_values(ascending=False).index[:max_cols]
    return X[keep]


def _sample_for_fit(X: pd.DataFrame, y: pd.Series, fit_sample_rows: int | None, seed: int = 0):
    """A bounded random row sample of the training fold, for the fit step only.

    Generic across methods: fit is where combinatorial search/scoring lives
    (candidate generation, RL episodes, DFS), so bounding its row count caps
    that cost regardless of which method is running. The full fold is still
    used for the (row-chunked) transform step below.
    """
    if not fit_sample_rows or len(X) <= fit_sample_rows:
        return X, y
    idx = np.random.RandomState(seed).choice(len(X), size=fit_sample_rows, replace=False)
    return X.iloc[idx], y.iloc[idx]


def _chunked_transform(method, X: pd.DataFrame, chunk_rows: int | None) -> pd.DataFrame:
    """Apply a fitted method's ``transform`` over row chunks and concatenate.

    Every adapter's ``transform`` already accepts an arbitrary-length frame
    and returns a same-length one, so chunking is a pure harness-side
    wrapper -- it bounds peak memory of applying the fitted transform to a
    large fold without any method-specific change.
    """
    if not chunk_rows or len(X) <= chunk_rows:
        return method.transform(X)
    parts = [method.transform(X.iloc[i:i + chunk_rows]) for i in range(0, len(X), chunk_rows)]
    return pd.concat(parts, axis=0)


def _generation_worker(method_name, X_train, y_train, X_test, task, queue,
                        fit_sample_rows, transform_chunk_rows, max_mem_gb) -> None:
    # RLIMIT_AS (virtual address space), not RLIMIT_DATA/RLIMIT_RSS: numpy/
    # lightgbm's large allocations go through mmap, which RLIMIT_DATA doesn't
    # cover, and RLIMIT_RSS has been advisory-only (unenforced) on Linux
    # since kernel 2.6. Set before any heavy work so a runaway allocation
    # raises MemoryError here, in this process, instead of the OS OOM-killer
    # picking an arbitrary victim on the host.
    if max_mem_gb:
        import resource

        limit_bytes = int(max_mem_gb * (1024 ** 3))
        try:
            resource.setrlimit(resource.RLIMIT_AS, (limit_bytes, limit_bytes))
        except (ValueError, OSError):
            pass  # a lower hard limit is already set (e.g. by a container) -- degrade to no cap
    try:
        method = resolve_method(method_name)()
        t0 = time.time()
        X_fit, y_fit = _sample_for_fit(X_train, y_train, fit_sample_rows)
        # isolated_cwd: methods may write scratch files to hardcoded relative
        # paths; method_task: methods only ever see classification/regression.
        with isolated_cwd(), quiet_method_warnings():
            method.fit_transform(X_fit, y_fit, method_task(task))
            t1 = time.time()
            X_train_gen = _chunked_transform(method, X_train, transform_chunk_rows)
            X_test_gen = _chunked_transform(method, X_test, transform_chunk_rows)
        t2 = time.time()
        # Compute-cost outputs: peak memory of the
        # generation subprocess (this process -- it did nothing else), and
        # inference-time cost (transform on unseen rows) separately from fit.
        import resource

        stats = {
            "fit_elapsed_s": t1 - t0,
            "transform_elapsed_s": t2 - t1,
            "peak_mem_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0,
            "n_candidates": getattr(method, "n_candidates_", None),
        }
        queue.put(("ok", X_train_gen, X_test_gen, stats))
    except MemoryError as exc:
        # Distinct from a generic bug: this is the RLIMIT_AS cap tripping.
        queue.put(("oom", str(exc) or f"MemoryError under {max_mem_gb} GB RLIMIT_AS cap",
                   None, None))
    except Exception as exc:  # noqa: BLE001 -- report to parent, don't crash the run
        queue.put(("error", str(exc), None, None))


def _run_method(method_name, X_train, y_train, X_test, task,
                 budget_seconds: float,
                 fit_sample_rows: int | None = DEFAULT_FIT_SAMPLE_ROWS,
                 transform_chunk_rows: int | None = DEFAULT_TRANSFORM_CHUNK_ROWS,
                 max_mem_gb: float | None = DEFAULT_MAX_MEM_GB) -> dict:
    # "spawn", not "fork": the parent process has already imported lightgbm/
    # openfe/featuretools/autofeat, which spin up native BLAS/numba worker
    # threads. Forking a multi-threaded process can deadlock the child on a
    # lock held by a thread that doesn't exist post-fork. Spawn starts a
    # clean interpreter instead, at the cost of slower startup.
    ctx = mp.get_context("spawn")
    queue = ctx.Queue()
    proc = ctx.Process(target=_generation_worker,
                        args=(method_name, X_train, y_train, X_test, task, queue,
                              fit_sample_rows, transform_chunk_rows, max_mem_gb))
    proc.start()
    # Poll instead of a single blocking queue.get(timeout=budget_seconds): a
    # crashed/OOM-killed child dies almost immediately, but one blocking get
    # can't distinguish "still working" from "gone" until the entire budget
    # elapses. Polling proc.is_alive() every _POLL_INTERVAL_S catches a dead
    # child within about a second instead.
    deadline = time.time() + budget_seconds
    status = a = b = stats = None
    got_result = False
    while time.time() < deadline:
        try:
            status, a, b, stats = queue.get(timeout=min(_POLL_INTERVAL_S, max(deadline - time.time(), 0)))
            got_result = True
            break
        except QueueEmpty:
            if not proc.is_alive():
                break  # died without putting a result

    if not got_result:
        if proc.is_alive():
            proc.terminate()
            proc.join()
            return {"status": "timeout", "elapsed_s": budget_seconds}
        proc.join()
        exitcode = proc.exitcode
        # A process killed by a signal reports exitcode = -signum on Linux;
        # the OS OOM-killer sends SIGKILL (-9). This only fires when our own
        # RLIMIT_AS cap didn't catch the allocation first (which instead
        # surfaces as status="oom" via the MemoryError branch below) -- e.g.
        # a C-extension allocation that bypasses Python's allocator.
        status = "oom" if exitcode == -9 else "crashed"
        return {"status": status, "elapsed_s": None}
    proc.join()
    if status in ("error", "oom"):
        return {"status": status, "error": a, "elapsed_s": None}
    return {"status": "ok", "X_train_gen": a, "X_test_gen": b,
            "elapsed_s": stats["fit_elapsed_s"] + stats["transform_elapsed_s"],
            **stats}


def run_dataset(
    key: str, methods: Iterable[str], model_families: Iterable[str] = MODEL_FAMILIES,
    budget_seconds: float = DEFAULT_BUDGET_SECONDS, use_cache: bool = True,
    max_cols: int | None = DEFAULT_MAX_COLS,
    fit_sample_rows: int | None = DEFAULT_FIT_SAMPLE_ROWS,
    transform_chunk_rows: int | None = DEFAULT_TRANSFORM_CHUNK_ROWS,
    max_mem_gb: float | None = DEFAULT_MAX_MEM_GB,
) -> Iterator[dict]:
    """Yield one result row per (method, model_family) for this dataset.

    Loads the dataset and computes its single frozen train/test split once,
    then reuses that same split for every method in ``methods`` -- the
    baseline and every AutoFE method are compared on identical data, so the
    only thing that differs between arms is which method generated the
    features. ``max_cols`` is applied here, once, so every
    method (including baseline) sees the identical column set.
    """
    spec = _BY_KEY[key]
    frame, meta = load(spec, use_cache=use_cache)
    target = meta["target"]
    y_full = frame[target]
    X_full = frame.drop(columns=[target])
    cat_cols, _ = _split_columns(X_full)
    protocol = protocol_for(spec)

    fold_id, train_idx, test_idx = next(iter_split_indices(
        spec, n=len(X_full),
        y=y_full.to_numpy() if spec.task != "regression" else None,
    ))
    X_train_raw, y_train = X_full.iloc[train_idx], y_full.iloc[train_idx]
    X_test_raw, y_test = X_full.iloc[test_idx], y_full.iloc[test_idx]
    X_train_num, X_test_num = prep_for_generation(X_train_raw, X_test_raw, cat_cols)
    if max_cols:
        X_train_num = _cap_columns(X_train_num, y_train, spec.task, max_cols)
        X_test_num = X_test_num[X_train_num.columns]

    try:
        for method_name in methods:
            gen = _run_method(method_name, X_train_num, y_train, X_test_num,
                               spec.task, budget_seconds,
                               fit_sample_rows, transform_chunk_rows, max_mem_gb)
            base = dict(key=key, method=method_name, fold_id=fold_id, protocol=protocol,
                        task=spec.task, status=gen["status"], gen_elapsed_s=gen.get("elapsed_s"),
                        fit_elapsed_s=gen.get("fit_elapsed_s"),
                        transform_elapsed_s=gen.get("transform_elapsed_s"),
                        peak_mem_mb=gen.get("peak_mem_mb"),
                        n_candidates=gen.get("n_candidates"))
            if gen["status"] != "ok":
                yield dict(base, model_family=None, metric=None, value=None,
                           error=gen.get("error"))
                continue

            X_train_gen, X_test_gen = gen["X_train_gen"], gen["X_test_gen"]
            n_generated = X_train_gen.shape[1] - X_train_num.shape[1]
            base["feature_efficiency"] = feature_efficiency(
                X_train_num, X_train_gen, y_train, spec.task)

            for family in model_families:
                Xtr_f, Xte_f = prepare_family_input(family, X_train_gen, X_test_gen)
                try:
                    score = fit_and_score(family, spec.task, Xtr_f, y_train, Xte_f, y_test)
                    yield dict(base, model_family=family, n_features_generated=n_generated,
                              n_features_final=X_train_gen.shape[1], **score)
                except Exception as exc:  # noqa: BLE001 -- record, keep the run going
                    yield dict(base, model_family=family, status="model_error",
                              error=str(exc), metric=None, value=None)
            del X_train_gen, X_test_gen
    finally:
        del frame, X_full, y_full, X_train_raw, X_test_raw, X_train_num, X_test_num
        gc.collect()


def _completed_pairs(out_path: Path) -> set[tuple[str, str]]:
    """(key, method) pairs already present in an existing results file."""
    if not out_path.exists():
        return set()
    pairs: set[tuple[str, str]] = set()
    with out_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            pairs.add((row["key"], row["method"]))
    return pairs


def run_benchmark(
    keys: Iterable[str], methods: Iterable[str], model_families: Iterable[str] = MODEL_FAMILIES,
    budget_seconds: float = DEFAULT_BUDGET_SECONDS,
    out_path: str | Path | None = None, use_cache: bool = True, resume: bool = True,
    max_cols: int | None = DEFAULT_MAX_COLS,
    fit_sample_rows: int | None = DEFAULT_FIT_SAMPLE_ROWS,
    transform_chunk_rows: int | None = DEFAULT_TRANSFORM_CHUNK_ROWS,
    max_mem_gb: float | None = DEFAULT_MAX_MEM_GB,
    progress: bool = True,
    report_path: str | Path | None = None,
) -> tuple[int, Path]:
    """Run every (dataset, method) pair sequentially, one dataset at a time.

    Datasets are processed one at a time -- never concurrently -- so peak
    memory is bounded by a single dataset's load + generation step, not
    multiplied by a worker count. Rows are appended and flushed to
    ``out_path`` immediately as they're produced. ``max_cols``/
    ``fit_sample_rows``/``transform_chunk_rows``/``max_mem_gb`` are the
    generic, method-agnostic memory guards described in the module
    docstring -- see ``run_dataset``/``_run_method``/``_generation_worker``.

    ``progress`` prints one line per completed (dataset, method) pair to
    stderr. ``report_path`` writes a markdown report there when the run
    finishes, built from every row in ``out_path`` -- including rows carried
    over from an earlier resumed run, so the report always covers the whole
    results file rather than just this invocation.
    """
    keys, methods = list(keys), list(methods)
    out = Path(out_path) if out_path else RESULTS_DIR / "benchmark_results.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    done = _completed_pairs(out) if resume else set()
    pending = [(k, m) for k in keys for m in methods if (k, m) not in done]
    reporter = ProgressReporter(total=len(pending), enabled=progress)
    reporter.start(methods, keys)

    n = 0
    with out.open("a") as f:
        for key in keys:
            pending_methods = [m for m in methods if (key, m) not in done]
            if not pending_methods:
                continue
            # One pair yields one row per model family; collect each pair's
            # rows so the progress line can show all family scores together.
            per_pair: dict[str, list[dict]] = {}
            for row in run_dataset(key, pending_methods, model_families,
                                   budget_seconds, use_cache,
                                   max_cols, fit_sample_rows, transform_chunk_rows,
                                   max_mem_gb):
                f.write(json.dumps(row, default=str) + "\n")
                f.flush()
                n += 1
                per_pair.setdefault(row["method"], []).append(row)
            for method in pending_methods:
                rows = per_pair.get(method)
                if not rows:
                    continue
                first = rows[0]
                reporter.pair_done(
                    key, method, first.get("status", "?"),
                    gen_elapsed_s=first.get("gen_elapsed_s"),
                    scores={r["model_family"]: r["value"] for r in rows
                            if r.get("model_family") and r.get("value") is not None},
                    error=first.get("error"))

    report_out = Path(report_path) if report_path else None
    if report_out:
        report_out.parent.mkdir(parents=True, exist_ok=True)
        report_out.write_text(build_report(load_rows(out)))
    reporter.finish(n, out_path=out, report_path=report_out)
    return n, out
