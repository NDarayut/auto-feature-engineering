"""Run AutoFE methods against the frozen single train/test split and the
model panel, under a fixed compute budget, and write results to JSONL.

One row per (dataset, method, model_family): task metric, generation time,
feature counts, and a ``status`` that is one of "ok", "timeout", "error", or
"model_error" -- failed/timed-out runs are recorded, not silently dropped
(draft_plan Sec. 6: "failure cases/timeouts/crashes included rather than
excluded").

Feature *generation* (the expensive, potentially-runaway step) runs in a
spawned subprocess with a hard wall-clock budget (draft_plan Sec. 3): if a
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

from ..encoders import _split_columns
from ..methods import METHODS, prep_for_generation
from .download import load
from .models import MODEL_FAMILIES, fit_and_score, prepare_family_input
from .registry import BENCHMARK
from .splits import iter_split_indices, protocol_for

RESULTS_DIR = Path(__file__).resolve().parent.parent.parent / "results"
DEFAULT_BUDGET_SECONDS = 300

_BY_KEY = {s.key: s for s in BENCHMARK}


def _generation_worker(method_name, X_train, y_train, X_test, task, queue) -> None:
    try:
        method = METHODS[method_name]()
        t0 = time.time()
        X_train_gen = method.fit_transform(X_train, y_train, task)
        X_test_gen = method.transform(X_test)
        queue.put(("ok", X_train_gen, X_test_gen, time.time() - t0))
    except Exception as exc:  # noqa: BLE001 -- report to parent, don't crash the run
        queue.put(("error", str(exc), None, None))


def _run_method(method_name, X_train, y_train, X_test, task,
                 budget_seconds: float) -> dict:
    # "spawn", not "fork": the parent process has already imported lightgbm/
    # openfe/featuretools/autofeat, which spin up native BLAS/numba worker
    # threads. Forking a multi-threaded process can deadlock the child on a
    # lock held by a thread that doesn't exist post-fork. Spawn starts a
    # clean interpreter instead, at the cost of slower startup.
    ctx = mp.get_context("spawn")
    queue = ctx.Queue()
    proc = ctx.Process(target=_generation_worker,
                        args=(method_name, X_train, y_train, X_test, task, queue))
    proc.start()
    # Read from the queue *before* joining: a result large enough to exceed
    # the OS pipe buffer makes the child block on Queue.put() until someone
    # reads, so join()-then-get() can deadlock until the timeout kills it.
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


def run_dataset(
    key: str, methods: Iterable[str], model_families: Iterable[str] = MODEL_FAMILIES,
    budget_seconds: float = DEFAULT_BUDGET_SECONDS, use_cache: bool = True,
) -> Iterator[dict]:
    """Yield one result row per (method, model_family) for this dataset.

    Loads the dataset and computes its single frozen train/test split once,
    then reuses that same split for every method in ``methods`` -- the
    baseline and every AutoFE method are compared on identical data, per
    draft_plan Sec. 1 ("only thing that should differ ... is which method
    generated the features").
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

    try:
        for method_name in methods:
            gen = _run_method(method_name, X_train_num, y_train, X_test_num,
                               spec.task, budget_seconds)
            base = dict(key=key, method=method_name, fold_id=fold_id, protocol=protocol,
                        task=spec.task, status=gen["status"], gen_elapsed_s=gen.get("elapsed_s"))
            if gen["status"] != "ok":
                yield dict(base, model_family=None, metric=None, value=None,
                           error=gen.get("error"))
                continue

            X_train_gen, X_test_gen = gen["X_train_gen"], gen["X_test_gen"]
            n_generated = X_train_gen.shape[1] - X_train_num.shape[1]

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
) -> tuple[int, Path]:
    """Run every (dataset, method) pair sequentially, one dataset at a time.

    Datasets are processed one at a time -- never concurrently -- so peak
    memory is bounded by a single dataset's load + generation step, not
    multiplied by a worker count. Rows are appended and flushed to
    ``out_path`` immediately as they're produced.
    """
    out = Path(out_path) if out_path else RESULTS_DIR / "benchmark_results.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    done = _completed_pairs(out) if resume else set()
    n = 0
    with out.open("a") as f:
        for key in keys:
            pending_methods = [m for m in methods if (key, m) not in done]
            if not pending_methods:
                continue
            for row in run_dataset(key, pending_methods, model_families,
                                   budget_seconds, use_cache):
                f.write(json.dumps(row, default=str) + "\n")
                f.flush()
                n += 1
    return n, out
