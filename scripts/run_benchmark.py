"""CLI entry point for the benchmark harness (afe.benchmark.run_benchmark).

Runs (dataset, method) pairs sequentially -- one dataset at a time, freeing
its memory before moving to the next -- and is resumable (skips pairs
already present in --out). Datasets are ordered smallest-scale-first by
default.

Only the no-op ``baseline`` method ships with the harness; every other method
is your own, referenced by import path (``mypkg.methods:MyMethod``) and
resolved at run time.

Examples:
    python -m scripts.run_benchmark --datasets nomao concrete-strength \\
        --methods baseline mypkg.methods:MyMethod --budget 300

    # everything, against two of your own methods
    python -m scripts.run_benchmark \\
        --methods baseline mypkg:MethodA mypkg:MethodB --budget 300

    # only the tree model, to skip linear/knn scoring
    python -m scripts.run_benchmark --methods baseline mypkg:MyMethod --models tree
"""

from __future__ import annotations

import argparse
import warnings

from afe.benchmark import (DEFAULT_BUDGET_SECONDS, DEFAULT_FIT_SAMPLE_ROWS,
                           DEFAULT_MAX_COLS, DEFAULT_MAX_MEM_GB,
                           DEFAULT_TRANSFORM_CHUNK_ROWS, run_benchmark)
from afe.benchmark.models import MODEL_FAMILIES
from afe.benchmark.registry import BENCHMARK
from afe.methods import METHODS

warnings.filterwarnings("ignore")

_SCALE_ORDER = {"small": 0, "medium": 1, "large": 2}


def _default_dataset_order() -> list[str]:
    # Smallest datasets first: cheap methods finish fast and surface bugs
    # early, before the run reaches the multi-hundred-thousand-row datasets.
    return [s.key for s in sorted(BENCHMARK, key=lambda s: _SCALE_ORDER[s.scale])]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="*", default=None,
                        help="benchmark keys to run (default: all 22, small-first)")
    parser.add_argument("--methods", nargs="*", default=["baseline"],
                        metavar="METHOD",
                        help="methods to benchmark: the built-in %s, and/or an "
                             "import path to your own, e.g. "
                             "'mypkg.methods:MyMethod'" % sorted(METHODS))
    parser.add_argument("--models", nargs="*", default=list(MODEL_FAMILIES),
                        choices=list(MODEL_FAMILIES),
                        help="which model families to evaluate generated features with")
    parser.add_argument("--budget", type=float, default=DEFAULT_BUDGET_SECONDS,
                        help="per (dataset, method) generation time budget, seconds")
    parser.add_argument("--out", default=None, help="output JSONL path")
    parser.add_argument("--no-resume", action="store_true",
                        help="re-run pairs even if already present in --out")
    parser.add_argument("--max-cols", type=int, default=DEFAULT_MAX_COLS,
                        help="cap a dataset to its N most target-associated columns "
                             "before any method runs, generically bounding any method's "
                             "combinatorial blowup in column count (0 disables)")
    parser.add_argument("--fit-sample-rows", type=int, default=DEFAULT_FIT_SAMPLE_ROWS,
                        help="cap the row count a method's fit step sees to a random "
                             "sample of this size (0 disables)")
    parser.add_argument("--transform-chunk-rows", type=int, default=DEFAULT_TRANSFORM_CHUNK_ROWS,
                        help="apply a fitted method's transform in row chunks of this "
                             "size instead of on the whole fold at once (0 disables)")
    parser.add_argument("--max-mem-gb", type=float, default=DEFAULT_MAX_MEM_GB,
                        help="hard RLIMIT_AS memory cap (GB) for each method's generation "
                             "subprocess, as a safety net (0 disables)")
    args = parser.parse_args()

    keys = args.datasets or _default_dataset_order()
    n, out_path = run_benchmark(
        keys, args.methods, args.models, budget_seconds=args.budget, out_path=args.out,
        resume=not args.no_resume,
        max_cols=args.max_cols or None,
        fit_sample_rows=args.fit_sample_rows or None,
        transform_chunk_rows=args.transform_chunk_rows or None,
        max_mem_gb=args.max_mem_gb or None)
    print(f"wrote {n} result rows to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
