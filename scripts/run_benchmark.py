"""CLI entry point for the benchmark harness (afe.benchmark.run_benchmark).

Runs (dataset, method) pairs sequentially -- one dataset at a time, freeing
its memory before moving to the next -- and is resumable (skips pairs
already present in --out). Datasets are ordered smallest-scale-first by
default.

Examples:
    python -m scripts.run_benchmark --datasets nomao concrete-strength \\
        --methods baseline openfe --budget 300

    # everything
    python -m scripts.run_benchmark --methods baseline openfe featuretools autofeat \\
        --budget 300

    # only the tree model, to skip linear/knn scoring
    python -m scripts.run_benchmark --methods baseline openfe --models tree
"""

from __future__ import annotations

import argparse
import warnings

from afe.benchmark import DEFAULT_BUDGET_SECONDS, run_benchmark
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
                        choices=sorted(METHODS),
                        help="which AutoFE methods to benchmark")
    parser.add_argument("--models", nargs="*", default=list(MODEL_FAMILIES),
                        choices=list(MODEL_FAMILIES),
                        help="which model families to evaluate generated features with")
    parser.add_argument("--budget", type=float, default=DEFAULT_BUDGET_SECONDS,
                        help="per (dataset, method) generation time budget, seconds")
    parser.add_argument("--out", default=None, help="output JSONL path")
    parser.add_argument("--no-resume", action="store_true",
                        help="re-run pairs even if already present in --out")
    args = parser.parse_args()

    keys = args.datasets or _default_dataset_order()
    n, out_path = run_benchmark(
        keys, args.methods, args.models, budget_seconds=args.budget, out_path=args.out,
        resume=not args.no_resume)
    print(f"wrote {n} result rows to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
