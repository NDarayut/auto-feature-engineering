"""Dataset sourcing for the MF-OpenFE benchmark + meta-training corpus.

Common entrypoints are re-exported here for convenience::

    from afe import iter_folds, run_benchmark, load, BENCHMARK

Every submodule (``afe.download``, ``afe.eval_data``, ``afe.benchmark``, ...)
remains directly importable too -- this is purely additive.
"""

from .benchmark import run_benchmark
from .download import load
from .eval_data import iter_folds
from .registry import (
    BENCHMARK,
    CORPUS_MAX_DATASETS,
    CORPUS_SUITES,
    DatasetSpec,
    benchmark_names_for_exclusion,
)

__all__ = [
    "BENCHMARK",
    "CORPUS_SUITES",
    "CORPUS_MAX_DATASETS",
    "DatasetSpec",
    "benchmark_names_for_exclusion",
    "iter_folds",
    "load",
    "run_benchmark",
]
