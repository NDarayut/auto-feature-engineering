"""AutoFE benchmark harness -- comparing methods against each other.

This subpackage is the comparison harness: a frozen dataset registry
(`registry.py`), fetch+cache (`download.py`), disjoint benchmark/corpus
manifests (`manifests.py`), a single fixed-seed split protocol (`splits.py`),
per-model-family encoding + fold iteration (`eval_data.py`), a 3-model
scoring panel (`models.py`), and the budget-limited benchmark runner
(`benchmark.py`) that compares baseline vs. OpenFE vs. CAFEM vs.
Featuretools vs. Autofeat.

Also home to `compare()` (implemented in `_compare.py`) -- a standalone,
algorithm-agnostic API for benchmarking *any* AutoFE method (yours, ours, or
a mix) against either the built-in datasets or your own data.

See the repo-root `README.md` for usage.
"""

from __future__ import annotations

from ..methods import (
    METHODS,
    AutoFEMethod,
    AutofeatMethod,
    BaselineMethod,
    CAFEMMethod,
    FeaturetoolsMethod,
    OpenFEMethod,
)
from ._compare import CompareResult, compare
from .benchmark import (DEFAULT_BUDGET_SECONDS, DEFAULT_FIT_SAMPLE_ROWS,
                        DEFAULT_MAX_COLS, DEFAULT_MAX_MEM_GB,
                        DEFAULT_TRANSFORM_CHUNK_ROWS, run_benchmark)
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
    "METHODS",
    "AutoFEMethod",
    "AutofeatMethod",
    "BaselineMethod",
    "CAFEMMethod",
    "CompareResult",
    "DEFAULT_BUDGET_SECONDS",
    "DEFAULT_FIT_SAMPLE_ROWS",
    "DEFAULT_MAX_COLS",
    "DEFAULT_MAX_MEM_GB",
    "DEFAULT_TRANSFORM_CHUNK_ROWS",
    "DatasetSpec",
    "FeaturetoolsMethod",
    "OpenFEMethod",
    "benchmark_names_for_exclusion",
    "compare",
    "iter_folds",
    "load",
    "run_benchmark",
]
