"""Dataset sourcing for the MF-OpenFE benchmark + meta-training corpus."""

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
]
