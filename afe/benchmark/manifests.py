"""Build and freeze the two disjoint dataset manifests.

* ``benchmark.json`` -- the evaluation suite from ``registry.BENCHMARK``.
* ``corpus.json``    -- members of ``registry.CORPUS_SUITES`` with every
  benchmark dataset removed (hard-disjointness rule).

Run ``python -m data.manifests`` to (re)generate them. Enumerating the OpenML
suites needs the ``openml`` package; if it is missing we still write the
benchmark manifest and leave the corpus one as a stub so the disjointness test
can run offline.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

from .registry import (
    BENCHMARK,
    CORPUS_MAX_DATASETS,
    CORPUS_SUITES,
    benchmark_names_for_exclusion,
)

MANIFEST_DIR = Path(__file__).resolve().parent / "manifests"
BENCHMARK_MANIFEST = MANIFEST_DIR / "benchmark.json"
CORPUS_MANIFEST = MANIFEST_DIR / "corpus.json"


def build_benchmark_manifest() -> list[dict]:
    return [dataclasses.asdict(s) for s in BENCHMARK]


def _enumerate_suite(suite_id: int | str) -> list[dict]:
    """Return ``[{did, name}]`` members of an OpenML suite (best-effort)."""
    import openml

    suite = openml.study.get_suite(suite_id)
    datasets = openml.datasets.list_datasets(
        data_id=list(suite.data), output_format="dataframe")
    return [{"did": int(r.did), "name": str(r["name"])}
            for _, r in datasets.iterrows()]


def build_corpus_manifest() -> list[dict]:
    """Enumerate corpus suites, drop benchmark overlaps, cap at the RL budget."""
    excluded = benchmark_names_for_exclusion()
    seen_names: set[str] = set()
    corpus: list[dict] = []
    for suite in CORPUS_SUITES:
        try:
            members = _enumerate_suite(suite["suite"])
        except Exception as exc:  # openml missing / offline -> leave a stub
            corpus.append({"suite": suite["label"], "error": str(exc)})
            continue
        for m in members:
            name = m["name"].lower()
            if name in excluded or name in seen_names:
                continue  # disjoint vs benchmark + de-duped across suites
            seen_names.add(name)
            corpus.append({**m, "suite": suite["label"], "task": suite["task"]})
    real = [c for c in corpus if "did" in c]
    return real[:CORPUS_MAX_DATASETS] + [c for c in corpus if "did" not in c]


def write_manifests() -> None:
    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
    BENCHMARK_MANIFEST.write_text(json.dumps(build_benchmark_manifest(), indent=2))
    CORPUS_MANIFEST.write_text(json.dumps(build_corpus_manifest(), indent=2))
    print(f"wrote {BENCHMARK_MANIFEST}")
    print(f"wrote {CORPUS_MANIFEST}")


if __name__ == "__main__":
    write_manifests()
