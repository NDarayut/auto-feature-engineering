"""Load a meta-training-corpus dataset by OpenML ``did`` (algorithm_plan Sec. 3).

The benchmark suite is loaded through ``afe.download`` by ``DatasetSpec``; the
corpus is different -- ``afe/manifests/corpus.json`` lists members only by
OpenML ``did``/``name`` (the disjoint set produced by ``afe.manifests``). This
module is their loader: fetch by ``did``, cache to parquet, and hand back a
fully-numeric, NaN-free feature matrix plus target ready for the RL search.

Kept deliberately separate from ``afe.download`` because corpus datasets are
never part of the benchmark and must not accidentally be evaluated against it
(the hard-disjointness rule). Heavy imports (openml/sklearn) are lazy so the
manifest tests stay offline.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import numpy as np
import pandas as pd

CORPUS_MANIFEST = (Path(__file__).resolve().parent.parent / "benchmark"
                    / "manifests" / "corpus.json")
CACHE_DIR = (Path(__file__).resolve().parent.parent.parent / "data" / "cache"
             / "corpus")


@dataclasses.dataclass(frozen=True)
class CorpusDataset:
    """A prepped corpus dataset: numeric features + target + task type."""

    did: int
    name: str
    task: str  # "classification" | "regression"
    X: pd.DataFrame  # fully numeric, NaN-free
    y: np.ndarray  # 1-D; class codes for classification, floats for regression
    feature_names: tuple[str, ...]


def load_corpus_manifest() -> list[dict]:
    """Return the frozen corpus entries that carry a real OpenML ``did``."""
    if not CORPUS_MANIFEST.exists():
        return []
    entries = json.loads(CORPUS_MANIFEST.read_text())
    return [e for e in entries if "did" in e]


def _prep_numeric(frame: pd.DataFrame, target: str, task: str) -> CorpusDataset | None:
    """Ordinal-code categoricals, median-impute numerics, coerce target.

    Corpus datasets are only ever used as RL search substrates, never for the
    benchmark, so a single dataset-wide numeric prep (no train/eval leakage
    concern across *the benchmark*) is acceptable here -- the RL wrapper does
    its own internal train/eval split for reward estimation.
    """
    if target not in frame.columns:
        return None
    y_raw = frame[target]
    X = frame.drop(columns=[target])

    # Drop columns that are entirely missing or constant -- no signal, and
    # constant columns break several transformation operators (e.g. zscore).
    X = X.loc[:, X.notna().any(axis=0)]
    nunique = X.nunique(dropna=False)
    X = X.loc[:, nunique > 1]
    if X.shape[1] == 0:
        return None

    num_parts = []
    for col in X.columns:
        s = X[col]
        if s.dtype.kind in "biufc":
            v = pd.to_numeric(s, errors="coerce")
        else:
            # Ordinal-code categoricals by first-appearance order.
            v = pd.Series(pd.factorize(s, use_na_sentinel=True)[0],
                          index=s.index, dtype="float64")
            v = v.replace(-1, np.nan)
        num_parts.append(v.rename(col))
    Xn = pd.concat(num_parts, axis=1)
    Xn = Xn.fillna(Xn.median(numeric_only=True))
    # Any column still all-NaN after median-fill (median undefined) -> drop.
    Xn = Xn.loc[:, Xn.notna().all(axis=0)]
    if Xn.shape[1] == 0:
        return None

    if task == "regression":
        y = pd.to_numeric(y_raw, errors="coerce").to_numpy(dtype="float64")
        keep = ~np.isnan(y)
    else:
        codes = pd.factorize(y_raw, use_na_sentinel=True)[0]
        y = codes.astype("float64")
        keep = codes >= 0
    if not keep.all():
        Xn, y = Xn.loc[keep], y[keep]
    if len(Xn) == 0:
        return None

    return CorpusDataset(
        did=-1, name="", task=task, X=Xn.reset_index(drop=True),
        y=y, feature_names=tuple(map(str, Xn.columns)),
    )


def load_corpus_dataset(
    did: int, name: str, task: str, use_cache: bool = True,
    max_rows: int | None = 20_000,
) -> CorpusDataset | None:
    """Fetch OpenML dataset ``did``, prep it, and cache the raw frame.

    ``max_rows`` subsamples very large corpus datasets (deterministically) so
    Stage-0 RL search stays affordable -- the point of the corpus is breadth
    across datasets, not exhaustive per-dataset scale (algorithm_plan Sec. 3).
    Returns ``None`` if the dataset can't be turned into a usable numeric table.
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache = CACHE_DIR / f"{did}.parquet"
    meta_path = CACHE_DIR / f"{did}.meta.json"

    if use_cache and cache.exists() and meta_path.exists():
        frame = pd.read_parquet(cache)
        target = json.loads(meta_path.read_text())["target"]
    else:
        from sklearn.datasets import fetch_openml

        bunch = fetch_openml(data_id=int(did), as_frame=True, parser="auto")
        frame = bunch.frame.copy()
        target = (bunch.target_names[0] if getattr(bunch, "target_names", None)
                  else frame.columns[-1])
        frame.to_parquet(cache)
        meta_path.write_text(json.dumps({"did": int(did), "name": name,
                                         "target": target}, indent=2))

    if max_rows is not None and len(frame) > max_rows:
        frame = frame.sample(n=max_rows, random_state=0).reset_index(drop=True)

    ds = _prep_numeric(frame, target, task)
    if ds is None:
        return None
    return dataclasses.replace(ds, did=int(did), name=name)
