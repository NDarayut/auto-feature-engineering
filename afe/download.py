"""Fetch datasets named in ``registry.py`` from their canonical source.

Every fetcher returns ``(frame, meta)`` where ``frame`` is a pandas DataFrame
(features + target) and ``meta`` is a dict of recorded fields required by the
plan: id, source, task, target, n_rows, n_features, n_categorical, sector,
license, and a content hash used for the disjointness check.

Heavy third-party libs (openml, sklearn, ucimlrepo, kaggle) are imported lazily
inside each fetcher, so importing this module -- or running the pure-metadata
tests -- never requires them to be installed.

Downloaded data is cached under ``data/cache/`` (gitignored) as parquet, so a
dataset is fetched from the network at most once.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import zipfile
from pathlib import Path

import pandas as pd

from .registry import DatasetSpec

# Downloaded data lives in the gitignored repo-root data/ dir, not in the package.
CACHE_DIR = Path(__file__).resolve().parent.parent / "data" / "cache"
RAW_DIR = CACHE_DIR / "raw"


def _content_hash(frame: pd.DataFrame) -> str:
    """Order-invariant-ish fingerprint: shape + column names + column dtypes.

    Cheap and deterministic; enough to catch a corpus dataset that is literally
    the same table as a benchmark one even if the source names differ.
    """
    payload = json.dumps(
        {
            "shape": list(frame.shape),
            "columns": sorted(map(str, frame.columns)),
            "dtypes": sorted(f"{c}:{t}" for c, t in frame.dtypes.astype(str).items()),
        },
        sort_keys=True,
    ).encode()
    return hashlib.sha256(payload).hexdigest()[:16]


def _meta(spec: DatasetSpec, frame: pd.DataFrame, target: str, license_: str) -> dict:
    n_cat = int(frame.drop(columns=[target], errors="ignore")
                .select_dtypes(include=["object", "category", "bool"]).shape[1])
    return {
        "key": spec.key,
        "display": spec.display,
        "source": spec.source,
        "fetch_key": spec.fetch_key,
        "task": spec.task,
        "sector": spec.sector,
        "scale": spec.scale,
        "target": target,
        "n_rows": int(frame.shape[0]),
        "n_features": int(frame.shape[1] - 1),
        "n_categorical": n_cat,
        "from_openfe": spec.from_openfe,
        "license": license_,
        "content_hash": _content_hash(frame),
    }


# --------------------------------------------------------------------------- #
# Source-specific fetchers
# --------------------------------------------------------------------------- #
def _fetch_sklearn(spec: DatasetSpec) -> tuple[pd.DataFrame, str]:
    from sklearn import datasets as skds

    loader = getattr(skds, f"fetch_{spec.fetch_key}", None) or \
        getattr(skds, f"load_{spec.fetch_key}")
    bunch = loader(as_frame=True)
    frame = bunch.frame.copy()
    target = spec.target or "target"
    if target not in frame.columns and hasattr(bunch, "target"):
        frame[target] = bunch.target
    return frame, target


def _fetch_openml(spec: DatasetSpec) -> tuple[pd.DataFrame, str]:
    from sklearn.datasets import fetch_openml

    version = spec.openml_version if spec.openml_version is not None else "active"
    try:
        bunch = fetch_openml(name=spec.fetch_key, version=version,
                             as_frame=True, parser="auto")
        frame = bunch.frame.copy()
        target = spec.target or bunch.target_names[0]
        return frame, target
    except ValueError as exc:
        if "Sparse ARFF" not in str(exc):
            raise
    # Sparse ARFF (e.g. SensIT-Vehicle): refetch raw and densify to a frame.
    import scipy.sparse as sp

    bunch = fetch_openml(name=spec.fetch_key, version=version, as_frame=False)
    X = bunch.data.toarray() if sp.issparse(bunch.data) else bunch.data
    frame = pd.DataFrame(X, columns=list(bunch.feature_names))
    target = spec.target or (bunch.target_names[0] if bunch.target_names else "target")
    frame[target] = bunch.target
    return frame, target


def _fetch_uci(spec: DatasetSpec) -> tuple[pd.DataFrame, str]:
    from ucimlrepo import fetch_ucirepo

    repo = fetch_ucirepo(name=spec.fetch_key)
    X, y = repo.data.features, repo.data.targets
    target = spec.target or (y.columns[0] if y is not None else "target")
    frame = pd.concat([X, y], axis=1)
    return frame, target


def _fetch_kaggle_competition(spec: DatasetSpec) -> tuple[pd.DataFrame, str]:
    """Download a Kaggle competition's train file via the kaggle CLI.

    Requires ``~/.kaggle/kaggle.json`` and that the competition's rules have
    been accepted in the browser. We deliberately never read an API token from
    the environment or arguments -- credentials stay in the user's local file.
    """
    dest = RAW_DIR / spec.key
    dest.mkdir(parents=True, exist_ok=True)
    if not any(dest.glob("*.csv")):
        subprocess.run(
            ["kaggle", "competitions", "download", "-c", spec.fetch_key,
             "-p", str(dest)],
            check=True,
        )
        # Some competitions nest the CSVs in per-file zips (e.g. train.csv.zip);
        # keep extracting until no zips remain.
        while list(dest.glob("*.zip")):
            for zpath in dest.glob("*.zip"):
                with zipfile.ZipFile(zpath) as zf:
                    zf.extractall(dest)
                zpath.unlink()
    train = next((p for p in dest.glob("*.csv") if "train" in p.name.lower()),
                 None) or next(dest.glob("*.csv"))
    frame = pd.read_csv(train)
    target = spec.target
    if target is None or target not in frame.columns:
        raise ValueError(
            f"{spec.key}: set an explicit `target` in registry for Kaggle "
            f"competitions; columns are {list(frame.columns)[:8]}...")
    return frame, target


def _fetch_openfe_reproduce(spec: DatasetSpec) -> tuple[pd.DataFrame, str]:
    dest = RAW_DIR / spec.key
    train = next((p for p in dest.glob("*.csv") if "train" in p.name.lower()), None)
    if train is None:
        raise FileNotFoundError(
            f"{spec.key}: no CSV containing 'train' found in {dest} -- fetch it "
            "from the IIIS-Li-Group/OpenFE_reproduce mirror's linked data archive "
            "and drop it there manually.")
    frame = pd.read_csv(train)
    target = spec.target or "target"
    return frame, target


_FETCHERS = {
    "sklearn": _fetch_sklearn,
    "openml": _fetch_openml,
    "uci": _fetch_uci,
    "kaggle_competition": _fetch_kaggle_competition,
    "openfe_reproduce": _fetch_openfe_reproduce,
}

# Coarse license tags; refine per dataset as needed.
_LICENSE = {
    "sklearn": "BSD-3 (bundled)",
    "openml": "see OpenML dataset page",
    "uci": "CC BY 4.0 (UCI, typical)",
    "kaggle_competition": "competition rules (non-redistributable)",
    "openfe_reproduce": "see OpenFE_reproduce repo",
}


def load(spec: DatasetSpec, use_cache: bool = True) -> tuple[pd.DataFrame, dict]:
    """Fetch (or load from cache) one dataset; return ``(frame, meta)``."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache = CACHE_DIR / f"{spec.key}.parquet"
    meta_path = CACHE_DIR / f"{spec.key}.meta.json"
    if use_cache and cache.exists() and meta_path.exists():
        return pd.read_parquet(cache), json.loads(meta_path.read_text())

    frame, target = _FETCHERS[spec.source](spec)
    meta = _meta(spec, frame, target, _LICENSE[spec.source])
    frame.to_parquet(cache)
    meta_path.write_text(json.dumps(meta, indent=2))
    return frame, meta
