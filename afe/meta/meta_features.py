"""LFE-style Quantile Sketch Array (QSA) + scalar meta-features for one feature.

LFE (IJCAI 2017) represents a feature to its meta-model by a *quantile sketch*:
a fixed-size, scale-invariant summary of the feature's value distribution, so
the same learned model applies across datasets with wildly different feature
scales. We use that sketch as the state for the Stage-0 RL agent *and* as the
input to the Stage-1 meta-model, so both see a feature the same way.

``feature_sketch(x, y, task)`` returns a fixed-length float vector:

* ``QSA_BINS`` values -- the quantile sketch: the feature is rank-normalized to
  [0, 1] (scale/outlier invariant) then histogrammed into equal-width bins,
  giving the distribution shape.
* a handful of scalar meta-features -- dispersion/shape stats plus a
  target-association term (the one target-aware summary; computed on training
  rows only by the caller), which is what makes "is this feature useful"
  learnable rather than purely distributional.

The length is constant (``SKETCH_DIM``) regardless of dataset, which is the
whole point -- one meta-model, any dataset.
"""

from __future__ import annotations

import numpy as np

QSA_BINS = 16
_SCALAR_NAMES = (
    "frac_unique", "frac_zero", "frac_negative", "log_abs_mean",
    "cv", "skew", "kurtosis", "target_assoc",
)
SKETCH_DIM = QSA_BINS + len(_SCALAR_NAMES)
_EPS = 1e-9


def _rank_normalize(x: np.ndarray) -> np.ndarray:
    """Map finite values to [0, 1] by rank (empirical CDF); NaN -> dropped."""
    finite = x[np.isfinite(x)]
    if finite.size == 0:
        return finite
    order = finite.argsort()
    ranks = np.empty(finite.size, dtype="float64")
    ranks[order] = np.arange(finite.size)
    return ranks / max(finite.size - 1, 1)


def _quantile_sketch(x: np.ndarray) -> np.ndarray:
    """Rank-normalize then histogram into ``QSA_BINS`` bins, sum-normalized."""
    u = _rank_normalize(x)
    if u.size == 0:
        return np.zeros(QSA_BINS, dtype="float64")
    counts, _ = np.histogram(u, bins=QSA_BINS, range=(0.0, 1.0))
    total = counts.sum()
    return counts / total if total > 0 else counts.astype("float64")


def _target_association(x: np.ndarray, y: np.ndarray, task: str) -> float:
    """Scale-invariant |association| between feature and target in [0, 1].

    Regression: |Spearman| (rank correlation, robust to the feature's scale).
    Classification: normalized between-class dispersion of the feature's ranks
    (a correlation-ratio style eta), which needs no per-class binning and works
    for any number of classes.
    """
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 3:
        return 0.0
    xr = _rank_normalize(x[mask])
    yv = y[mask]
    if task == "regression":
        yr = _rank_normalize(yv)
        if xr.std() < _EPS or yr.std() < _EPS:
            return 0.0
        return float(abs(np.corrcoef(xr, yr)[0, 1]))
    # classification: correlation ratio eta between feature ranks and classes
    grand = xr.mean()
    ss_total = float(((xr - grand) ** 2).sum())
    if ss_total < _EPS:
        return 0.0
    ss_between = 0.0
    for cls in np.unique(yv):
        g = xr[yv == cls]
        if g.size:
            ss_between += g.size * (g.mean() - grand) ** 2
    return float(np.sqrt(max(ss_between, 0.0) / ss_total))


def _scalars(x: np.ndarray, y: np.ndarray, task: str) -> np.ndarray:
    finite = x[np.isfinite(x)]
    n = finite.size
    if n == 0:
        return np.zeros(len(_SCALAR_NAMES), dtype="float64")
    mean, std = finite.mean(), finite.std()
    frac_unique = np.unique(finite).size / n
    frac_zero = float(np.mean(np.abs(finite) < _EPS))
    frac_negative = float(np.mean(finite < 0))
    log_abs_mean = float(np.log1p(np.abs(mean)))
    cv = float(std / (abs(mean) + _EPS))
    if std < _EPS:
        skew = kurt = 0.0
    else:
        z = (finite - mean) / std
        skew = float(np.mean(z ** 3))
        kurt = float(np.mean(z ** 4) - 3.0)
    assoc = _target_association(x, y, task)
    return np.array([frac_unique, frac_zero, frac_negative, log_abs_mean,
                     cv, skew, kurt, assoc], dtype="float64")


def feature_sketch(x: np.ndarray, y: np.ndarray, task: str) -> np.ndarray:
    """Fixed-length (``SKETCH_DIM``) meta-feature vector for one feature."""
    x = np.asarray(x, dtype="float64")
    y = np.asarray(y, dtype="float64")
    sketch = _quantile_sketch(x)
    scalars = _scalars(x, y, task)
    vec = np.concatenate([sketch, scalars])
    return np.where(np.isfinite(vec), vec, 0.0)
