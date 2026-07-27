"""Scale-invariant feature/target association.

Used by two method-agnostic parts of the harness: the ``feature_efficiency``
metric in ``models.py`` (what fraction of a method's generated features are
individually predictive) and, conceptually, any width cap that needs to rank
columns by usefulness without knowing anything about a particular method.

Kept deliberately dependency-free -- numpy only.
"""

from __future__ import annotations

import numpy as np

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


def target_association(x: np.ndarray, y: np.ndarray, task: str) -> float:
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
