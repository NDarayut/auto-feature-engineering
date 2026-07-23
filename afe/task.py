"""Shared classification/regression task-type inference.

Used anywhere a target ``y`` needs a best-effort task guess when the caller
doesn't specify one explicitly -- e.g. ``afe.meta.online.MFOpenFE`` and
``afe.benchmark.compare``. Deliberately dependency-free and not coupled to
any particular algorithm or dataset registry.
"""

from __future__ import annotations

from typing import Literal

import pandas as pd

Task = Literal["classification", "regression"]


def infer_task(y: pd.Series) -> Task:
    """Best-effort classification/regression guess; override if it's wrong."""
    y = pd.Series(y)
    if y.dtype.kind in "OSUb":  # object / string / bool
        return "classification"
    n_unique = y.nunique()
    if y.dtype.kind == "f":
        return "classification" if n_unique <= 2 else "regression"
    return "classification" if n_unique <= min(20, max(2, len(y) // 20)) else "regression"
