"""Unary transformation operators for the Feature Transformation Graph.

CAFEM's per-feature agent (PAKDD 2020) navigates a graph whose edges apply a
transformation operator to the current feature. We use the standard LFE/CAFEM
unary operator library -- the transforms that act on a single numeric column
and are cheap, deterministic, and leakage-free (no target, no cross-row fit
beyond simple column statistics computed on the training rows).

Each operator maps a 1-D float array -> 1-D float array of the same length,
must tolerate zeros/negatives/NaN without raising, and returns ``np.nan`` for
undefined entries (cleaned up downstream). ``OPERATORS`` is the ordered action
set the DQN chooses among; the order is frozen so a trained agent's action
indices stay stable.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

_EPS = 1e-9


def _safe(fn: Callable[[np.ndarray], np.ndarray]) -> Callable[[np.ndarray], np.ndarray]:
    """Wrap an operator so inf -> nan and dtype is always float64."""

    def wrapped(x: np.ndarray) -> np.ndarray:
        with np.errstate(all="ignore"):
            out = np.asarray(fn(np.asarray(x, dtype="float64")), dtype="float64")
        return np.where(np.isfinite(out), out, np.nan)

    return wrapped


def _log(x):  # signed log of magnitude: sign(x)*log(1+|x|), defined everywhere
    return np.sign(x) * np.log1p(np.abs(x))


def _sqrt(x):
    return np.sign(x) * np.sqrt(np.abs(x))


def _reciprocal(x):
    return 1.0 / np.where(np.abs(x) < _EPS, np.nan, x)


def _zscore(x):
    mu, sd = np.nanmean(x), np.nanstd(x)
    return (x - mu) / (sd if sd > _EPS else np.nan)


def _minmax(x):
    lo, hi = np.nanmin(x), np.nanmax(x)
    return (x - lo) / ((hi - lo) if (hi - lo) > _EPS else np.nan)


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -60, 60)))


# Ordered, frozen action set. Index in this list == the DQN action id.
OPERATORS: dict[str, Callable[[np.ndarray], np.ndarray]] = {
    "log": _safe(_log),
    "sqrt": _safe(_sqrt),
    "square": _safe(lambda x: x * x),
    "reciprocal": _safe(_reciprocal),
    "sigmoid": _safe(_sigmoid),
    "tanh": _safe(np.tanh),
    "zscore": _safe(_zscore),
    "minmax": _safe(_minmax),
}

OPERATOR_NAMES: tuple[str, ...] = tuple(OPERATORS.keys())
N_OPERATORS: int = len(OPERATOR_NAMES)


def apply_operator(name: str, x: np.ndarray) -> np.ndarray:
    """Apply operator ``name`` to a 1-D array; result is finite-or-NaN float64."""
    return OPERATORS[name](x)
