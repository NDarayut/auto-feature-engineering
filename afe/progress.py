"""Progress reporting for long-running AutoFE stages.

A single small reporter used by ``MFOpenFE`` (and reusable by anything else
in the package): short stage markers with elapsed time for one-shot steps
(data prep, fitting a verification model, ...), and a ``tqdm`` progress bar
for genuinely iterable steps (scoring N candidates one at a time).

Degrades gracefully with ``enabled=False`` (silent) and if ``tqdm`` isn't
installed (falls back to the plain iterable, stage markers still print).
"""

from __future__ import annotations

import time
from typing import Iterable, Iterator, TypeVar

T = TypeVar("T")


class ProgressReporter:
    """``enabled=True`` prints stage markers and shows tqdm bars for loops."""

    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self._t0 = time.time()

    def stage(self, message: str) -> None:
        """Print a one-line stage marker with elapsed time since construction."""
        if not self.enabled:
            return
        print(f"[MF-OpenFE {time.time() - self._t0:6.1f}s] {message}")

    def iter(self, iterable: Iterable[T], desc: str = "", total: int | None = None) -> Iterator[T]:
        """Wrap ``iterable`` in a tqdm bar if enabled; passthrough otherwise."""
        if not self.enabled:
            return iter(iterable)
        try:
            from tqdm import tqdm
        except ImportError:
            return iter(iterable)
        return iter(tqdm(iterable, desc=desc, total=total, leave=False))

    def done(self, message: str) -> None:
        """Final marker -- same as ``stage`` (kept as a distinct name for clarity
        at call sites)."""
        self.stage(message)
