"""AutoFE Benchmark -- compare automatic feature engineering methods fairly.

Every method runs on identical frozen train/test splits, receives identically
preprocessed data, and is scored by the same downstream model panel. Failures
(timeouts, crashes, out-of-memory) are recorded as result rows, never
silently dropped.

Public entrypoint::

    from afe import compare

    result = compare(
        methods=[BaselineMethod, OpenFEMethod, my_own_method],
        datasets=["german-credit"],
    )
    print(result)

The harness itself -- frozen dataset registry, fetch/cache, split protocol,
per-fold encoding, the 3-model scoring panel and the budget-limited runner --
lives in the ``afe.benchmark`` subpackage; the method adapters live in
``afe.methods``. See the repo-root ``README.md`` for usage.
"""

from __future__ import annotations

from .benchmark import compare

__all__ = ["compare"]
