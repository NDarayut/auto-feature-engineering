"""Fetch every benchmark dataset and print its recorded metadata row.

Fails loudly on any 404 / auth / missing-dep error so a broken source is
obvious. Kaggle datasets need ``~/.kaggle/kaggle.json`` + accepted rules;
openfe_reproduce datasets need a manual CSV drop (see download.py).

Usage:
    python -m dev.smoke_download            # all benchmark datasets
    python -m dev.smoke_download nomao german-credit   # a subset by key
"""

from __future__ import annotations

import sys
import traceback

from afe.benchmark.download import load
from afe.benchmark.registry import BENCHMARK

HEADER = f"{'key':24s} {'task':13s} {'rows':>8s} {'feat':>5s} {'cat':>4s} {'sector':20s} license"


def main(keys: list[str]) -> int:
    specs = [s for s in BENCHMARK if not keys or s.key in keys]
    print(HEADER)
    print("-" * len(HEADER))
    failures: list[tuple[str, str]] = []
    for spec in specs:
        try:
            _, m = load(spec)
            print(f"{m['key']:24s} {m['task']:13s} {m['n_rows']:>8d} "
                  f"{m['n_features']:>5d} {m['n_categorical']:>4d} "
                  f"{m['sector']:20s} {m['license']}")
        except Exception as exc:  # noqa: BLE001 -- want every failure surfaced
            failures.append((spec.key, f"{type(exc).__name__}: {exc}"))
            traceback.print_exc(limit=1)
    if failures:
        print("\nFAILURES:")
        for key, msg in failures:
            print(f"  {key}: {msg}")
        return 1
    print("\nall datasets fetched OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
