"""Aggregate a benchmark JSONL into a markdown report.

The aggregation itself lives in ``afe.benchmark.report`` so that
``run_benchmark()`` and ``compare()`` can produce a report inline; this
script is the standalone CLI for regenerating one from a results file.
Reports are built purely from the JSONL -- regenerating never re-runs the
benchmark.

    python -m scripts.report_benchmark results/benchmark_results.jsonl
    python -m scripts.report_benchmark results/my_run.jsonl --out report.md
"""

from __future__ import annotations

import argparse
from pathlib import Path

from afe.benchmark.report import build_report, load_rows

__all__ = ["build_report", "load_rows"]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("jsonl_path")
    parser.add_argument("--out", default=None, help="write report to this path (default: stdout)")
    args = parser.parse_args()

    rows = load_rows(Path(args.jsonl_path))
    report = build_report(rows)
    if args.out:
        Path(args.out).write_text(report)
        print(f"wrote {args.out}")
    else:
        print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
