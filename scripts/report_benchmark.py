"""Aggregate afe.benchmark's JSONL results into the draft_plan Sec. 6 report
structure: split by task type -> sector -> aggregate overall, each row shows
baseline vs method with the point delta on the single fixed-seed split
(draft_plan Sec. 5.3 -- one estimate per (dataset, method, model), not a
distribution, so no mean/std/significance marker is computed).

Failure rows (status != "ok") are counted and reported, not dropped
(draft_plan Sec. 6).

    python -m scripts.report_benchmark results/benchmark_results.jsonl
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from afe.benchmark.registry import BENCHMARK

_SECTOR = {s.key: s.sector for s in BENCHMARK}


def load_rows(path: Path) -> list[dict]:
    rows = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def build_report(rows: list[dict]) -> str:
    ok_rows = [r for r in rows if r.get("status") == "ok" and r.get("value") is not None]
    fail_rows = [r for r in rows if r.get("status") != "ok"]

    # group: (dataset, method, model_family) -> value (one point estimate)
    grouped: dict[tuple, float] = {}
    task_of: dict[str, str] = {}
    for r in ok_rows:
        grouped[(r["key"], r["method"], r["model_family"])] = r["value"]
        task_of[r["key"]] = r["task"]

    datasets = sorted({k for k, _, _ in grouped})
    methods = sorted({m for _, m, _ in grouped} - {"baseline"})
    families = sorted({f for _, _, f in grouped})

    lines = ["# Benchmark Report", ""]
    for task in sorted({task_of[k] for k in datasets}):
        lines.append(f"## Task: {task}")
        by_sector: dict[str, list[str]] = defaultdict(list)
        for key in datasets:
            if task_of[key] != task:
                continue
            by_sector[_SECTOR.get(key, "unknown")].append(key)

        for sector, keys in sorted(by_sector.items()):
            lines.append(f"### Sector: {sector}")
            lines.append("")
            lines.append("| dataset | model | baseline | " +
                        " | ".join(methods) + " |")
            lines.append("|---|---|---|" + "---|" * len(methods))
            for key in sorted(keys):
                for family in families:
                    base_val = grouped.get((key, "baseline", family))
                    if base_val is None:
                        continue
                    cells = [f"{base_val:.3f}"]
                    for method in methods:
                        val = grouped.get((key, method, family))
                        if val is None:
                            cells.append("—")
                            continue
                        delta = val - base_val
                        sign = "+" if delta >= 0 else ""
                        cells.append(f"{val:.3f} ({sign}{delta:.3f})")
                    lines.append(f"| {key} | {family} | " + " | ".join(cells) + " |")
            lines.append("")

    lines.append("## Failures / timeouts / crashes")
    lines.append("")
    if not fail_rows:
        lines.append("(none)")
    else:
        by_status: dict[str, int] = defaultdict(int)
        for r in fail_rows:
            by_status[(r["key"], r["method"], r["status"])] += 1
        lines.append("| dataset | method | status | count |")
        lines.append("|---|---|---|---|")
        for (key, method, status), count in sorted(by_status.items()):
            lines.append(f"| {key} | {method} | {status} | {count} |")

    return "\n".join(lines)


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
