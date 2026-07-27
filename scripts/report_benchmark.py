"""Aggregate afe.benchmark's JSONL results into three markdown tables:

1. **Overview** -- one row per method, one column per dataset; the cell is
   the mean of the three model-family scores (each family's score first
   averaged over folds), so a method's overall usefulness on a dataset is
   one number.
2. **Per-method scores** -- the breakdown behind the overview: a ``method``
   column grouping three ``model`` rows (tree/linear/knn), one column per
   dataset, cell = fold-mean metric value for that (method, family, dataset).
3. **Speed** -- one row per method, one column per dataset, cell = fold-mean
   feature-generation wall-time, plus a per-method median column.

Dataset columns use abbreviations (initials of the hyphen-separated key);
a legend up top maps each abbreviation to the full key, task, and metric.

Failure rows (status != "ok") are counted and reported, not dropped
(draft_plan Sec. 6).

    python -m scripts.report_benchmark results/benchmark_results.jsonl
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean, median


def load_rows(path: Path) -> list[dict]:
    rows = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _abbreviate(keys: list[str]) -> dict[str, str]:
    """dataset key -> short column label (initials of hyphenated words).

    ``california-housing`` -> ``CH``; collisions get a numeric suffix so
    every column label stays unique regardless of which keys are present.
    """
    abbrevs: dict[str, str] = {}
    used: set[str] = set()
    for key in keys:
        base = "".join(w[0] for w in key.split("-")).upper()
        label, i = base, 2
        while label in used:
            label, i = f"{base}{i}", i + 1
        abbrevs[key] = label
        used.add(label)
    return abbrevs


def _fmt_score(v: float) -> str:
    # A diverged model (e.g. a linear fit on an exploded feature) can produce
    # an astronomically negative R^2; scientific notation keeps the table
    # readable instead of printing a 20-digit cell.
    return f"{v:.2e}" if abs(v) >= 1000 else f"{v:.3f}"


def _fmt_seconds(s: float) -> str:
    if s >= 120:
        return f"{s / 60:.1f} min"
    return f"{s:.1f} s"


def _method_order(methods: set[str]) -> list[str]:
    # baseline first -- it is the reference every other method is read against.
    return (["baseline"] if "baseline" in methods else []) + sorted(methods - {"baseline"})


def build_report(rows: list[dict]) -> str:
    ok_rows = [r for r in rows if r.get("status") == "ok" and r.get("value") is not None]
    fail_rows = [r for r in rows if r.get("status") != "ok"]

    # Fold-mean per (dataset, method, family) -- files may hold one row per
    # fold (cv5/seed_repeat5 protocols); collapsing by mean first makes every
    # downstream cell a fold-mean rather than an arbitrary fold's value.
    values: dict[tuple, list[float]] = defaultdict(list)
    times: dict[tuple, list[float]] = defaultdict(list)
    task_of: dict[str, str] = {}
    metric_of: dict[str, str] = {}
    for r in ok_rows:
        values[(r["key"], r["method"], r["model_family"])].append(r["value"])
        if r.get("gen_elapsed_s") is not None:
            times[(r["key"], r["method"])].append(r["gen_elapsed_s"])
        task_of[r["key"]] = r.get("task", "?")
        if r.get("metric"):
            metric_of[r["key"]] = r["metric"]
    score: dict[tuple, float] = {k: mean(v) for k, v in values.items()}
    gen_s: dict[tuple, float] = {k: mean(v) for k, v in times.items()}

    datasets = sorted({k for k, _, _ in score})
    methods = _method_order({m for _, m, _ in score})
    families = sorted({f for _, _, f in score})
    abbrev = _abbreviate(datasets)

    lines = ["# Benchmark Report", ""]

    # -- legend ------------------------------------------------------------
    lines += ["## Datasets", "",
              "| abbrev | dataset | task | metric |", "|---|---|---|---|"]
    for key in datasets:
        lines.append(f"| {abbrev[key]} | {key} | {task_of.get(key, '?')} "
                     f"| {metric_of.get(key, '?')} |")
    lines.append("")

    # -- table 1: overview -------------------------------------------------
    def _overview_cell(method: str, key: str) -> float | None:
        vals = [score[(key, method, f)] for f in families if (key, method, f) in score]
        return mean(vals) if vals else None

    lines += ["## Overview", "",
              "Mean held-out score across the three model families "
              "(per-family scores are fold-means; metrics differ per dataset "
              "-- see the legend).", "",
              "| method | " + " | ".join(abbrev[k] for k in datasets) + " |",
              "|---|" + "---|" * len(datasets)]
    best = {k: max((v for m in methods if (v := _overview_cell(m, k)) is not None),
                   default=None) for k in datasets}
    for method in methods:
        cells = []
        for key in datasets:
            val = _overview_cell(method, key)
            if val is None:
                cells.append("—")
            else:
                cells.append(f"**{_fmt_score(val)}**" if val == best[key]
                             else _fmt_score(val))
        lines.append(f"| {method} | " + " | ".join(cells) + " |")
    lines.append("")

    # -- table 2: per-method breakdown ------------------------------------
    lines += ["## Per-method scores", "",
              "| method | model | " + " | ".join(abbrev[k] for k in datasets) + " |",
              "|---|---|" + "---|" * len(datasets)]
    for method in methods:
        for i, family in enumerate(families):
            cells = []
            for key in datasets:
                val = score.get((key, method, family))
                cells.append(_fmt_score(val) if val is not None else "—")
            label = method if i == 0 else ""
            lines.append(f"| {label} | {family} | " + " | ".join(cells) + " |")
    lines.append("")

    # -- table 3: speed ----------------------------------------------------
    lines += ["## Speed (feature-generation wall-time)", "",
              "| method | " + " | ".join(abbrev[k] for k in datasets) + " | median |",
              "|---|" + "---|" * (len(datasets) + 1)]
    for method in methods:
        cells, method_times = [], []
        for key in datasets:
            t = gen_s.get((key, method))
            if t is None:
                cells.append("—")
            else:
                cells.append(_fmt_seconds(t))
                method_times.append(t)
        med = _fmt_seconds(median(method_times)) if method_times else "—"
        lines.append(f"| {method} | " + " | ".join(cells) + f" | {med} |")
    lines.append("")

    # -- failures ----------------------------------------------------------
    lines += ["## Failures / timeouts / crashes", ""]
    if not fail_rows:
        lines.append("(none)")
    else:
        by_status: dict[tuple[str, str, str], int] = defaultdict(int)
        for r in fail_rows:
            by_status[(r["key"], r["method"], r["status"])] += 1
        lines += ["| dataset | method | status | count |", "|---|---|---|---|"]
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
