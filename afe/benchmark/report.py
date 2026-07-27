"""Aggregate afe.benchmark's JSONL results into markdown tables:

1. **Overview** -- one row per method, one column per dataset; the cell is
   the mean of the three model-family scores (each family's score first
   averaged over folds), so a method's overall usefulness on a dataset is
   one number.
2. **Per-method scores** -- the breakdown behind the overview: a ``method``
   column grouping three ``model`` rows (tree/linear/knn), one column per
   dataset, cell = fold-mean metric value for that (method, family, dataset).
3. **Speed** -- one row per method, one column per dataset, cell = fold-mean
   feature-generation wall-time, plus a per-method median column.
4. **By task** / **By sector** -- mean overview score per method, grouped by
   task (classification/multiclass/regression) and by dataset sector
   (``afe.benchmark.registry.DatasetSpec.sector``), so results can be read at
   a coarser grain than one row per dataset.

Dataset columns use abbreviations (initials of the hyphen-separated key);
a legend up top maps each abbreviation to the full key, task, sector, and
metric. Columns/rows throughout are grouped by task then sector then key,
so the task/sector structure is visible even in the per-dataset tables.

Failure rows (status != "ok") are counted and reported, not dropped.

``run_benchmark()`` and ``compare()`` call ``build_report`` themselves when
given a ``report_path``; ``python -m scripts.report_benchmark <jsonl>``
regenerates a report from a results file at any later time.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from statistics import mean, median

from afe.benchmark.registry import BENCHMARK

SECTOR_OF: dict[str, str] = {spec.key: spec.sector for spec in BENCHMARK}


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
    n_before: dict[tuple, list[float]] = defaultdict(list)
    n_after: dict[tuple, list[float]] = defaultdict(list)
    task_of: dict[str, str] = {}
    metric_of: dict[str, str] = {}
    for r in ok_rows:
        values[(r["key"], r["method"], r["model_family"])].append(r["value"])
        if r.get("gen_elapsed_s") is not None:
            times[(r["key"], r["method"])].append(r["gen_elapsed_s"])
        # n_features_generated/n_features_final are computed once per
        # (dataset, method) generation step and repeated on every
        # model_family row for that pair -- constant within the pair, so
        # averaging just guards against any float noise.
        if r.get("n_features_final") is not None and r.get("n_features_generated") is not None:
            n_after[(r["key"], r["method"])].append(r["n_features_final"])
            n_before[(r["key"], r["method"])].append(
                r["n_features_final"] - r["n_features_generated"])
        task_of[r["key"]] = r.get("task", "?")
        if r.get("metric"):
            metric_of[r["key"]] = r["metric"]
    score: dict[tuple, float] = {k: mean(v) for k, v in values.items()}
    gen_s: dict[tuple, float] = {k: mean(v) for k, v in times.items()}
    n_before_of: dict[tuple, float] = {k: mean(v) for k, v in n_before.items()}
    n_after_of: dict[tuple, float] = {k: mean(v) for k, v in n_after.items()}

    sector_of: dict[str, str] = {k: SECTOR_OF.get(k, "?") for k in task_of}
    # Group by task then sector then key everywhere, so the task/sector
    # structure is visible even in tables keyed one-column-per-dataset.
    datasets = sorted({k for k, _, _ in score},
                       key=lambda k: (task_of.get(k, "?"), sector_of.get(k, "?"), k))
    methods = _method_order({m for _, m, _ in score})
    families = sorted({f for _, _, f in score})
    abbrev = _abbreviate(datasets)

    lines = ["# Benchmark Report", ""]

    # -- legend ------------------------------------------------------------
    lines += ["## Datasets", "",
              "| abbrev | dataset | task | sector | metric |",
              "|---|---|---|---|---|"]
    for key in datasets:
        lines.append(f"| {abbrev[key]} | {key} | {task_of.get(key, '?')} "
                     f"| {sector_of.get(key, '?')} | {metric_of.get(key, '?')} |")
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

    # -- table 4: feature counts --------------------------------------------
    lines += ["## Feature counts (before -> after)", "",
              "Number of columns fed into the downstream models before "
              "(original, post max-cols-cap) and after a method's generated "
              "features are added.", "",
              "| method | " + " | ".join(abbrev[k] for k in datasets) + " |",
              "|---|" + "---|" * len(datasets)]
    for method in methods:
        cells = []
        for key in datasets:
            before, after = n_before_of.get((key, method)), n_after_of.get((key, method))
            if before is None or after is None:
                cells.append("—")
            else:
                cells.append(f"{before:.0f} -> {after:.0f}")
        lines.append(f"| {method} | " + " | ".join(cells) + " |")
    lines.append("")

    # -- table 5/6: grouped by task, then by sector -------------------------
    def _grouped_table(title: str, group_of: dict[str, str], group_order: list[str]) -> None:
        lines.append(f"## {title}")
        lines.append("")
        lines.append("Mean of the per-dataset overview score (mean across model "
                     "families) over each dataset's group; a method missing on "
                     "every dataset in a group shows —.")
        lines.append("")
        lines.append("| method | " + " | ".join(group_order) + " |")
        lines.append("|---|" + "---|" * len(group_order))
        def _group_cell(method: str, g: str) -> float | None:
            keys_in_group = [k for k in datasets if group_of.get(k) == g]
            per_key = [v for k in keys_in_group
                       if (v := _overview_cell(method, k)) is not None]
            return mean(per_key) if per_key else None

        group_best = {g: max((v for m in methods if (v := _group_cell(m, g)) is not None),
                             default=None) for g in group_order}
        for method in methods:
            cells = []
            for g in group_order:
                val = _group_cell(method, g)
                if val is None:
                    cells.append("—")
                else:
                    cells.append(f"**{_fmt_score(val)}**" if val == group_best[g]
                                 else _fmt_score(val))
            lines.append(f"| {method} | " + " | ".join(cells) + " |")
        lines.append("")

    task_order = sorted({task_of.get(k, "?") for k in datasets})
    _grouped_table("By task", task_of, task_order)

    sector_order = sorted({sector_of.get(k, "?") for k in datasets})
    _grouped_table("By sector", sector_of, sector_order)

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
