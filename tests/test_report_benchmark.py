"""scripts.report_benchmark guards: fold-mean aggregation, the three-table
layout (overview / per-method breakdown / speed), failure reporting, and
tolerance of the narrower compare()-style row schema."""

from scripts.report_benchmark import build_report


def _row(key="ds-one", method="baseline", family="tree", value=0.5,
         fold="cv0", gen_s=1.0, status="ok", **extra):
    return dict(key=key, method=method, model_family=family, value=value,
                fold_id=fold, gen_elapsed_s=gen_s, status=status,
                task="classification", metric="auc", **extra)


def _rows_two_methods():
    rows = []
    for method, base in (("baseline", 0.5), ("openfe", 0.7)):
        for family in ("tree", "linear", "knn"):
            for fold, bump in (("cv0", 0.0), ("cv1", 0.2)):
                rows.append(_row(method=method, family=family,
                                 value=base + bump, fold=fold, gen_s=10.0))
    return rows


def test_overview_cell_is_mean_of_family_fold_means():
    report = build_report(_rows_two_methods())
    # every family fold-mean is base+0.1, so the overview cell equals it too;
    # openfe (0.800) beats baseline (0.600) and gets bolded.
    assert "| baseline | 0.600 |" in report
    assert "| openfe | **0.800** |" in report


def test_breakdown_groups_model_rows_under_one_method_cell():
    report = build_report(_rows_two_methods())
    lines = report.splitlines()
    start = lines.index("## Per-method scores")
    table = [l for l in lines[start:] if l.startswith("|")]
    # header + 2 methods x 3 family rows
    assert table[0] == "| method | model | DO |"
    assert "| baseline | knn | 0.600 |" in table
    assert "|  | linear | 0.600 |" in table  # method blank on subsequent rows
    assert "| openfe | knn | 0.800 |" in table
    assert sum(1 for l in table if "| tree |" in l or "| linear |" in l
               or "| knn |" in l) == 6


def test_speed_table_shows_fold_mean_time_and_median():
    report = build_report(_rows_two_methods())
    lines = report.splitlines()
    start = lines.index("## Speed (feature-generation wall-time)")
    table = [l for l in lines[start:] if l.startswith("|")]
    assert table[0] == "| method | DO | median |"
    assert "| openfe | 10.0 s | 10.0 s |" in table


def test_failures_are_reported():
    rows = _rows_two_methods() + [
        dict(key="ds-two", method="openfe", status="oom", model_family=None,
             metric=None, value=None, error="MemoryError")]
    report = build_report(rows)
    assert "| ds-two | openfe | oom | 1 |" in report


def test_tolerates_narrow_compare_schema():
    # compare()'s rows lack fold_id/protocol/gen_elapsed_s entirely.
    rows = [dict(key="ds-one", method="baseline", model_family="tree",
                 value=0.5, status="ok", task="classification", metric="auc")]
    report = build_report(rows)
    assert "| baseline | **0.500** |" in report
    assert "| baseline | — | — |" in report  # no timing info -> em-dash cells
