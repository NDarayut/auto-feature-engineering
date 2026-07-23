"""Coverage guards: the benchmark suite must span draft_plan Sec. 2's grid."""

from collections import defaultdict

from afe.benchmark.registry import BENCHMARK


def _sector_family(sector: str) -> str:
    s = sector.lower()
    if "finance" in s or "telco" in s:
        return "finance"
    if "health" in s:
        return "healthcare"
    if "material" in s or "physical" in s or "chem" in s or "energy" in s:
        return "physical-science"
    return "general"


def test_both_task_types_present():
    tasks = {s.task for s in BENCHMARK}
    assert "regression" in tasks
    assert tasks & {"classification", "multiclass"}


def test_required_sectors_present():
    fams = {_sector_family(s.sector) for s in BENCHMARK}
    for required in ["finance", "healthcare", "physical-science", "general"]:
        assert required in fams, f"no benchmark dataset in sector {required}"


def test_scale_spans_small_to_large():
    scales = {s.scale for s in BENCHMARK}
    assert {"small", "medium", "large"} <= scales


def test_openfe_ten_present():
    assert sum(s.from_openfe for s in BENCHMARK) == 10


def test_print_coverage_matrix(capsys):
    matrix: dict[str, set[str]] = defaultdict(set)
    for s in BENCHMARK:
        matrix[_sector_family(s.sector)].add(
            "regression" if s.task == "regression" else "classification")
    # Every sector should have at least one dataset; report the grid for humans.
    for fam, tasks in sorted(matrix.items()):
        print(f"{fam:18s} -> {sorted(tasks)}")
    assert len(matrix) >= 4
