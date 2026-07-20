"""Declarative dataset registry for the MF-OpenFE project.

Two disjoint pools are defined here (see the approved dataset plan):

* ``BENCHMARK``  -- the ~22-dataset evaluation suite (draft_plan.md).
* ``CORPUS_SUITES`` -- curated OpenML suites used, minus the benchmark, as the
  meta-training corpus (algorithm_plan.md Sec. 3).

Nothing here downloads anything; it is pure metadata. ``data/download.py``
consumes these specs, and ``data/manifests.py`` freezes the disjoint split.

Design choice: datasets are addressed by *name / slug* for their source rather
than by numeric id, so a wrong hand-copied OpenML/UCI id can never silently
fetch the wrong table. ``download.py`` resolves the name at fetch time.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

Task = Literal["classification", "multiclass", "regression"]
Source = Literal["sklearn", "openml", "uci", "kaggle_competition", "openfe_reproduce"]
Scale = Literal["small", "medium", "large"]


@dataclass(frozen=True)
class DatasetSpec:
    """One benchmark dataset and everything needed to fetch + tag it.

    ``fetch_key`` is the identifier understood by ``source`` (an OpenML/UCI
    dataset name, a sklearn loader name, or a Kaggle competition slug).
    """

    key: str  # our stable internal id (kebab-case)
    display: str
    task: Task
    sector: str
    scale: Scale
    source: Source
    fetch_key: str
    target: str | None = None  # target column name where source doesn't fix it
    openml_version: int | None = None  # pin when OpenML has >1 active version
    from_openfe: bool = False  # part of OpenFE's original 10-dataset benchmark
    note: str = ""
    aliases: tuple[str, ...] = field(default_factory=tuple)  # names to also
    # treat as "this dataset" during the corpus disjointness check


# --- Unified benchmark suite (one suite; `from_openfe` flags the OpenFE 10) ---
BENCHMARK: list[DatasetSpec] = [
    # ---- OpenFE's original 10 (parity anchor) ----
    DatasetSpec("california-housing", "California Housing", "regression",
                "general/real-estate", "medium", "sklearn", "california_housing",
                target="MedHouseVal", from_openfe=True, note="parity anchor"),
    DatasetSpec("microsoft-mslr", "Microsoft (MSLR web-rank)", "regression",
                "general", "large", "openfe_reproduce", "microsoft",
                from_openfe=True, note="learning-to-rank relevance as regression"),
    DatasetSpec("medical", "Medical", "regression", "healthcare", "medium",
                "openfe_reproduce", "medical", from_openfe=True),
    DatasetSpec("diabetes-130us", "Diabetes 130-US (Strack 2014)", "classification",
                "healthcare", "large", "openml", "Diabetes130US", openml_version=1,
                from_openfe=True, aliases=("diabetes130us", "diabetes_130_us"),
                note="OpenML Diabetes130US did=4541 (UCI tabular API unavailable)"),
    DatasetSpec("nomao", "Nomao", "classification", "general", "medium",
                "openml", "nomao", openml_version=1, from_openfe=True),
    DatasetSpec("vehicle-sensit", "Vehicle (SensIT)", "classification",
                "physical-sensor", "medium", "openml", "SensIT-Vehicle-Combined",
                from_openfe=True, aliases=("vehicle_sensIT", "sensit")),
    DatasetSpec("broken-machine", "Broken Machine", "classification", "industrial",
                "large", "openfe_reproduce", "broken_machine", from_openfe=True),
    DatasetSpec("telecom-churn", "Telecom churn", "classification", "finance/telco",
                "medium", "openml", "Telco-Customer-Churn", openml_version=1,
                target="Churn", from_openfe=True, aliases=("telecom", "telco_churn"),
                note="OpenML stand-in; use OpenFE_reproduce for exact split parity"),
    DatasetSpec("jannis", "Jannis", "classification", "general", "medium",
                "openml", "jannis", openml_version=1, from_openfe=True,
                note="AutoML suite dataset (did=41168)"),
    DatasetSpec("covertype", "Covertype", "multiclass", "physical-science", "large",
                "openml", "covertype", openml_version=1, from_openfe=True,
                aliases=("covtype", "forest-covertype"),
                note="OpenML covertype did=150 (UCI tabular API flaky)"),

    # ---- Sector add-ons (draft_plan Sec. 2 coverage + expert-FE ground truth) ----
    DatasetSpec("ieee-cis-fraud", "IEEE-CIS Fraud Detection", "classification",
                "finance", "large", "kaggle_competition", "ieee-fraud-detection",
                target="isFraud", note="OpenFE showcase; winner FE writeups exist"),
    DatasetSpec("bnp-paribas-claims", "BNP Paribas Cardif Claims", "classification",
                "finance", "medium", "kaggle_competition",
                "bnp-paribas-cardif-claims-management", target="target",
                note="OpenFE 2nd showcase"),
    DatasetSpec("home-credit-default", "Home Credit Default Risk", "classification",
                "finance", "large", "kaggle_competition", "home-credit-default-risk",
                target="TARGET", note="expert bureau-aggregation FE (ground truth)"),
    DatasetSpec("german-credit", "German Credit", "classification", "finance", "small",
                "uci", "Statlog (German Credit Data)", aliases=("german", "credit-g")),
    DatasetSpec("heart-disease", "Heart Disease", "classification", "healthcare",
                "small", "uci", "Heart Disease"),
    DatasetSpec("breast-cancer-wisconsin", "Breast Cancer Wisconsin", "classification",
                "healthcare", "small", "sklearn", "breast_cancer", target="target"),
    DatasetSpec("superconductivity", "Superconductivity", "regression",
                "physical/materials", "medium", "uci", "Superconductivty Data",
                note="predict critical temperature", aliases=("superconduct",)),
    DatasetSpec("concrete-strength", "Concrete Compressive Strength", "regression",
                "physical/materials", "small", "uci",
                "Concrete Compressive Strength", aliases=("concrete",)),
    DatasetSpec("qsar-biodegradation", "QSAR Biodegradation", "classification",
                "chemistry", "small", "openml", "qsar-biodeg", openml_version=1,
                note="UCI id=254 lacks tabular API; use OpenML qsar-biodeg",
                aliases=("qsar", "qsar biodegradation")),
    DatasetSpec("house-prices", "House Prices (Adv. Regression)", "regression",
                "general/real-estate", "medium", "kaggle_competition",
                "house-prices-advanced-regression-techniques", target="SalePrice",
                note="canonical expert-FE regression (ground truth)"),
    DatasetSpec("bank-marketing", "Bank Marketing", "classification", "finance",
                "medium", "openml", "bank-marketing", openml_version=1,
                note="GELFE-referenced (OpenML v1 = 45211 rows)"),
    DatasetSpec("electricity", "Electricity", "classification", "energy", "medium",
                "openml", "electricity", openml_version=1,
                note="GELFE failure case -> motivates Stage-2 gatekeeper"),
]


# --- Meta-training corpus sources (fetched then de-duplicated vs BENCHMARK) ---
# Referenced by name; download.py expands each suite to its member datasets.
CORPUS_SUITES: list[dict] = [
    {"source": "openml", "suite": 99, "label": "OpenML-CC18",
     "task": "classification", "note": "72 curated classification datasets"},
    {"source": "openml", "suite": 353, "label": "OpenML-CTR23",
     "task": "regression", "note": "35 curated tabular regression datasets"},
    {"source": "openml", "suite": 271, "label": "AMLB-Classification",
     "task": "classification", "note": "AutoML Benchmark, 71 classification"},
    {"source": "openml", "suite": 269, "label": "AMLB-Regression",
     "task": "regression", "note": "AutoML Benchmark, 33 regression"},
]

# Target ceiling for the corpus (algorithm_plan Sec. 3 -- bounded by Stage-0 RL cost).
CORPUS_MAX_DATASETS = 100


def benchmark_names_for_exclusion() -> set[str]:
    """All names by which a corpus dataset might collide with the benchmark.

    Lower-cased union of internal keys, fetch keys, display names and aliases --
    used by the corpus builder to drop overlaps (e.g. CC18's jannis / nomao /
    covertype / electricity / bank-marketing) before meta-training.
    """
    names: set[str] = set()
    for spec in BENCHMARK:
        names.add(spec.key.lower())
        names.add(spec.fetch_key.lower())
        names.add(spec.display.lower())
        names.update(a.lower() for a in spec.aliases)
    return names
