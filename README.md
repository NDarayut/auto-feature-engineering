# MF-OpenFE: benchmarking AutoFE, and a meta-filtered variant of OpenFE

Two things live in this repo:

1. **A benchmark suite** (`draft_plan.md`) that compares automatic
   feature-engineering (AutoFE) methods — OpenFE, Featuretools, Autofeat,
   and a no-op baseline — fairly: same split, same model panel, same
   compute budget, per dataset.
2. **MF-OpenFE** (`algorithm_plan.md`), a new AutoFE method under
   development: OpenFE's own candidate generation, filtered by a small
   meta-model trained offline (once, on ~100 historical datasets) to predict
   which candidates are worth evaluating — so the expensive evaluation step
   runs on a pre-vetted pool instead of everything OpenFE generates.

The benchmark and the meta-training corpus are **disjoint by construction**
(`afe/manifests.py`) — MF-OpenFE is never trained on a dataset it's later
graded on.

## Quickstart

```bash
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
pip install -e .          # makes `afe` / `afe.meta` importable from anywhere
pytest tests/ -q          # 46 tests, offline, no network
```

```python
from afe import iter_folds, run_benchmark, BENCHMARK
from afe.meta import MetaModel, train_meta_model

# leak-free train/test data for any benchmark dataset, ready for a model
fold = next(iter_folds("german-credit", encoding="tree"))

# a trained MF-OpenFE meta-model (after running the offline pipeline below)
model = MetaModel.load("results/meta/meta_model.pkl")
```

Fetching real data (OpenML/UCI/sklearn/Kaggle) needs network access and, for
4 datasets, Kaggle credentials — see `docs/dataset_setup.md`.

## Repo layout

```
afe/            production library (pip-installable, `import afe`)
  meta/           MF-OpenFE's offline meta-learning pipeline (Stage 0-1)
  manifests/      frozen benchmark/corpus/split manifests (version-controlled)
scripts/        production CLI entrypoints — see scripts/README.md
dev/            one-off smoke/sanity utilities, not production — see dev/README.md
tests/          pytest suite, offline/no-network
docs/           how-to guides (setup, running the benchmark, ctl script usage)
research/       source papers backing algorithm_plan.md's design choices
data/, results/ gitignored: dataset cache, benchmark/meta-model outputs
```

- **`afe/README.md`** — the library's own layout + verify/fetch-status reference.
- **`afe/meta/README.md`** — the MF-OpenFE offline pipeline (Stage 0 RL search
  → Stage 1 meta-model), how to run it, and the design rationale.
- **`docs/benchmark_guide.md`** — full benchmark data flow, dataset → report row.
- **`docs/dataset_setup.md`** — standing up the benchmark suite + corpus from scratch.
- **`docs/benchmark_ctl_usage.md`** — running a long benchmark in the background.
- **`draft_plan.md`** — the approved benchmark methodology (splits, encoding,
  budget, model panel, reporting).
- **`algorithm_plan.md`** — MF-OpenFE's design, staged offline (RL-allowed) vs.
  online (deterministic, fast) — every technique attributed to its source paper.
- **`synthesis_report.md`** — literature synthesis behind `algorithm_plan.md`.

## Running things

```bash
# Benchmark: baseline + AutoFE methods vs. the 3-model panel, one split each
python -m scripts.run_benchmark --datasets german-credit concrete-strength \
    --methods baseline openfe --models tree linear knn --budget 300
python -m scripts.report_benchmark results/benchmark_results.jsonl

# MF-OpenFE offline pipeline: RL label generation, then train the meta-model
python -m scripts.run_stage0 --episodes 80
python -m scripts.train_meta_model
```

`scripts/benchmark_ctl.sh` wraps the benchmark run for long background
sessions (start/status/tail/stop/report) — see `docs/benchmark_ctl_usage.md`.

## Status

- Benchmark harness: implemented, sequential/single-split, tested end-to-end.
- MF-OpenFE Stage 0 (RL label generation) + Stage 1 (meta-model training):
  implemented and run on the full ~100-dataset corpus — leave-datasets-out
  AUC 0.80 predicting feature usefulness on unseen datasets.
- Not yet built: Stage 2 (gatekeeper), Stages 3-6 (the online meta-filter
  wired into OpenFE), and MF-OpenFE's entry in the benchmark itself.
