# `afe` — dataset sourcing

Implements the approved dataset plan: a **benchmark** evaluation suite and a
disjoint **meta-training corpus**. (The plan sketched this under a `data/`
package; `data/` is gitignored for the dataset cache, so the code lives here in
`afe/` and downloaded data goes to the repo-root `data/cache/`.)

## Layout
- `registry.py` — declarative `BENCHMARK` (22 datasets) + `CORPUS_SUITES`. Datasets
  addressed by source *name/slug*, not numeric id; OpenML versions pinned where >1 active.
- `download.py` — `load(spec)` fetches + caches one dataset to `data/cache/*.parquet`,
  returns `(frame, meta)`. Lazy imports, so metadata-only use needs no heavy deps.
- `manifests.py` — `python -m afe.manifests` (re)builds `afe/manifests/{benchmark,corpus}.json`,
  removing every benchmark dataset from the corpus (hard disjointness rule).
- `splits.py` / `encoders.py` / `eval_data.py` — frozen per-dataset single
  fixed-seed train/test split, per-model-family encoders, and the `iter_folds()`
  fold-iteration contract.
- `methods.py` / `models.py` / `benchmark.py` — AutoFE method adapters (baseline, OpenFE,
  Featuretools, Autofeat), the 3-model panel, and the budget-limited benchmark runner.
  Full flow: **`docs/benchmark_guide.md`**.
- `meta/` — MF-OpenFE's offline meta-learning pipeline (`algorithm_plan.md`):
  Stage 0 RL label generation (Double DQN over a Feature Transformation Graph)
  → Stage 1 per-operator meta-model. Runs on the disjoint corpus, never at
  online usage time. See **`afe/meta/README.md`**.

## Setup
```bash
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
pip install -e .                       # makes `afe`/`afe.meta` importable from anywhere
```

## Verify
```bash
pytest tests/ -q                       # disjointness + coverage (offline, no network)
python -m dev.smoke_download           # fetch every benchmark dataset, print metadata
python -m dev.parity_check             # raw-feature LightGBM baseline (OpenFE Table-3 sanity)
python -m afe.manifests                # rebuild the frozen split
```

## Production vs. dev
- `scripts/` — production entrypoints (`run_benchmark.py`, `report_benchmark.py`,
  `run_stage0.py`, `train_meta_model.py`, `benchmark_ctl.sh`). See `scripts/README.md`.
- `dev/` — one-off smoke/sanity utilities (`smoke_download.py`, `parity_check.py`),
  not part of the production pipeline. See `dev/README.md`.

## Fetch status
- **Auto (no auth), verified:** california-housing, breast-cancer-wisconsin, nomao,
  vehicle-sensit, jannis, telecom-churn, electricity, bank-marketing, german-credit,
  heart-disease, concrete-strength, superconductivity, qsar-biodegradation. Names confirmed
  via OpenML metadata for covertype, diabetes-130us (same OpenML path, high confidence).
- **Needs Kaggle auth** (`~/.kaggle/kaggle.json` + accept each competition's rules once):
  ieee-cis-fraud, bnp-paribas-claims, home-credit-default, house-prices. Set an explicit
  `target` in the registry (already done for house-prices).
- **Manual CSV drop** into `data/cache/raw/<key>/` (from `IIIS-Li-Group/OpenFE_reproduce`):
  microsoft-mslr, medical, broken-machine. These 3 have no clean canonical source; the
  reproduce mirror also guarantees split parity for the OpenFE-comparable subset.
