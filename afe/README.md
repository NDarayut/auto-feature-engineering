# `afe` — the MF-OpenFE library

Top-level package. See the repo-root [`README.md`](../README.md) for the
quickstart (`from afe import MFOpenFE`). This doc covers the package's own
internal layout.

## Layout
- `methods.py` — AutoFE method adapters (baseline, OpenFE, Featuretools,
  Autofeat) sharing a common `fit_transform`/`transform` contract, plus
  `prep_for_generation()` (ordinal-encode categoricals + median-impute,
  fit on train only). Used by both `afe.benchmark` (comparing all 4) and
  `afe.meta.online.MFOpenFE` (data prep only).
- `encoders.py` — `TreeEncoder`/`LinearEncoder`, per-model-family encoding
  fit on the training fold only.
- `progress.py` — `ProgressReporter`: stage markers + tqdm bars, used by
  `MFOpenFE` (`progress=True`).
- `meta/` — MF-OpenFE itself:
  - **Online** (`online.py`) — `MFOpenFE`, the public per-dataset entrypoint
    (algorithm_plan.md Stages 3-6): generate candidates with OpenFE's own
    operator library, meta-filter, verify + select.
  - **Offline** (`stage0.py`/`stage1.py`/`environment.py`/`ddqn.py`/
    `operators.py`/`meta_features.py`) — the one-time training pipeline that
    produces `models/meta_model.pkl`, the artifact `online.py` loads. See
    **`afe/meta/README.md`**.
- `benchmark/` — the comparison/research harness (frozen dataset registry,
  download/cache, split protocol, per-fold encoding, the 3-model scoring
  panel, and the budget-limited benchmark runner). Not needed to run
  `MFOpenFE`; see **`docs/benchmark_guide.md`**.

## Setup
```bash
python -m venv .venv && . .venv/bin/activate
pip install -e .                       # core: just enough to run MFOpenFE
pip install -e ".[benchmark]"          # + the comparison harness's deps
```

## Verify
```bash
pytest tests/ -q                       # 51 tests, offline, no network
python -m dev.smoke_download           # fetch every benchmark dataset, print metadata
python -m dev.parity_check             # raw-feature LightGBM baseline (OpenFE Table-3 sanity)
python -m afe.benchmark.manifests      # rebuild the frozen benchmark/corpus split
```

## Production vs. dev
- `scripts/` — production entrypoints (`run_benchmark.py`, `report_benchmark.py`,
  `run_stage0.py`, `train_meta_model.py`, `benchmark_ctl.sh`). See `scripts/README.md`.
- `dev/` — one-off smoke/sanity utilities (`smoke_download.py`, `parity_check.py`),
  not part of the production pipeline. See `dev/README.md`.

## Benchmark dataset fetch status
(Only relevant if you're running `afe.benchmark` — irrelevant to `MFOpenFE` itself.)
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
