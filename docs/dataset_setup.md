# Dataset Setup & Reproduction Guide

How to stand up the MF-OpenFE **benchmark suite** and **meta-training corpus** on a
fresh machine. Follow top to bottom; the whole thing (minus the large Kaggle
downloads) takes a few minutes.

- Code: `afe/` package.
- Frozen split (version-controlled): `afe/benchmark/manifests/{benchmark,corpus}.json`.
- Downloaded data (gitignored, machine-local): `data/cache/`.

See `algorithm_plan.md` / `draft_plan.md` for *why* these datasets; this doc is the *how*.

---

## 1. Prerequisites
- **Python 3.12** (developed on 3.12.13). 3.10+ should work.
- **git**, and ~5–10 GB free disk if you pull the large Kaggle competitions.
- Network access to `openml.org`, `archive.ics.uci.edu`, `kaggle.com`.

## 2. Clone & install
```bash
git clone <your-repo-url> auto-feature-engineering
cd auto-feature-engineering

python -m venv .venv
. .venv/bin/activate            # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e .      # makes `afe`/`afe.meta` importable from anywhere
```

`requirements.txt` pins: pandas, numpy, pyarrow (parquet cache), scikit-learn,
openml, ucimlrepo, kaggle, lightgbm (parity baseline), pytest.

## 3. Repo layout
```
afe/
  meta/online.py    # MFOpenFE -- the public entrypoint, unrelated to dataset sourcing
  benchmark/
    registry.py     # 22 benchmark DatasetSpecs + CORPUS_SUITES (source of truth)
    download.py     # load(spec) -> (DataFrame, meta); caches to data/cache/*.parquet
    manifests.py    # builds afe/benchmark/manifests/{benchmark,corpus}.json (disjoint split)
    manifests/      # the frozen split (committed)
scripts/            # production entrypoints (run_benchmark.py, run_stage0.py, ...)
dev/
  smoke_download.py  # fetch every benchmark dataset, print a metadata row
  parity_check.py    # raw-feature LightGBM baseline (OpenFE Table-3 sanity)
tests/
  test_disjoint.py   # benchmark ∩ corpus == ∅  (offline)
  test_coverage.py   # task × sector × scale coverage  (offline)
conftest.py          # puts repo root on sys.path so `import afe` works
data/cache/          # gitignored download cache (created on first fetch)
```

## 4. Credentials — Kaggle (only for the 4 competition datasets)
IEEE-CIS Fraud, BNP Paribas Claims, Home Credit, House Prices come from Kaggle.

1. Kaggle → **Account → API → Create New API Token** → downloads `kaggle.json`.
2. Place it and lock permissions:
   ```bash
   mkdir -p ~/.kaggle && mv ~/Downloads/kaggle.json ~/.kaggle/
   chmod 600 ~/.kaggle/kaggle.json
   ```
3. **Accept each competition's rules once** in the browser (visit the competition
   page → *Rules* → *I Understand and Accept*), or downloads 403.

> The code never reads a raw token from the environment or arguments — Kaggle
> credentials stay in your local `~/.kaggle/kaggle.json`. Do not paste tokens into
> chats/tickets; if you have, rotate them.

## 5. Fetch the data
```bash
# All auto (no-auth) benchmark datasets + the Kaggle ones (if creds set):
python -m dev.smoke_download

# Or a subset by key:
python -m dev.smoke_download nomao california-housing german-credit
```
Each dataset is fetched once and cached to `data/cache/<key>.parquet` (+ `.meta.json`).
Re-runs read the cache. Delete a `.parquet`+`.meta.json` pair to force a refetch.

### 5a. Three datasets need a manual drop
`microsoft-mslr`, `medical`, `broken-machine` have no clean canonical source. They ship
as RTDL-style `.npy` dumps in a `data.zip` linked from **`ZhangTP1996/OpenFE_reproduce`**
(the mirror the main OpenFE repo's README points to under "Data Download" -> Part 2 --
not `IIIS-Li-Group/OpenFE_reproduce`, which doesn't host the data). Download it, unzip,
and drop each dataset's folder at:
```
data/cache/raw/<key>/            # <key> = microsoft-mslr | medical | broken-machine
  N_train.npy  N_val.npy  N_test.npy   # numeric features (required)
  C_train.npy  C_val.npy  C_test.npy   # categorical features (optional)
  y_train.npy  y_val.npy  y_test.npy   # target (required)
```
(the zip's internal folder names are `microsoft`, `medical`, `broken_machine` --
rename to the `<key>` above when copying). `download.py` concatenates all 3 source
splits into one frame; the source's own train/val/test split isn't preserved --
`afe.benchmark.splits` freezes its own fixed split for every dataset uniformly.

## 6. Build the frozen split (benchmark vs. corpus)
```bash
python -m afe.benchmark.manifests
```
Writes `afe/benchmark/manifests/benchmark.json` (22 datasets) and `corpus.json` (~100 OpenML
datasets: OpenML-CC18 classification + OpenML-CTR23 regression, capped at
`CORPUS_MAX_DATASETS=100`). The builder **removes every benchmark dataset from the
corpus** (hard disjointness rule, `algorithm_plan.md` §3) — e.g. it drops CC18's
`jannis`, `nomao`, `covertype`, `electricity`, `bank-marketing`. Needs the `openml`
package + network; without them it degrades to per-suite stubs instead of crashing.

Commit the regenerated manifests so every machine trains/evaluates on the same split.

## 7. Verify
```bash
pytest tests/ -q                 # 9 tests, offline, ~instant
python -m dev.parity_check       # raw-feature LightGBM baseline
```
Expected `parity_check` output (single split, default params — sanity, not a full run):
```
california-housing     regression  R2 = 0.848
nomao                  binary-clf  AUC = 0.996     # matches OpenFE Table-3 "Base"
concrete-strength      regression  R2 = 0.938
```
If Nomao AUC lands near 0.996 and California Housing R² near 0.85, the load/split path
is correct and you're ready to build AutoFE on top.

## 8. The benchmark suite (what gets downloaded)
`OpenFE?` marks the 10 reused from OpenFE (kept for paper parity). Addressed by source
*name/slug* — OpenML versions pinned so the table is deterministic.

| key | task | sector | scale | source | fetch key / slug | OpenFE |
|---|---|---|---|---|---|:--:|
| california-housing | reg | general/real-estate | med | sklearn | california_housing | ✓ |
| microsoft-mslr | reg | general | large | manual | OpenFE_reproduce | ✓ |
| medical | reg | healthcare | med | manual | OpenFE_reproduce | ✓ |
| diabetes-130us | clf | healthcare | large | openml | Diabetes130US (v1, did 4541) | ✓ |
| nomao | clf | general | med | openml | nomao (v1) | ✓ |
| vehicle-sensit | clf | physical-sensor | med | openml | SensIT-Vehicle-Combined | ✓ |
| broken-machine | clf | industrial | large | manual | OpenFE_reproduce | ✓ |
| telecom-churn | clf | finance/telco | med | openml | Telco-Customer-Churn (v1), target=Churn | ✓ |
| jannis | clf | general | med | openml | jannis (v1, did 41168) | ✓ |
| covertype | multiclass | physical-science | large | openml | covertype (v1, did 150) | ✓ |
| ieee-cis-fraud | clf | finance | large | kaggle | ieee-fraud-detection | |
| bnp-paribas-claims | clf | finance | med | kaggle | bnp-paribas-cardif-claims-management | |
| home-credit-default | clf | finance | large | kaggle | home-credit-default-risk | |
| german-credit | clf | finance | small | uci | Statlog (German Credit Data) | |
| heart-disease | clf | healthcare | small | uci | Heart Disease | |
| breast-cancer-wisconsin | clf | healthcare | small | sklearn | breast_cancer | |
| superconductivity | reg | physical/materials | med | uci | Superconductivty Data | |
| concrete-strength | reg | physical/materials | small | uci | Concrete Compressive Strength | |
| qsar-biodegradation | clf | chemistry | small | openml | qsar-biodeg (v1) | |
| house-prices | reg | general/real-estate | med | kaggle | house-prices-advanced-regression-techniques, target=SalePrice | |
| bank-marketing | clf | finance | med | openml | bank-marketing (v1) | |
| electricity | clf | energy | med | openml | electricity (v1) | |

## 9. Adding a dataset
Append a `DatasetSpec(...)` to `BENCHMARK` in `afe/benchmark/registry.py`. Required: `key`,
`display`, `task`, `sector`, `scale`, `source`, `fetch_key`. Set `target` when the
source doesn't name it (always for Kaggle). Set `openml_version` if OpenML lists >1
active version. Add `aliases` for any other names by which it might appear in a corpus
suite (so the disjointness filter catches it). Then re-run §6 and §7.

## 10. Troubleshooting
- **`Multiple active versions ... returning version 1`** — pin `openml_version=` on that
  spec (already done for the built-in ones).
- **`Sparse ARFF datasets cannot be loaded with as_frame=True`** — handled: `download.py`
  refetches raw and densifies (e.g. vehicle-sensit).
- **UCI `DatasetNotFoundError: ... exists ... but is not available for import`** — that
  UCI entry has no tabular API; route the spec to `openml` instead (done for QSAR, and
  why covertype/diabetes-130us use OpenML).
- **Kaggle `403`** — you haven't accepted that competition's rules (§4.3).
- **`import afe` fails in a script** — run from the repo root with the venv active (or
  `pip install -e .` once), then use `python -m scripts.<name>` / `python -m dev.<name>`
  / `python -m afe.<name>`.
- **Reproducibility** — OpenML versions are pinned and the split is frozen in
  `afe/benchmark/manifests/`; keep those committed and everyone stays in sync.
