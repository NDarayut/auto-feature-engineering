# AutoFE Benchmark

A modular, algorithm-agnostic benchmark harness for comparing automatic
feature engineering (AutoFE) methods fairly: every method runs on identical
frozen train/test splits, receives identically preprocessed data, and is
scored by the same downstream model panel. Failures (timeouts, crashes,
out-of-memory) are recorded as result rows, never silently dropped.

## Contents

- [Installation](#installation)
- [Usage](#usage)
  - [Command line](#command-line)
  - [Python API](#python-api)
  - [Benchmarking your own method](#benchmarking-your-own-method)
- [Parameters](#parameters)
- [Available methods](#available-methods)
- [Available datasets](#available-datasets)
- [Output](#output)

## Installation

```bash
git clone <this-repo> && cd auto-feature-engineering
python -m venv .venv && . .venv/bin/activate
pip install -e ".[benchmark]"      # harness + dataset fetchers (openml, ucimlrepo, kaggle, ...)
pip install -e ".[benchmark,test]" # + pytest
```

Python 3.10+ (developed on 3.12). Most datasets download automatically on
first use and are cached under `data/cache/`; a few need Kaggle credentials
or a one-time manual drop — see [`docs/dataset_setup.md`](docs/dataset_setup.md).

## Usage

### Command line

```bash
# Compare methods on chosen datasets, 900s generation budget per (dataset, method):
python -m scripts.run_benchmark \
    --datasets german-credit concrete-strength house-prices \
    --methods baseline openfe cafem \
    --budget 900 --out results/my_run.jsonl

# Everything: all 22 datasets (smallest first), resumable if interrupted:
python -m scripts.run_benchmark --methods baseline openfe cafem featuretools autofeat

# Aggregate a results file into a per-task/sector report:
python -m scripts.report_benchmark results/my_run.jsonl
```

Runs are **resumable**: (dataset, method) pairs already present in `--out`
are skipped on restart, so an interrupted sweep continues where it left off.
For long unattended runs there is a supervised background runner —
[`docs/benchmark_ctl_usage.md`](docs/benchmark_ctl_usage.md).

### Python API

```python
from afe.benchmark import compare, BaselineMethod, OpenFEMethod, CAFEMMethod

result = compare(
    methods=[BaselineMethod, OpenFEMethod, CAFEMMethod],
    datasets=["german-credit", "concrete-strength"],   # built-in suite keys
)
print(result)          # per-model-family markdown score table
df = result.to_frame() # raw rows as a pandas DataFrame
```

### Benchmarking your own method

A method is either a **plain function**:

```python
def my_method(X_train, y_train, X_test, task):
    ...                                   # task is "classification" | "regression"
    return X_train_new, X_test_new        # original + engineered columns
```

or a **class** with the `fit_transform`/`transform` contract (a structural
protocol — no inheritance needed):

```python
class MyMethod:
    name = "my-method"                    # display name in result tables

    def fit_transform(self, X_train, y_train, task): ...  # only ever sees train
    def transform(self, X_test): ...                      # replays fitted state
```

Then:

```python
from afe.benchmark import compare

result = compare(
    methods=[my_method, MyMethod, ("my-tuned-variant", MyMethod())],
    datasets=["german-credit"],               # built-in suite, and/or:
    custom_datasets={"my-data": (X, y)},      # your own (X, y) data
    out_path="results/my_run.jsonl",          # optional: persist + resume
    budget_seconds=600,                       # optional: subprocess isolation + hard timeout
)
```

Both dataset sources can be mixed in one call. Full API reference:
[`afe/benchmark/README.md`](afe/benchmark/README.md).

### How a run works

For each dataset, the harness:

1. computes a single **frozen 80/20 train/test split** (seeded
   deterministically from the dataset key — same split on every machine, in
   every run, for every method);
2. applies **uniform preprocessing** (categoricals ordinal-encoded, numerics
   median-imputed, fit on the training fold only) so every method receives
   the identical numeric, NaN-free matrix;
3. runs each method's **feature generation in an isolated subprocess** under
   a wall-clock budget and a hard memory cap — a method that runs too long,
   crashes, or exhausts memory becomes a `timeout`/`crashed`/`oom` result
   row while the sweep continues;
4. scores the engineered features with a **3-family model panel** — LightGBM
   (`tree`), Logistic Regression / Ridge (`linear`), and kNN (`knn`) — since
   a feature set that helps all three families is genuinely better
   representation, not a model-specific artifact;
5. writes **one JSONL row per (dataset, method, model family)** as it is
   produced.

Metrics: ROC AUC for binary classification, macro one-vs-rest ROC AUC for
multiclass, R² (plus MAE) for regression.

## Parameters

All flags of `python -m scripts.run_benchmark`:

| flag | default | meaning |
|---|---|---|
| `--datasets` | all 22, smallest-scale-first | benchmark keys to run — space-separated, e.g. `--datasets nomao covertype` (see [Available datasets](#available-datasets)) |
| `--methods` | `baseline` | which methods to benchmark — any of `baseline openfe cafem featuretools autofeat` |
| `--models` | all three | which model families to score generated features with — any of `tree linear knn` |
| `--budget` | `300` | per (dataset, method) **generation** time budget in seconds; a method exceeding it is killed and recorded as `status="timeout"` |
| `--out` | `results/benchmark_results.jsonl` | output JSONL path |
| `--no-resume` | off | re-run pairs even if already present in `--out` (default is to skip pairs already completed there) |
| `--max-cols` | `200` | cap a dataset to its N most target-associated columns before any method runs — a generic guard against any method's candidate-generation cost blowing up on very wide datasets; `0` disables |
| `--fit-sample-rows` | `20000` | cap the row count a method's *fit* step sees to a random sample of this size (the full fold is still used for scoring — see `--transform-chunk-rows`); `0` disables |
| `--transform-chunk-rows` | `20000` | apply a fitted method's *transform* in row chunks of this size instead of on the whole fold at once, bounding peak memory while building the final feature matrix; `0` disables |
| `--max-mem-gb` | `16.0` | hard memory ceiling (via `RLIMIT_AS`) for each method's generation subprocess, as a safety net for whatever the caps above don't catch — an over-limit run is killed and recorded as `status="oom"` rather than risking the host; `0` disables |

The last four flags exist because a method's internal algorithm (candidate
search, RL rollout, DFS, …) can scale combinatorially with a dataset's row
or column count — they apply identically regardless of which `--methods` you
pick, so a large/wide dataset that would otherwise run out of memory now
degrades gracefully to a smaller effective sample and a bounded-memory
failure instead of crashing the machine. Defaults are tuned for a large-ish
workstation (dozens of GB free); lower `--max-mem-gb` (and/or
`--fit-sample-rows`) on a smaller machine, or raise them if you want
closer-to-full-data runs and have the memory to spare.

## Available methods

| method | class | what it does |
|---|---|---|
| `baseline` | `BaselineMethod` | No feature engineering — the raw (prepped) features. Every comparison should include it. |
| `openfe` | `OpenFEMethod` | [OpenFE](https://github.com/IIIS-Li-Group/OpenFE) (NeurIPS 2022): full candidate expansion + two-stage boosting-based evaluation; keeps the top `n_new_features` (default 10). |
| `cafem` | `CAFEMMethod` | CAFEM-style (PAKDD 2020) per-dataset RL: a Double DQN searches the Feature Transformation Graph on the training fold; the best-scoring transformation chain per feature is kept and replayed on test. |
| `featuretools` | `FeaturetoolsMethod` | Single-table Deep Feature Synthesis (pairwise arithmetic primitives). |
| `autofeat` | `AutofeatMethod` | Autofeat's iterative non-linear expansion + L1-based selection. |

All are re-exported from `afe.benchmark` and follow the same
`fit_transform`/`transform` contract, so your own method plugs in alongside
them (see [Benchmarking your own method](#benchmarking-your-own-method)).

## Available datasets

The frozen suite has **22 datasets** covering both task types, small through
large scale, and a spread of sectors. The registry
(`afe/benchmark/registry.py`) is the source of truth; download instructions:
[`docs/dataset_setup.md`](docs/dataset_setup.md). The metric is chosen by
the harness from the data: `r2` (+ `mae`) for regression, `auc` for binary
classification, macro one-vs-rest `auc_ovr` when there are more than two
classes.

| key | task | metric | rows | features | classes |
|---|---|---|---:|---:|---:|
| california-housing | regression | r2 | 20 640 | 8 | — |
| microsoft-mslr | regression | r2 | 723 412 | 136 | — |
| medical | regression | r2 | 104 361 | 5 | — |
| superconductivity | regression | r2 | 21 263 | 81 | — |
| concrete-strength | regression | r2 | 1 030 | 8 | — |
| house-prices | regression | r2 | 1 460 | 80 | — |
| diabetes-130us | classification | auc_ovr | 101 766 | 49 | 3 |
| nomao | classification | auc | 34 465 | 118 | 2 |
| vehicle-sensit | classification | auc_ovr | 98 528 | 100 | 3 |
| broken-machine | classification | auc | 576 000 | 58 | 2 |
| telecom-churn | classification | auc | 7 043 | 19 | 2 |
| jannis | classification | auc_ovr | 83 733 | 54 | 4 |
| covertype | multiclass | auc_ovr | 110 393 | 54 | 7 |
| ieee-cis-fraud | classification | auc | 590 540 | 393 | 2 |
| bnp-paribas-claims | classification | auc | 114 321 | 132 | 2 |
| home-credit-default | classification | auc | 307 511 | 121 | 2 |
| german-credit | classification | auc | 1 000 | 20 | 2 |
| heart-disease | classification | auc_ovr | 303 | 13 | 5 |
| breast-cancer-wisconsin | classification | auc | 569 | 30 | 2 |
| qsar-biodegradation | classification | auc | 1 055 | 41 | 2 |
| bank-marketing | classification | auc | 45 211 | 16 | 2 |
| electricity | classification | auc | 45 312 | 8 | 2 |

Row/feature/class counts are measured from the cached tables (features =
columns excluding the target).

Splits and OpenML versions are pinned in committed manifests
(`afe/benchmark/manifests/`), so every machine evaluates the same tables on
the same splits.

## Output

`--out` is a JSONL file, one row per (dataset, method, model family),
flushed as each row is produced. A successful row looks like:

```json
{"key": "nomao", "method": "cafem", "fold_id": "split0", "protocol": "holdout", "task": "classification", "status": "ok", "gen_elapsed_s": 11.10, "fit_elapsed_s": 11.01, "transform_elapsed_s": 0.09, "peak_mem_mb": 564.7, "n_candidates": 8, "feature_efficiency": 1.0, "model_family": "tree", "n_features_generated": 8, "n_features_final": 126, "metric": "auc", "value": 0.9941}
```

A failed/killed run still gets a row (`model_family`/`metric`/`value` are
`null`) instead of being silently dropped:

```json
{"key": "ieee-cis-fraud", "method": "openfe", "fold_id": "split0", "protocol": "holdout", "task": "classification", "status": "oom", "gen_elapsed_s": null, "model_family": null, "metric": null, "value": null, "error": "MemoryError under 16.0 GB RLIMIT_AS cap"}
```

### Row fields

| field | meaning |
|---|---|
| `key` | dataset key (see [Available datasets](#available-datasets)) |
| `method` | method name (`baseline`, `openfe`, `cafem`, …) |
| `fold_id`, `protocol` | which frozen split the row was scored on (`split0`, `holdout`) |
| `task` | `classification`, `multiclass`, or `regression` |
| `status` | `ok`, `timeout` (exceeded `--budget`), `oom` (exceeded `--max-mem-gb`), `crashed` (subprocess died), `error` (method raised), or `model_error` (scoring model raised) |
| `metric`, `value` | primary metric name and held-out score — `auc`, `auc_ovr`, or `r2` |
| `model_family` | which panel model produced the score — `tree`, `linear`, or `knn` |
| `gen_elapsed_s`, `fit_elapsed_s`, `transform_elapsed_s` | total generation wall-time, and its fit / transform components |
| `peak_mem_mb` | peak memory of the generation subprocess |
| `n_candidates` | features surviving the method's *own* internal selection |
| `n_features_generated`, `n_features_final` | new features actually added / total feature count after generation |
| `feature_efficiency` | fraction of newly generated features that are individually predictive (univariate target association ≥ 0.1 on the train fold) — "many features that are also *good*", not just many features |
| `error` | error message for failed rows, `null` otherwise |

### Working with results

```python
import pandas as pd

df = pd.read_json("results/my_run.jsonl", lines=True)
df.groupby(["method", "model_family"])["value"].mean()
```

Or generate the markdown report:

```bash
python -m scripts.report_benchmark results/my_run.jsonl          # to stdout
python -m scripts.report_benchmark results/my_run.jsonl --out docs/benchmark_report.md
```

The report has four tables (excerpt — full generated example:
[`docs/benchmark_report.md`](docs/benchmark_report.md)):

**Datasets legend** — maps the abbreviated column labels used by every other
table to the full dataset key, its task, and its metric:


| abbrev | dataset | task | metric |
|---|---|---|---|
| CH | california-housing | regression | r2 |
| GC | german-credit | classification | auc |


**Overview** — one row per method (baseline first), one column per dataset;
the cell is the mean score across the three model families (best per column
in bold):


| method | CH | GC | ... |
|---|---|---|---|
| baseline | 0.715 | **0.774** | ... |
| openfe | **0.767** | 0.771 | ... |


**Per-method scores** — the breakdown behind the overview: each method's
three model-family rows, per dataset:


| method | model | CH | GC | ... |
|---|---|---|---|---|
| baseline | knn | 0.694 | 0.756 | ... |
|  | linear | 0.606 | 0.787 | ... |
|  | tree | 0.844 | 0.780 | ... |
| openfe | knn | 0.779 | 0.758 | ... |
|  | linear | 0.675 | 0.788 | ... |
|  | tree | 0.848 | 0.767 | ... |


**Speed** — feature-generation wall-time per (method, dataset), with a
per-method median:


| method | CH | GC | ... | median |
|---|---|---|---|---|
| baseline | 0.0 s | 0.0 s | ... | 0.0 s |
| openfe | 35.5 s | 60.1 s | ... | 42.0 s |


A final **Failures / timeouts / crashes** table counts every non-`ok` row
per (dataset, method, status). Scores are fold-means when the results file
contains multiple folds per (dataset, method, model family).
