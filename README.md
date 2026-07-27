# AutoFE Benchmark

A modular, algorithm-agnostic benchmark harness for comparing automatic
feature engineering (AutoFE) methods fairly: every method runs on identical
frozen train/test splits, receives identically preprocessed data, and is
scored by the same downstream model panel. Failures (timeouts, crashes,
out-of-memory) are recorded as result rows, never silently dropped.

**No AutoFE library is bundled.** The harness ships exactly one method — the
no-op `baseline` every comparison is read against. Methods under test are
yours: your own algorithm, or a third-party one you install and adapt in a
few lines. That keeps the benchmark neutral and its dependency surface small.

## Usage

### Install

```bash
git clone https://github.com/NDarayut/auto-feature-engineering.git
cd auto-feature-engineering
python -m venv .venv && . .venv/bin/activate
pip install -e .                  # core: harness + your own methods and data
pip install -e ".[datasets]"      # + fetchers for the built-in 22-dataset suite
pip install -e ".[datasets,test]" # + pytest
```

Python 3.10+ (developed on 3.12). To use it from another project:

```bash
pip install "auto-feature-engineering[datasets] @ git+https://github.com/NDarayut/auto-feature-engineering.git"
```

### Start here: which entry point?

There are two, and they share the same core — identical preprocessing,
identical frozen split, identical 3-model scoring panel, identical output
rows. They differ only in where the data comes from and how much the run is
guarded.

**`compare()` — the Python API. Start here.** For your own data, or a quick
check against a few built-in datasets. This example is complete and runnable:

```python
import numpy as np, pandas as pd
from afe.benchmark import compare, BaselineMethod

# Toy data whose signal is an interaction: y = 1 when a*b > 0.
rng = np.random.RandomState(0)
X = pd.DataFrame({"a": rng.randn(800), "b": rng.randn(800)})
y = pd.Series(((X.a * X.b) > 0).astype(int))

# A "method" is just a function that returns the two frames, with new columns.
def add_interaction(X_train, y_train, X_test, task):
    f = lambda d: d.assign(a_times_b=d.a * d.b)
    return f(X_train), f(X_test)

print(compare(methods=[BaselineMethod, add_interaction],
              custom_datasets={"demo": (X, y)}))
```

## linear
| dataset | add_interaction | baseline |
|---|---|---|
| demo    | 1.000           | 0.540    |

## tree
| dataset | add_interaction | baseline |
|---|---|---|
| demo    | 1.000           | 1.000    |

## knn
| dataset | add_interaction | baseline |
|---|---|---|
| demo    | 0.999           | 0.995    |


Read that as: the engineered feature is *essential* for the linear model
(0.540 — chance — up to 1.000) and irrelevant to the tree, which already
learns interactions on its own. That split is the whole reason three model
families are scored: a feature set that only helps one family is a
model-specific patch, not better representation.

**`scripts.run_benchmark` — the CLI.** For the full 22-dataset suite: long,
resumable, resource-guarded runs that write JSONL for later reporting.

```bash
python -m scripts.run_benchmark --methods baseline mypkg:MyMethod \
    --datasets german-credit concrete-strength --out results/run.jsonl
python -m scripts.report_benchmark results/run.jsonl --out report.md
```

| | `compare()` | `run_benchmark` CLI |
|---|---|---|
| **your own `(X, y)` data** | yes, via `custom_datasets=` | no — built-in keys only |
| **built-in 22 datasets** | yes, via `datasets=` | yes |
| **how you pass a method** | the object/function itself | an import path string |
| **result** | a `CompareResult` you can print | a JSONL file |
| **timeout per method** | off unless you set `budget_seconds=` | on (`--budget`, default 300s) |
| **memory/width guards** | **none** | `--max-cols`, `--fit-sample-rows`, `--transform-chunk-rows`, `--max-mem-gb` |
| **resumable** | with `out_path=` | yes |

> **Use the CLI for the large datasets.** `compare()` applies no width cap,
> row sampling, or memory ceiling, so `compare(datasets=["ieee-cis-fraud"])`
> (590k rows × 393 columns) can exhaust your machine's RAM. The CLI's guards
> are on by default and degrade gracefully instead.

The rest of this section covers both in more detail.

### Command line

Reference your method by import path; `baseline` is the one built-in name.

```bash
# Compare your method against the baseline, 900s budget per (dataset, method):
python -m scripts.run_benchmark \
    --datasets german-credit concrete-strength house-prices \
    --methods baseline mypkg.methods:MyMethod \
    --budget 900 --out results/my_run.jsonl

# Everything: all 22 datasets (smallest first), resumable if interrupted:
python -m scripts.run_benchmark --methods baseline mypkg.methods:MyMethod

# Aggregate a results file into a per-task/sector report:
python -m scripts.report_benchmark results/my_run.jsonl
```

Runs are **resumable**: (dataset, method) pairs already present in `--out`
are skipped on restart, so an interrupted sweep continues where it left off.

### Python API

```python
from afe.benchmark import compare, BaselineMethod
from mypkg.methods import MyMethod

result = compare(
    methods=[BaselineMethod, MyMethod],
    datasets=["german-credit", "concrete-strength"],   # built-in suite keys
)
print(result)          # per-model-family markdown score table
df = result.to_frame() # raw rows as a pandas DataFrame
```

### Plugging in a method

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

Both dataset sources can be mixed in one call.

### Benchmarking a third-party library

Nothing about a competitor method is special — install it and write the same
plain function. Wrapping [OpenFE](https://github.com/IIIS-Li-Group/OpenFE)
(NeurIPS 2022) takes four lines:

```bash
pip install openfe
```

```python
# adapters.py
def openfe(X_train, y_train, X_test, task):
    from openfe import OpenFE, transform
    feats = OpenFE().fit(data=X_train, label=y_train, task=task, n_jobs=1, verbose=False)
    return transform(X_train, X_test, feats[:10], n_jobs=1)
```

```bash
python -m scripts.run_benchmark --datasets german-credit \
    --methods baseline adapters:openfe --budget 300
```

That is the complete adapter. Pass `seed=0` to `.fit()` if you want runs to
be reproducible — OpenFE is otherwise nondeterministic, and its score will
drift slightly between runs.

Two things the harness handles so your adapter doesn't have to:

- **`task` is always `"classification"` or `"regression"`.** Datasets are
  also tagged `multiclass`, but that only changes how the *scoring panel*
  computes AUC — it is narrowed before your method sees it, so you never
  write `"regression" if task == "regression" else "classification"`.
- **Generation runs in a fresh temp working directory.** Several libraries
  (openfe included) write scratch files to hardcoded relative paths that
  collide between runs. The harness isolates the cwd around every
  `fit_transform`/`transform`, so this class of bug cannot occur.
- **Warnings from inside your method are silenced.** Method internals can
  warn per model fit — OpenFE emits ~70 identical LightGBM deprecation
  warnings in one small run — which buries the harness's own output. Only
  the generation step is silenced; dataset loading, encoding, and scoring
  warnings still surface. Set `AFE_METHOD_WARNINGS=default` (or `once`,
  `error`) to see them while debugging your method.

Use the class form only when a library needs state carried from fit to
transform that the function form can't express.

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

### Dataset setup

Most datasets download automatically on first use and are cached under
`data/cache/`. Two exceptions:

**Kaggle credentials** — needed for `ieee-cis-fraud`, `bnp-paribas-claims`,
`home-credit-default`, and `house-prices`:

1. Kaggle → **Account → API → Create New API Token** (downloads `kaggle.json`).
2. ```bash
   mkdir -p ~/.kaggle && mv ~/Downloads/kaggle.json ~/.kaggle/
   chmod 600 ~/.kaggle/kaggle.json
   ```
3. **Accept each competition's rules once** in the browser (competition page →
   *Rules* → *I Understand and Accept*), or downloads fail with HTTP 403.

**Datasets needing a manual drop** — `microsoft-mslr`, `medical`, and
`broken-machine` have no clean canonical source. They ship as RTDL-style
`.npy` dumps in a `data.zip` linked from
[`ZhangTP1996/OpenFE_reproduce`](https://github.com/ZhangTP1996/OpenFE_reproduce).
Download it, unzip, and drop each dataset's folder at:

```
data/cache/raw/<key>/            # <key> = microsoft-mslr | medical | broken-machine
  N_train.npy  N_val.npy  N_test.npy   # numeric features (required)
  C_train.npy  C_val.npy  C_test.npy   # categorical features (optional)
  y_train.npy  y_val.npy  y_test.npy   # target (required)
```

The zip's internal folder names are `microsoft`, `medical`, and
`broken_machine` — rename each to the `<key>` above when copying. All three
source splits are concatenated into one frame; the harness then freezes its
own fixed split, uniformly, for every dataset.

### Long unattended runs

A full sweep takes many hours. `scripts/benchmark_ctl.sh` wraps the
backgrounding so you don't hand-manage `nohup`, PID files, or log
redirection — and it avoids a real pitfall: killing a naively-backgrounded
Python process can leave its `multiprocessing` workers alive and consuming
memory.

```bash
# Anything after `--` is passed straight through to run_benchmark.py:
./scripts/benchmark_ctl.sh start -- --methods baseline mypkg.methods:MyMethod --budget 900
./scripts/benchmark_ctl.sh status     # running? + row/status counts so far
./scripts/benchmark_ctl.sh tail       # follow the log
./scripts/benchmark_ctl.sh stop       # kills the whole process group
./scripts/benchmark_ctl.sh report -- --out docs/benchmark_report.md
```

## Parameters

All flags of `python -m scripts.run_benchmark`:

| flag | default | meaning |
|---|---|---|
| `--datasets` | all 22, smallest-scale-first | benchmark keys to run — space-separated, e.g. `--datasets nomao covertype` |
| `--methods` | `baseline` | methods to benchmark: the built-in `baseline`, and/or import paths to your own, e.g. `mypkg.methods:MyMethod`. A path may point at a class or a plain function |
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

### Available methods

| method | class | what it does |
|---|---|---|
| `baseline` | `BaselineMethod` | No feature engineering — the raw (prepped) features. The reference arm every comparison is read against. |

That is the complete list. Everything else is supplied by you — see
[Plugging in a method](#plugging-in-a-method) above and
[Benchmarking a third-party library](#benchmarking-a-third-party-library)
below.

### Available datasets

The frozen suite has **22 datasets** covering both task types, small through
large scale, and a spread of sectors. `afe/benchmark/registry.py` is the
source of truth. The metric is chosen by the harness from the data: `r2`
(+ `mae`) for regression, `auc` for binary classification, macro one-vs-rest
`auc_ovr` when there are more than two classes.

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
columns excluding the target). Splits and OpenML versions are pinned in
committed manifests (`afe/benchmark/manifests/`), so every machine evaluates
the same tables on the same splits.

## Reports

### Raw output

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

| field | meaning |
|---|---|
| `key` | dataset key |
| `method` | method name — `baseline`, or the import path / `.name` of a supplied method |
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

Straight into pandas:

```python
import pandas as pd

df = pd.read_json("results/my_run.jsonl", lines=True)
df.groupby(["method", "model_family"])["value"].mean()
```

### Generating a markdown report

```bash
python -m scripts.report_benchmark results/my_run.jsonl                        # to stdout
python -m scripts.report_benchmark results/my_run.jsonl --out docs/report.md   # to a file
```

Reports are generated purely from the JSONL — regenerating one never
re-runs the benchmark. Dataset columns are abbreviated (initials of the
hyphenated key) and grouped by task, then sector, then key. Scores are
fold-means when a results file holds multiple folds per (dataset, method,
model family).

**Full worked example: [`docs/benchmark_report.md`](docs/benchmark_report.md)**
— OpenFE vs. a CAFEM-style RL method across all 22 datasets. Neither ships
with this harness; both were plugged in as external methods exactly as
described in [Benchmarking a third-party library](#benchmarking-a-third-party-library),
which makes the report a working demonstration of that path. It contains:

*Datasets* — the legend mapping each abbreviation to its full key, task,
sector, and metric:

| abbrev | dataset | task | sector | metric |
|---|---|---|---|---|
| QB | qsar-biodegradation | classification | chemistry | auc |
| CH | california-housing | regression | general/real-estate | r2 |

*Overview* — one row per method, one column per dataset; the cell is the mean
score across the three model families, best per column in bold:

| method | QB | E | BM | ... |
|---|---|---|---|---|
| cafem | **0.924** | **0.890** | 0.882 | ... |
| openfe | 0.922 | 0.860 | **0.902** | ... |

*Per-method scores* — the breakdown behind the overview, each method's three
model-family rows per dataset.

*Speed* — feature-generation wall-time per (method, dataset), plus a
per-method median.

*Feature counts* — columns fed to the downstream models before and after each
method's generated features are added (e.g. `41 -> 51`).

*By task* / *By sector* — the same overview score averaged over
classification / multiclass / regression, and over each dataset's sector:

| method | classification | multiclass | regression |
|---|---|---|---|
| cafem | 0.820 | **0.852** | 0.618 |
| openfe | **0.871** | — | **0.843** |

*Failures / timeouts / crashes* — every non-`ok` row counted per (dataset,
method, status), so a method that wins on the datasets it finished is never
confused with one that finished everywhere.
