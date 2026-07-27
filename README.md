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

### 1. Install

Python 3.10+ (developed on 3.12), on Linux, macOS, or Windows.

```bash
git clone https://github.com/NDarayut/auto-feature-engineering.git
cd auto-feature-engineering
python -m venv .venv
```

Activate it, then install:

| | activate |
|---|---|
| Linux / macOS | `source .venv/bin/activate` |
| Windows (PowerShell) | `.venv\Scripts\Activate.ps1` |
| Windows (cmd) | `.venv\Scripts\activate.bat` |

```bash
pip install -e ".[datasets]"   # harness + dataset fetchers
```

If PowerShell blocks the activation script, allow it for the current session
with `Set-ExecutionPolicy -Scope Process RemoteSigned`.

From another project:

```bash
pip install "auto-feature-engineering[datasets] @ git+https://github.com/NDarayut/auto-feature-engineering.git"
```

Everything below is shown with Unix shell syntax. Two things differ on
Windows — `> /dev/null` becomes `> $null` (PowerShell) or `> NUL` (cmd), and
`scripts/benchmark_ctl.sh` needs WSL or Git Bash. See
[Platform notes](#platform-notes) for the full list.

### 2. Set up the datasets

Everything downloads automatically to `data/cache/` on first use, except:

**Kaggle** (for `ieee-cis-fraud`, `bnp-paribas-claims`, `home-credit-default`,
`house-prices`) — download an API token
([Account → API → Create New API Token](https://www.kaggle.com/settings))
and put `kaggle.json` where the Kaggle client looks for it:

```bash
# Linux / macOS
mkdir -p ~/.kaggle && mv ~/Downloads/kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json
```

```powershell
# Windows (PowerShell) -- no chmod needed
New-Item -ItemType Directory -Force "$env:USERPROFILE\.kaggle" | Out-Null
Move-Item "$env:USERPROFILE\Downloads\kaggle.json" "$env:USERPROFILE\.kaggle\"
```

Then accept each competition's rules once, or downloads fail with HTTP 403:
[IEEE-CIS Fraud](https://www.kaggle.com/competitions/ieee-fraud-detection/rules) ·
[BNP Paribas Claims](https://www.kaggle.com/competitions/bnp-paribas-cardif-claims-management/rules) ·
[Home Credit Default](https://www.kaggle.com/competitions/home-credit-default-risk/rules) ·
[House Prices](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/rules)

**Manual drop** (for `microsoft-mslr`, `medical`, `broken-machine`) — grab
`data.zip` from
[`ZhangTP1996/OpenFE_reproduce`](https://github.com/ZhangTP1996/OpenFE_reproduce),
unzip, and copy each folder (`microsoft`, `medical`, `broken_machine`) to
`data/cache/raw/<key>/`, renamed to the dataset key. Each must contain
`N_*.npy`, `y_*.npy` (and optionally `C_*.npy`) for `train`/`val`/`test`.

Skip either step if you don't need those datasets — the other 15 work
without it.

### 3. Run a benchmark

Install a method to test and write a small adapter. OpenFE takes four lines:

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

Then compare it against the baseline:

```python
from afe.benchmark import compare, BaselineMethod
from adapters import openfe

compare(
    methods=[BaselineMethod, openfe],
    datasets=["german-credit", "concrete-strength"],
    report_path="results/my_run.md",
)
```

Progress prints as it runs, and the report is written when it finishes — no
separate reporting step:

```
benchmarking 2 methods on 2 datasets -- 4 pairs
[1/4] german-credit / baseline: ok in 0.0s -- knn 0.780  linear 0.815  tree 0.829
[2/4] german-credit / openfe: ok in 43.7s -- knn 0.776  linear 0.822  tree 0.831
[3/4] concrete-strength / baseline: ok in 0.0s -- knn 0.716  linear 0.647  tree 0.935
[4/4] concrete-strength / openfe: ok in 6.8s -- knn 0.818  linear 0.789  tree 0.936
done: 12 result rows in 52.0s
report: results/my_run.md
```

`report_path` is optional — without it the report goes to
`results/compare_report.md`. Pass `report_path=None` to skip it.

Inside `results/my_run.md`:


## Overview

| method | GC | CS |
|---|---|---|
| baseline | 0.808 | 0.766 |
| openfe | **0.810** | **0.848** |

## Per-method scores

| method | model | GC | CS |
|---|---|---|---|
| baseline | knn | 0.780 | 0.716 |
|  | linear | 0.815 | 0.647 |
|  | tree | 0.829 | 0.935 |
| openfe | knn | 0.776 | 0.818 |
|  | linear | 0.822 | 0.789 |
|  | tree | 0.831 | 0.936 |


Read the per-method table across model families: on `concrete-strength`
(CS) OpenFE's features are worth a lot to the linear model (0.647 → 0.789)
and the kNN (0.716 → 0.818), but almost nothing to the tree
(0.935 → 0.936), which already captures those interactions on its own. That
is why three families are scored — a single-model view would have called
this a big win or no win at all, depending on which model you picked.

The report also carries dataset/speed/feature-count/by-sector tables and a
failure summary — see [Reports](#reports). (These exact numbers will shift a
little between runs: OpenFE is nondeterministic unless you pass `seed=0`.)

### Common variations

```python
from afe.benchmark import compare, BENCHMARK, BaselineMethod
from adapters import openfe

# Every dataset in the suite (long — see "Long unattended runs" below).
compare(methods=[BaselineMethod, openfe],
        datasets=[spec.key for spec in BENCHMARK])

# Your own data instead of (or alongside) the built-in suite.
compare(methods=[BaselineMethod, openfe],
        custom_datasets={"my-data": (X, y)})

# Score with one model family only — roughly 3x faster.
compare(methods=[BaselineMethod, openfe],
        datasets=["german-credit"], model_families=["tree"])

# Kill any method that exceeds 300s, and save + resume to JSONL.
compare(methods=[BaselineMethod, openfe],
        datasets=["nomao", "electricity"],
        budget_seconds=300,
        out_path="results/my_run.jsonl")

# Silence the progress lines, and skip the report.
compare(methods=[BaselineMethod, openfe],
        datasets=["german-credit"],
        progress=False, report_path=None)

# The return value carries the same rows, for your own analysis.
result = compare(methods=[BaselineMethod, openfe], datasets=["german-credit"])
df = result.to_frame()      # one row per (dataset, method, model family)
```

`budget_seconds` also moves generation into a subprocess, so a method that
hangs or crashes is recorded and the run continues. Without it, methods run
in-process with no timeout.

> **Large datasets: use the command line.** `compare()` applies no width cap,
> row sampling, or memory ceiling, so `compare(datasets=["ieee-cis-fraud"])`
> (590k rows × 393 columns) can exhaust your RAM. The CLI's guards are on by
> default and degrade gracefully instead.

### 4. The same thing from the command line

Reference a method by import path; `baseline` is the one built-in name.

```bash
# The run from step 3:
python -m scripts.run_benchmark \
    --datasets german-credit concrete-strength \
    --methods baseline adapters:openfe \
    --out results/my_run.jsonl

# All 22 datasets (smallest first), 900s budget, resumable if interrupted:
python -m scripts.run_benchmark --methods baseline adapters:openfe --budget 900
```

It reports progress as it goes and writes the markdown report itself:

```
benchmarking 2 method(s) on 2 dataset(s) -- 4 pairs
[1/4] german-credit / baseline: ok in 0.0s -- knn 0.780  linear 0.815  tree 0.829
[2/4] german-credit / adapters:openfe: ok in 2.4m -- knn 0.780  linear 0.821  tree 0.822
[3/4] concrete-strength / baseline: ok in 0.0s -- knn 0.716  linear 0.647  tree 0.935
[4/4] concrete-strength / adapters:openfe: ok in 42.6s -- knn 0.818  linear 0.789  tree 0.936
done: 12 result rows in 3.3m -> results/my_run.jsonl
report: results/my_run.md
```

Failures appear in the same stream (`TIMEOUT`, `OOM`, `CRASHED`, `ERROR`
with the message), so a method failing everywhere is obvious immediately
rather than at the end of a multi-hour sweep.

The report lands next to the JSONL (`--report PATH` to place it elsewhere,
`--no-report` to skip, `--quiet` to drop the progress lines).

Progress goes to **stderr**, which matters more than it sounds: many AutoFE
libraries print heavily to stdout — one OpenFE run here emitted 24 MB of
LightGBM logging. Redirect stdout away and you get a clean progress feed:

```bash
python -m scripts.run_benchmark --methods baseline adapters:openfe > /dev/null
```

Runs are **resumable**: (dataset, method) pairs already in `--out` are
skipped on restart, and the report still covers the whole results file, not
just the pairs this invocation ran. Unlike `compare()`, the CLI applies
memory and width guards by default — see [Parameters](#parameters).

### 5. Plug in your own method

A method is either a **plain function**:

```python
def my_method(X_train, y_train, X_test, task):
    ...                                   # task is "classification" | "regression"
    return X_train_new, X_test_new        # original + engineered columns
```

or a **class** with the `fit_transform`/`transform` contract (a structural
protocol — no inheritance needed), for libraries that carry state from fit
to transform:

```python
class MyMethod:
    name = "my-method"                    # display name in result tables

    def fit_transform(self, X_train, y_train, task): ...  # only ever sees train
    def transform(self, X_test): ...                      # replays fitted state
```

Either form drops into the same calls:

```python
compare(methods=[BaselineMethod, my_method, MyMethod,
                 ("my-tuned-variant", MyMethod())],   # (name, instance) to relabel
        datasets=["german-credit"])
```

```bash
python -m scripts.run_benchmark --methods baseline mypkg.methods:MyMethod
```

Three things the harness handles so your method doesn't have to:

- **`task` is always `"classification"` or `"regression"`.** Datasets are
  also tagged `multiclass`, but that only changes how the *scoring panel*
  computes AUC — it is narrowed before your method sees it.
- **Generation runs in a fresh temp working directory.** Several libraries
  (openfe included) write scratch files to hardcoded relative paths that
  collide between runs.
- **Warnings from inside your method are silenced.** OpenFE alone emits ~70
  identical LightGBM deprecation warnings per small run. Only the generation
  step is silenced — dataset loading, encoding, and scoring warnings still
  surface. Set `AFE_METHOD_WARNINGS=default` to see them while debugging.

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
   (`tree`), Logistic Regression / Ridge (`linear`), and kNN (`knn`);
5. writes **one JSONL row per (dataset, method, model family)** as it is
   produced.

Metrics: ROC AUC for binary classification, macro one-vs-rest ROC AUC for
multiclass, R² (plus MAE) for regression.

### Long unattended runs

A full sweep takes many hours. `scripts/benchmark_ctl.sh` wraps the
backgrounding so you don't hand-manage `nohup`, PID files, or log
redirection — and it avoids a real pitfall: killing a naively-backgrounded
Python process can leave its `multiprocessing` workers alive and consuming
memory.

It is a bash script: Linux and macOS run it directly, Windows needs WSL or
Git Bash. On native Windows use `Start-Process` (or just run the CLI in a
dedicated terminal) — runs are resumable, so an interrupted sweep continues
where it stopped.

```bash
# Anything after `--` is passed straight through to run_benchmark.py:
./scripts/benchmark_ctl.sh start -- --methods baseline adapters:openfe --budget 900
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
| `--report` | `--out` with a `.md` suffix | where to write the markdown report generated when the run finishes |
| `--no-report` | off | skip report generation |
| `--quiet` | off | suppress the per-(dataset, method) progress lines |

`--max-mem-gb` is **not enforced on Windows** — see
[Platform notes](#platform-notes).

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
[Plug in your own method](#5-plug-in-your-own-method).

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

## Platform notes

The harness runs on Linux, macOS, and Windows. Splits, preprocessing,
scoring, and reports are identical everywhere — a results file from one
platform is directly comparable to another's. Three things do differ:

| | Linux | macOS | Windows |
|---|---|---|---|
| `--max-mem-gb` hard cap | enforced | enforced | **not enforced** |
| `peak_mem_mb` in results | yes | yes | `null` |
| `scripts/benchmark_ctl.sh` | yes | yes | WSL / Git Bash |

**`--max-mem-gb` on Windows.** The cap uses `RLIMIT_AS` from the POSIX
`resource` module, which Windows has no equivalent for. The flag is accepted
and silently skipped rather than failing — so a method that would have been
killed as `oom` on Linux can instead exhaust the machine. The other guards
(`--max-cols`, `--fit-sample-rows`, `--transform-chunk-rows`) are pure Python
and work everywhere, so lower those on Windows for the largest datasets.

**`peak_mem_mb`** comes from the same module and is `null` on Windows. Every
other field is populated on all three platforms. Note that macOS and Linux
report `ru_maxrss` in different units (bytes vs kilobytes); the harness
converts per-platform, so the values are comparable.

**Paths and shells.** Output paths accept forward slashes on every platform.
Redirect stdout with `> /dev/null` (Linux/macOS), `> $null` (PowerShell), or
`> NUL` (cmd) — progress goes to stderr and survives all three.

## Reports

### Raw output

`--out` (CLI) and `out_path=` (`compare()`) write a JSONL file, one row per
(dataset, method, model family), flushed as each row is produced. A
successful row:

```json
{"key": "nomao", "method": "openfe", "fold_id": "split0", "protocol": "holdout", "task": "classification", "status": "ok", "gen_elapsed_s": 11.10, "fit_elapsed_s": 11.01, "transform_elapsed_s": 0.09, "peak_mem_mb": 564.7, "n_candidates": 8, "feature_efficiency": 1.0, "model_family": "tree", "n_features_generated": 8, "n_features_final": 126, "metric": "auc", "value": 0.9941}
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

Both entry points write one automatically — the CLI next to `--out`,
`compare()` to `report_path` or `results/compare_report.md`. To regenerate
from an existing results file, or to report on a run someone else produced:

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
with this harness; both were plugged in as external methods, which makes the
report a working demonstration of that path. It contains:

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
