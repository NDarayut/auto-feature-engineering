# MF-OpenFE

Automatic feature engineering with a meta-learned filter, plus a **modular,
algorithm-agnostic benchmark harness** for comparing any AutoFE method
fairly on a frozen dataset suite.

- **`afe.MFOpenFE`** — generate candidate features with
  [OpenFE](https://github.com/IIIS-Li-Group/OpenFE)'s operator library, prune
  the pool with a meta-model trained offline on ~100 historical datasets,
  then verify and keep only the features that measurably help.
- **`afe.benchmark`** — a plug-and-play benchmark: bring any AutoFE method
  (a plain function or a small class — yours, ours, or a third-party
  library's), and it is evaluated on identical frozen splits, with an
  identical model panel and metrics, alongside any other method.

```python
from afe import MFOpenFE

mfe = MFOpenFE(task="classification")
X_train_fe = mfe.fit_transform(X_train, y_train)
X_test_fe = mfe.transform(X_test)
```

---

## Contents

- [Installation](#installation)
- [Quickstart: MF-OpenFE](#quickstart-mf-openfe)
- [The benchmark](#the-benchmark)
  - [Quick example](#quick-example)
  - [Plugging in your own method](#plugging-in-your-own-method)
  - [Built-in methods](#built-in-methods)
  - [How methods are benchmarked](#how-methods-are-benchmarked)
  - [Metrics](#metrics)
  - [Running from the command line](#running-from-the-command-line)
  - [Benchmark datasets](#benchmark-datasets)
  - [Results: OpenFE vs. CAFEM](#results-openfe-vs-cafem)
- [How MF-OpenFE works](#how-mf-openfe-works)
- [Training your own meta-model](#training-your-own-meta-model)
- [Project layout](#project-layout)

---

## Installation

```bash
git clone <this-repo> && cd auto-feature-engineering
python -m venv .venv && . .venv/bin/activate
pip install -e .                  # just MFOpenFE (pandas, numpy, sklearn, openfe, lightgbm)
pip install -e ".[benchmark]"     # + the benchmark harness (openml, ucimlrepo, kaggle, ...)
pip install -e ".[benchmark,test]" # + pytest
```

Python 3.10+ (developed on 3.12). To download the benchmark datasets, see
[`docs/dataset_setup.md`](docs/dataset_setup.md) — most fetch automatically,
four need Kaggle credentials, three need a one-time manual drop.

## Quickstart: MF-OpenFE

```python
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from afe import MFOpenFE

X, y = load_breast_cancer(as_frame=True, return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

mfe = MFOpenFE(task="classification")
X_train_fe = mfe.fit_transform(X_train, y_train)   # fit only ever touches train
X_test_fe = mfe.transform(X_test)                  # replays the same kept features

print(f"{X_train.shape[1]} raw -> {X_train_fe.shape[1]} engineered "
      f"({mfe.n_features_kept_} kept of {mfe.n_candidates_generated_} generated)")
```

`task` is optional (inferred from `y`); categoricals need no manual encoding.
Progress reporting is on by default (`progress=False` to silence).

---

## The benchmark

`afe.benchmark` is deliberately **standalone and symmetric**: it has no
privileged notion of "our method" vs. "their method". Every method — including
MF-OpenFE itself — enters through the same interface, runs on the same frozen
split, and is scored by the same model panel. You can use it to benchmark any
automatic feature engineering technique without editing library source.

### Quick example

```python
from afe.benchmark import compare, BaselineMethod, OpenFEMethod, CAFEMMethod

result = compare(
    methods=[BaselineMethod, OpenFEMethod, CAFEMMethod],
    datasets=["german-credit", "concrete-strength"],   # built-in suite keys
)
print(result)          # per-model-family markdown score table
df = result.to_frame() # raw rows as a pandas DataFrame
```

Output (illustrative):

```
## tree
| dataset | baseline | cafem | openfe |
|---|---|---|---|
| concrete-strength | 0.938 | 0.940 | 0.947 |
| german-credit | 0.829 | 0.830 | 0.841 |
| **mean** | 0.884 | 0.885 | 0.894 |
```

### Plugging in your own method

A method is either a **plain function**:

```python
def my_method(X_train, y_train, X_test, task):
    ...                                   # task is "classification" | "regression"
    return X_train_new, X_test_new        # original + engineered columns
```

or a **class** with the `fit_transform`/`transform` contract (this is
`afe.methods.AutoFEMethod`, a structural protocol — no inheritance needed):

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

Both dataset sources can be mixed in one call. `budget_seconds=None` (the
default) runs in-process — works from a notebook or REPL; setting it opts
into subprocess isolation with a hard wall-clock budget per (dataset, method)
pair, which requires the method to be picklable/importable. Full API
reference: [`afe/benchmark/README.md`](afe/benchmark/README.md).

### Built-in methods

All re-exported from `afe.benchmark`, all following the same contract:

| method | class | what it does |
|---|---|---|
| `baseline` | `BaselineMethod` | No feature engineering — the raw (prepped) features. Every comparison should include it. |
| `openfe` | `OpenFEMethod` | [OpenFE](https://github.com/IIIS-Li-Group/OpenFE) (NeurIPS 2022): full candidate expansion + two-stage boosting-based evaluation; keeps the top `n_new_features` (default 10). |
| `cafem` | `CAFEMMethod` | CAFEM-style (PAKDD 2020) per-dataset RL: a Double DQN searches the Feature Transformation Graph on the training fold; the best-scoring transformation chain per feature is kept and replayed on test. |
| `featuretools` | `FeaturetoolsMethod` | Single-table Deep Feature Synthesis (pairwise arithmetic primitives). |
| `autofeat` | `AutofeatMethod` | Autofeat's iterative non-linear expansion + L1-based selection. |
| — | `afe.MFOpenFE` | This project's method; pass it in `methods=` like any other. |

### How methods are benchmarked

The harness holds everything except the method constant, so score differences
reflect the method — not the split, the preprocessing, or the downstream model:

1. **Frozen split.** Each dataset gets a single fixed 80/20 train/test holdout
   (stratified for classification). The seed is derived deterministically from
   the dataset key (`SHA256(key)`, offset `20260101`) using NumPy's
   stability-guaranteed legacy RNG, so *same dataset ⇒ same split*, on any
   machine, in every run, for every method.
2. **Uniform preprocessing.** Before any method sees the data: categoricals
   are ordinal-encoded and numerics median-imputed, **fit on the training fold
   only**. Every method receives the identical numeric, NaN-free matrix, so no
   method gains an advantage from its own dtype handling.
3. **Leak-safety contract.** `fit_transform` only ever receives the training
   fold; `transform` replays fitted state on the test fold. The harness never
   passes test rows to a fitting step.
4. **Generation under budget (opt-in).** With a budget set, feature generation
   runs in a spawned subprocess with a hard wall-clock limit. Methods that
   exceed it are killed and recorded as `timeout`; crashes are recorded as
   `error`. **Failures are rows in the results, not silent exclusions.**
5. **A 3-family model panel.** The engineered features are evaluated with
   three deliberately different downstream learners, per task:

   | family | classifier | regressor | post-generation prep |
   |---|---|---|---|
   | `tree` | LightGBM (200 trees) | LightGBM (200 trees) | ±inf → NaN (LightGBM handles NaN natively) |
   | `linear` | Logistic Regression | Ridge | median-impute + standard-scale, fit on train |
   | `knn` | kNN (k=10) | kNN (k=10) | median-impute + standard-scale, fit on train |

   A feature set that only helps one model family is a model-specific artifact;
   one that helps all three is genuinely better representation. (Because kNN
   prediction cost is O(n_train × n_test), the kNN family fits on at most
   50 000 seeded-subsample train rows and scores at most 20 000 test rows —
   identically for every method on a dataset, so comparisons are unaffected.)
6. **One row per (dataset, method, model family)** is written to JSONL as it
   is produced: metric value, generation wall-time, feature counts, and
   status. Runs are resumable — completed pairs are skipped on restart.

### Metrics

| task | primary metric | notes |
|---|---|---|
| binary classification | **ROC AUC** | threshold-free, robust to class imbalance |
| multiclass classification | **macro one-vs-rest ROC AUC** | reported as `auc_ovr` |
| regression | **R²** | + **MAE** recorded as a secondary, scale-aware metric |

Beyond the predictive metric, each result row records the full evaluation
criteria from the benchmark plan ([`draft_plan.md`](draft_plan.md) §3.2/§5.2):

| field | criterion |
|---|---|
| `fit_elapsed_s`, `transform_elapsed_s`, `gen_elapsed_s` | generation cost and inference-time cost of computing the features on new data |
| `peak_mem_mb` | peak memory of the generation subprocess |
| `n_candidates` | features surviving the method's *own* internal selection |
| `n_features_generated`, `n_features_final` | feature yield actually added / total |
| `feature_efficiency` | fraction of newly generated features that are individually predictive (univariate target association ≥ 0.1 on the train fold) — "many features that are also *good*", not just many features |

A method is preferred only if it improves performance *per compute budget*
over the baseline, not merely if it adds volume.

### Running from the command line

```bash
# Compare methods on chosen datasets, 900s generation budget each:
python -m scripts.run_benchmark \
    --datasets german-credit concrete-strength house-prices \
    --methods baseline openfe cafem \
    --budget 900 --out results/my_run.jsonl

# Everything (all 22 datasets, smallest first), resumable:
python -m scripts.run_benchmark --methods baseline openfe cafem

# Aggregate a results file into a task/sector report:
python -m scripts.report_benchmark results/my_run.jsonl
```

For long runs there is a supervised background runner —
[`docs/benchmark_ctl_usage.md`](docs/benchmark_ctl_usage.md).

### Benchmark datasets

The frozen suite has **22 datasets** chosen to cover both task types, small
through large scale, and a spread of sectors; 10 (marked ✓) are reused from
the OpenFE paper for parity with its published results. The registry
(`afe/benchmark/registry.py`) is the source of truth; download instructions:
[`docs/dataset_setup.md`](docs/dataset_setup.md).

| key | task | sector | scale | source | OpenFE paper |
|---|---|---|---|---|:--:|
| california-housing | regression | real estate | medium | sklearn | ✓ |
| microsoft-mslr | regression | web ranking | large | OpenFE_reproduce | ✓ |
| medical | regression | healthcare | medium | OpenFE_reproduce | ✓ |
| superconductivity | regression | materials | medium | UCI | |
| concrete-strength | regression | materials | small | UCI | |
| house-prices | regression | real estate | medium | Kaggle | |
| diabetes-130us | classification | healthcare | large | OpenML | ✓ |
| nomao | classification | general | medium | OpenML | ✓ |
| vehicle-sensit | classification | sensors | medium | OpenML | ✓ |
| broken-machine | classification | industrial | large | OpenFE_reproduce | ✓ |
| telecom-churn | classification | telco | medium | OpenML | ✓ |
| jannis | classification | general | medium | OpenML | ✓ |
| covertype | multiclass | environment | large | OpenML | ✓ |
| ieee-cis-fraud | classification | finance | large | Kaggle | |
| bnp-paribas-claims | classification | finance | medium | Kaggle | |
| home-credit-default | classification | finance | large | Kaggle | |
| german-credit | classification | finance | small | UCI | |
| heart-disease | classification | healthcare | small | UCI | |
| breast-cancer-wisconsin | classification | healthcare | small | sklearn | |
| qsar-biodegradation | classification | chemistry | small | OpenML | |
| bank-marketing | classification | finance | medium | OpenML | |
| electricity | classification | energy | medium | OpenML | |

Two hard rules keep results honest:

- **Frozen manifests.** The suite and its split protocol are committed
  (`afe/benchmark/manifests/`); OpenML versions are pinned, so every machine
  evaluates the same tables on the same splits.
- **Benchmark ∩ meta-training corpus = ∅.** MF-OpenFE's meta-model is trained
  on a separate ~100-dataset OpenML corpus, and the manifest builder removes
  every benchmark dataset (and its aliases) from that corpus — no method is
  ever graded on data it was trained on. Enforced by `tests/test_disjoint.py`.

### Results: OpenFE vs. CAFEM

Baseline vs. OpenFE vs. CAFEM on the 10 small/medium suite datasets
(900 s budget, no timeouts, 90/90 cells completed). Mean held-out score per
model family (AUC / R² mixed — directional only; per-dataset tables and
discussion in [`docs/benchmark_results.md`](docs/benchmark_results.md)):

| model family | baseline | openfe | cafem |
|---|---|---|---|
| tree (LightGBM) | **0.867** | 0.860 | **0.867** |
| linear | 0.666 | **0.713** | 0.685 |
| knn | 0.807 | **0.818** | 0.807 |
| median generation time | 0 s | 31.6 s | 1.0 s |

In short: OpenFE clearly helps linear/kNN models (+0.047 mean on linear) but
does not reliably beat raw features under LightGBM; CAFEM's unary-chain
search is ~30× cheaper and near-harmless by construction (it keeps nothing
when nothing helps), but rarely finds large wins.

---

## How MF-OpenFE works

One `fit_transform` call runs four steps:

1. **Generate** — `openfe.get_candidate_features()` builds the candidate pool
   using OpenFE's operator library (arithmetic combinations, groupby
   aggregations, `log`/`sqrt`/`square`/`sigmoid`/…).
2. **Meta-filter** — each candidate is scored by a model trained offline on a
   ~100-dataset corpus: given the feature's scale-invariant distributional
   sketch and the operator applied, predict whether it is worth evaluating.
   Candidates below the threshold are dropped before any expensive evaluation.
   (Currently covers `log`/`sqrt`/`square`/`sigmoid` candidates; operators
   without training signal pass through unfiltered rather than being guessed
   at. Held-out corpus AUC: 0.80 against a 13% base rate.)
3. **Verify + select** — survivors are added to the raw features, one LightGBM
   fit ranks them by importance, and low-importance candidates are dropped.
4. **Replay** — `transform` applies exactly the kept features to new data.

The full staged design and each technique's source paper:
[`algorithm_plan.md`](algorithm_plan.md).

```python
MFOpenFE(
    task="regression",          # inferred from y if omitted
    order=1,                    # OpenFE candidate-generation order
    filter_threshold=0.5,       # min meta-model score to survive the filter
    max_candidates=100,         # cap on candidates reaching verify/select
    importance_threshold=0.0,   # min LightGBM importance to be kept
    progress=True,
)
```

## Training your own meta-model

The shipped meta-model is trained in two offline steps — a CAFEM-style RL
search over the corpus produces labeled examples, then a supervised model is
fit on them. To retrain (e.g. on a different corpus):

```bash
pip install -e ".[benchmark]"
python -m scripts.run_stage0 --episodes 80     # RL label generation
python -m scripts.train_meta_model             # trains + saves models/meta_model.pkl
```

Design details: [`afe/meta/README.md`](afe/meta/README.md).

## Project layout

```
afe/            production library (pip-installable, `import afe`)
  meta/           MF-OpenFE itself: online path + offline training pipeline
  methods.py      AutoFE method adapters (baseline/openfe/cafem/featuretools/autofeat)
  benchmark/      the benchmark harness + the compare() API
scripts/        CLI entrypoints (run_benchmark, report_benchmark, run_stage0, ...)
dev/            smoke/sanity utilities (smoke_download, parity_check)
tests/          pytest suite — offline, no network
docs/           guides: dataset setup, benchmark methodology, results
research/       source papers backing the design choices
data/, results/, models/   gitignored: dataset cache, run outputs, trained models
```

```bash
pytest tests/ -q   # offline, ~10s
```

Further reading: [`docs/benchmark_guide.md`](docs/benchmark_guide.md) (harness
data flow), [`docs/dataset_setup.md`](docs/dataset_setup.md) (standing up the
data), [`algorithm_plan.md`](algorithm_plan.md) (the method's design),
[`synthesis_report.md`](synthesis_report.md) (literature synthesis).
