# MF-OpenFE

Automatic feature engineering: generate candidate features with
[OpenFE](https://github.com/IIIS-Li-Group/OpenFE)'s own operator library
(arithmetic, groupby aggregations, log/sqrt/square/..., not a fixed hand-picked
set), prune the candidate pool with a meta-model trained offline on ~100
historical datasets to predict which transformations are actually worth
evaluating, then verify + keep the ones that measurably help. The idea is
the same as running OpenFE directly, minus paying full evaluation cost on
every single candidate it generates — a trained model decides which
candidates are worth evaluating in the first place.

```python
from afe import MFOpenFE

mfe = MFOpenFE(task="classification")
X_train_fe = mfe.fit_transform(X_train, y_train)
X_test_fe = mfe.transform(X_test)
```

## Install

```bash
pip install auto-feature-engineering
```

That's the base install — just what's needed to run `MFOpenFE` (pandas,
numpy, scikit-learn, openfe, lightgbm, tqdm). From a clone instead of PyPI:

```bash
git clone <this-repo> && cd auto-feature-engineering
python -m venv .venv && . .venv/bin/activate
pip install -e .
```

## Quickstart

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

`task` is optional — inferred from `y` if omitted. Categorical columns need
no manual encoding; `MFOpenFE` handles that internally.

Feature generation can take a while on wider datasets, so `MFOpenFE` reports
what it's doing as it goes (on by default, `progress=False` to silence it):

```
[MF-OpenFE    0.0s] Preparing data (encoding categoricals, imputing)...
[MF-OpenFE    0.0s] Loading meta-model (models/meta_model.pkl)...
[MF-OpenFE    0.1s] Generating candidate features (OpenFE operator library)...
[MF-OpenFE    0.6s] Generated 2850 raw candidates
Meta-filtering: 100%|██████████| 120/120 [00:01<00:00, 91.2it/s]
[MF-OpenFE    2.4s] 30 candidates survive the meta-filter (threshold=0.5, cap=30)
[MF-OpenFE    2.4s] Verifying + selecting from 30 candidates...
[MF-OpenFE    2.7s] Done: kept 21 engineered features (of 2850 generated)
```

`mfe.n_candidates_generated_`, `mfe.n_candidates_after_filter_`, and
`mfe.n_features_kept_` are set after `fit_transform`, useful for logging or
a dashboard in a larger pipeline.

## How it works

MF-OpenFE runs in four steps, each one call to `fit_transform`:

**1. Generate.** `openfe.get_candidate_features()` builds a pool of candidate
features using OpenFE's own operator library — arithmetic combinations,
groupby aggregations, `log`/`sqrt`/`square`/`sigmoid`/etc. This is the same
candidate space OpenFE itself would explore, not a hand-picked subset.

**2. Meta-filter.** Each candidate is scored by a model trained offline, once,
on ~100 historical datasets: given a feature's distributional shape (a
scale-invariant sketch, so the model transfers across datasets of any scale)
and the transformation applied to it, predict whether it's likely to be
useful. Candidates below the score threshold are dropped before the
expensive evaluation step below ever runs. The trained model currently
scores candidates built with `log`, `sqrt`, `square`, or `sigmoid` — the
operators it was trained on that also appear in OpenFE's real candidate
vocabulary; candidates built with other operators (arithmetic combinations,
groupby aggregations, `abs`, `freq`, `round`, ...) pass through this step
unfiltered rather than being guessed at, since the model has no training
signal for them yet. On a held-out slice of the training corpus, the model
scores candidate usefulness at AUC 0.80 against a 13% base rate — evidence
the learned pattern generalizes to datasets it never saw, which is the whole
premise of doing this offline instead of per-dataset.

**3. Verify + select.** The surviving candidates are added to the raw
features and one LightGBM model is fit; candidates whose feature importance
doesn't clear a threshold are dropped. This plays the same role as OpenFE's
own two-stage internal evaluation (residual-fitting plus successive
halving), simplified to one fit.

The full staged design behind this — including why the offline training step
(reinforcement learning over a feature-transformation graph) never runs at
usage time, and every technique's source paper — is in
[`algorithm_plan.md`](algorithm_plan.md). A dataset-level gatekeeper that
skips generation entirely on datasets predicted not to benefit
(`algorithm_plan.md` Stage 2) isn't built yet — `MFOpenFE` always attempts
generation for now.

### Configuration

```python
MFOpenFE(
    task="regression",          # "classification" | "regression"; inferred if omitted
    order=1,                    # OpenFE candidate-generation order
    filter_threshold=0.5,       # min meta-model usefulness score to survive the meta-filter
    max_candidates=100,         # cap on candidates reaching verify/select
    importance_threshold=0.0,  # min LightGBM feature importance to be kept
    progress=True,
)
```

## Training your own meta-model

The shipped `models/meta_model.pkl` was trained once on a ~100-dataset
corpus, kept disjoint from any benchmark data so results are never trained
on what they're graded on. Training happens in two offline steps: a
reinforcement-learning search explores which transformations help on each
historical dataset (producing labeled examples), then a supervised model is
fit on those examples. Retraining — for example on a different corpus — runs
the same two steps:

```bash
pip install "auto-feature-engineering[benchmark]"   # openml/ucimlrepo/kaggle etc.
python -m scripts.run_stage0 --episodes 80          # RL label generation
python -m scripts.train_meta_model                  # trains + saves models/meta_model.pkl
```

Full design: [`afe/meta/README.md`](afe/meta/README.md).

## Benchmarking

**Benchmark your own algorithm.** `afe.benchmark.compare()` is a standalone,
algorithm-agnostic API — plug in any AutoFE method (a plain function or a
class, yours or ours) and compare it against others on the built-in dataset
suite or your own data, no editing library source:

```python
from afe.benchmark import compare

def my_method(X_train, y_train, X_test, task):
    ...
    return X_train_new, X_test_new

result = compare(
    methods=[my_method],
    datasets=["german-credit"],              # built-in suite, optional
    custom_datasets={"my-data": (X, y)},     # your own data, optional
)
print(result)
```

Full reference, including the class-based method style and the opt-in
subprocess/budget execution mode: [`afe/benchmark/README.md`](afe/benchmark/README.md).

**The research harness.** This project's own comparison of baseline vs.
OpenFE vs. Featuretools vs. Autofeat vs. MF-OpenFE, run fairly (same split,
same model panel, same compute budget) against a frozen 22-dataset suite.
Methodology: [`draft_plan.md`](draft_plan.md); full flow:
[`docs/benchmark_guide.md`](docs/benchmark_guide.md). The benchmark's dataset
registry and the meta-training corpus are disjoint by construction
(`afe/benchmark/manifests.py`) — MF-OpenFE is never trained on a dataset it's
later graded on. Install the extra it needs: `pip install -e ".[benchmark]"`.

## Development

```
afe/            production library (pip-installable, `import afe`)
  meta/           MF-OpenFE itself: online path + offline training pipeline
  benchmark/      research harness + the compare() API
scripts/        production CLI entrypoints — see scripts/README.md
dev/            one-off smoke/sanity utilities, not production — see dev/README.md
tests/          pytest suite, offline/no-network
docs/           how-to guides (setup, running the benchmark, ctl script usage)
research/       source papers backing algorithm_plan.md's design choices
data/, results/, models/  gitignored: dataset cache, run outputs/logs, trained models
```

```bash
pytest tests/ -q   # offline, no network
```

- [`afe/README.md`](afe/README.md) — the package's own internal layout.
- [`docs/dataset_setup.md`](docs/dataset_setup.md) — standing up the
  benchmark suite + corpus from scratch.
- [`docs/benchmark_ctl_usage.md`](docs/benchmark_ctl_usage.md) — running a
  long benchmark in the background.
- [`synthesis_report.md`](synthesis_report.md) — the literature synthesis
  behind `algorithm_plan.md`'s design choices.
