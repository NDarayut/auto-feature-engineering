# Benchmark Guide: End-to-End Flow

The single reference for how a dataset goes from "raw source" to "a row in a
benchmark report" in this repo. `algorithm_plan.md` explains *why* the
project is designed this way (MF-OpenFE's staged approach); `draft_plan.md`
is the approved benchmark *methodology* this pipeline implements. This doc is
the *how* — the concrete code path, one document, start to finish.

```
registry.py --> download.py --> splits.py --> methods.py --> models.py --> benchmark.py --> report_benchmark.py
 (which          (fetch+cache      (frozen single   (AutoFE      (model        (orchestrate,   (aggregate JSONL
 datasets)        raw data)        fixed-seed       method       panel:        one dataset at   into a report:
                                    train/test       adapters)    tree/linear/  a time: budget-  point delta vs
                                    split)                        knn)         limited          baseline per
                                                                                generation +     dataset/method/
                                                                                scoring, JSONL   model)
                                                                                output, free
                                                                                memory, next
                                                                                dataset)
```

## 1. Environment

```bash
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
```

`requirements.txt` pins `scikit-learn>=1.3,<1.6`: `>=1.3` for
`sklearn.preprocessing.TargetEncoder` (used by `afe/encoders.py` and
`afe/methods.py`), `<1.6` because `openfe` 0.0.12 calls the removed
`mean_squared_error(..., squared=False)` argument internally — sklearn 1.6+
breaks it. If openfe is dropped from the method list in the future, this
ceiling can be relaxed.

Kaggle-sourced datasets (4 of 22) need `~/.kaggle/kaggle.json` and each
competition's rules accepted once in-browser — see `docs/dataset_setup.md`
for the full walkthrough; this doc assumes the datasets are already fetched.

## 2. Dataset sourcing (`afe/registry.py`, `afe/download.py`)

`afe/registry.py` declares `BENCHMARK`: 22 `DatasetSpec`s (key, task,
sector, scale, source, target column, ...) spanning classification/
regression/multiclass, 4+ sectors, and small/medium/large scale — the
coverage grid `draft_plan.md` §2 requires. A disjoint `CORPUS_SUITES` (~100
OpenML datasets) is the meta-training corpus for the project's own
algorithm work; it's not touched by the benchmark harness.

`afe/download.py`'s `load(spec)` fetches (or reads the cache) one dataset:
a raw, unsplit, unencoded `(DataFrame, meta_dict)`. Cached at
`data/cache/<key>.parquet`. Full details, including the 3 manual-CSV and 4
Kaggle datasets: `docs/dataset_setup.md`.

Frozen manifests (regenerate with `python -m afe.manifests`):
`afe/manifests/benchmark.json`, `afe/manifests/corpus.json`.

## 3. Split protocol (`afe/splits.py`)

Per `draft_plan.md` §2/§5.3, every dataset gets **one fixed train/test
split**, decided once and reused identically by every method: a single
fixed-seed **80/20 holdout split**, the same protocol regardless of dataset
scale. This is a deliberate simplification over CV/seed-repeats — one point
estimate per (dataset, method, model), not a distribution — trading
statistical rigor for much lower compute and memory cost.

Classification/multiclass splits are class-stratified; regression splits are
plain random. The *recipe* (protocol, per-dataset seed, test size) is frozen
to `afe/manifests/splits.json` via `python -m afe.splits` — not materialized
row indices, since regenerating them from a fixed seed
(`numpy.random.RandomState`, whose bit-stream is version-stable) is cheap
and keeps the manifest metadata-sized. `iter_split_indices(spec, n, y)`
yields a single `("split0", train_idx, test_idx)`.

## 4. Per-model-family encoding (`afe/encoders.py`)

Two profiles, per `draft_plan.md` §4, each fit on the training fold only:

- **`TreeEncoder`** (boosted trees/RF): ordinal-coded categoricals by
  default; NaN passes through untouched (LightGBM/XGBoost/CatBoost handle
  it natively). `categorical_encoding="target"` swaps in
  `sklearn.preprocessing.TargetEncoder` (leak-safe internal cross-fitting).
- **`LinearEncoder`** (linear/logistic/SVM/kNN): one-hot categoricals +
  standard-scaled, median/most-frequent-imputed numerics by default;
  `categorical_encoding="target"` swaps one-hot for target encoding (this
  profile's answer to §4's "one-hot or WoE" allowance).

`afe/eval_data.py`'s `iter_folds(key, encoding="tree"|"linear")` ties
`load()` + `splits` + `encoders` into ready-to-train folds — the contract
any *non-AutoFE* consumer (e.g. a plain model comparison) can use directly.
The benchmark harness (`afe/benchmark.py`) uses the lower-level pieces
directly instead, because it needs the *raw* fold to hand to AutoFE methods
before any model-family encoding is applied (see §6).

## 5. AutoFE methods (`afe/methods.py`)

Every method implements `fit_transform(X_train, y_train, task) -> DataFrame`
(train fold only) / `transform(X_test) -> DataFrame` (held-out fold, using
state fit in step one — never refit). Inputs are first normalized by
`prep_for_generation()`: ordinal-encode categoricals + median-impute
numerics, fit on train only — this hides each library's differing
raw-dtype/NaN support from the benchmark loop.

| method | `name` | library | notes |
|---|---|---|---|
| No-op baseline | `baseline` | — | passthrough; the reference every method is compared against |
| OpenFE | `openfe` | `openfe` (pip) | `OpenFE().fit()` + `transform()`; top 10 generated features kept |
| Featuretools | `featuretools` | `featuretools` | single-table DFS (`add/subtract/multiply/divide_numeric`, depth 1 — no relational structure available) |
| Autofeat | `autofeat` | `autofeat` | `AutoFeatRegressor`/`AutoFeatClassifier`, 1 feature-engineering step |

All three were smoke-tested end-to-end on `concrete-strength` (small,
regression) and produce sane, improving-over-baseline results for the
linear/kNN families (trees are already close to raw-feature ceiling on this
dataset, so smaller/no tree lift is expected, not a bug).

## 6. The 3-model panel (`afe/models.py`)

Per `draft_plan.md` §5.1: one boosted-tree model, one linear model, one
non-tree/non-linear (distance-based) model.

| family | model (regression) | model (classification) | encoding used |
|---|---|---|---|
| `tree` | `LGBMRegressor` | `LGBMClassifier` | `TreeEncoder` |
| `linear` | `Ridge` | `LogisticRegression` | `LinearEncoder` |
| `knn` | `KNeighborsRegressor` | `KNeighborsClassifier` | `LinearEncoder` |

`prepare_family_input()` does post-generation cleanup fit on train only:
trees get `+/-inf → NaN` (generated ratios/logs can produce inf; LightGBM
handles NaN, not inf); linear/kNN additionally get median-imputed +
standard-scaled. `fit_and_score()` reports R²+MAE (regression) or
AUC/macro-AUC-OvR (classification).

## 7. Running the benchmark (`afe/benchmark.py`, `scripts/run_benchmark.py`)

```bash
python -m scripts.run_benchmark --datasets nomao concrete-strength \
    --methods baseline openfe featuretools autofeat --budget 300

# everything, sequential, resumable
python -m scripts.run_benchmark --methods baseline openfe featuretools autofeat \
    --budget 300

# only the tree model, to cut runtime/memory if linear/knn aren't needed yet
python -m scripts.run_benchmark --methods baseline openfe featuretools autofeat \
    --budget 300 --models tree
```

**Execution is strictly sequential, one dataset at a time** — never
concurrent. For each dataset, `run_dataset()` loads it once, computes its
single split once, then loops over every requested method against that same
split; once every method's rows for that dataset are written, the dataset's
in-memory objects are explicitly dropped and garbage-collected before moving
to the next dataset (`afe/benchmark.py`'s `run_dataset`). This is what bounds
peak memory to roughly one dataset's worth at a time, instead of scaling with
a worker count — there is no `--workers` flag.

For each `(dataset, method)`, the AutoFE method's `fit_transform`/`transform`
step still runs in a **subprocess with a hard wall-clock budget** (`--budget`
seconds, default 300) — per `draft_plan.md` §3's fixed-compute-budget
requirement, and to guarantee that whatever memory featuretools/autofeat/
OpenFE use internally is released the instant that subprocess exits, rather
than accumulating across methods in the main process. If generation doesn't
finish in time, the subprocess is killed and the row is recorded as a
`timeout`, not silently dropped (§6: "failure cases/timeouts/crashes
included"). The subprocess uses `multiprocessing`'s **spawn** context, not
fork: the parent has already imported lightgbm/openfe/featuretools/autofeat,
which spin up native BLAS/numba worker threads, and forking a multi-threaded
process risks the child deadlocking on a lock held by a thread that doesn't
exist post-fork.

**Resumable by default**: before starting, `_completed_pairs()` scans the
output file and skips any `(dataset, method)` pair that already has rows —
pass `--no-resume` to force a clean re-run.

> **Implementation notes (already fixed, documented so they aren't
> reintroduced):**
> - The generation subprocess's result is read via `queue.get(timeout=...)`
>   **before** `proc.join()`. Doing it in the other order can deadlock: if
>   the result is large enough to exceed the OS pipe buffer, the child
>   blocks writing to the queue while the parent blocks joining, and neither
>   proceeds until the outer timeout fires.
> - `openfe`'s `fit()`/`transform()` write to **hardcoded, relative** temp
>   filenames (`transform()`'s `./openfe_tmp_data.feather` has no override
>   parameter at all). Concurrent openfe calls from different processes can
>   collide on that shared path and corrupt each other's temp file (surfaced
>   as `status: "crashed"` with no error message, since the subprocess died
>   mid-write, not mid-`except`). Fixed by `chdir`-ing into a fresh temp
>   directory for the duration of each openfe call (`afe/methods.py`'s
>   `_IsolatedCwd`) — still relevant even though execution is sequential now,
>   since a previous run's leftover temp file could otherwise collide with
>   the current one.
> - `autofeat`'s `fit_transform`/`transform` return a DataFrame with its
>   *own* (reset) index. Wrapping that as `pd.DataFrame(out, index=X_train.index)`
>   performs a **label-based reindex**, not a positional relabel — since the
>   label sets differ, this silently scrambled rows and produced garbage
>   (even negative) R²/AUC with no error or warning. Fixed by going through
>   a bare `np.asarray(out)` first so the caller's index is applied
>   positionally (`AutofeatMethod._to_frame`).

One JSONL row per `(dataset, method, model_family)` is appended to
`results/benchmark_results.jsonl` (or `--out`): `status` (`ok`/`timeout`/
`error`/`model_error`/`crashed`), `metric`/`value`, `gen_elapsed_s`,
`n_features_generated`, `n_features_final`.

**Compute-budget reality check:** with 22 datasets × 4 AutoFE methods run
sequentially, a run that hits the full `--budget` on every timeout can still
take a long time. `scripts/run_benchmark.py` orders datasets
smallest-scale-first by default so bugs and per-method behavior surface early
and cheaply, before the run reaches the 300k–700k-row datasets; use
`--datasets`/`--methods`/`--models` to scope a first pass to a subset before
committing to the full run.

**Memory** is bounded by one dataset at a time, not a worker count: since
execution is strictly sequential (§7), only one dataset's data plus one
generation subprocess are ever resident together, and `run_dataset()` drops
and garbage-collects the dataset's objects before moving on. featuretools/
autofeat can still individually use anywhere from a few hundred MB to 20+ GB
on the larger datasets — that's the underlying libraries' own memory use
during generation, not something the harness can shrink — but it's no longer
multiplied by concurrent workers. If you see `crashed` rows, that's the
kernel OOM-killer taking out that one dataset/method's generation subprocess;
it's resumable, so re-running costs time, not lost work.

### `scripts/benchmark_ctl.sh` — run and monitor it yourself

A thin wrapper around the above so you don't need to babysit a foreground
terminal or reconstruct the `nohup`/PID-tracking dance by hand. Full usage
guide: **`docs/benchmark_ctl_usage.md`**.

```bash
scripts/benchmark_ctl.sh start -- --budget 300
scripts/benchmark_ctl.sh status   # row/status counts + running/not
scripts/benchmark_ctl.sh tail     # follow the live log (Ctrl-C just stops watching)
scripts/benchmark_ctl.sh stop     # stop it; progress is preserved, resumable later
scripts/benchmark_ctl.sh report -- --out results/report.md
```

It runs the job with bash job control (`set -m`) so the whole process tree
(including the nested feature-generation subprocess) shares one process
group, and tracks that group's PGID in `results/benchmark.pid`. `stop` kills
the **entire group** at once (`kill -- -PGID`), not just the top process --
killing only the top process (e.g. a bare `kill $PID`, or `timeout` wrapping
a command with `| tail`) leaves the nested `spawn`-created subprocess
orphaned and still consuming memory, which is exactly what happened during
development before this was fixed.

## 8. Reading the results (`scripts/report_benchmark.py`)

```bash
python -m scripts.report_benchmark results/benchmark_results.jsonl --out results/report.md
```

Produces the `draft_plan.md` §6 reporting structure: split by task type →
sector → per-dataset/model row, each method shown as its point value with the
delta vs. the baseline on the same fixed-seed split (no mean±std or
significance marker — a single split gives one estimate, not a distribution).
A trailing "Failures / timeouts / crashes" table reports non-`ok` rows by
count instead of excluding them.

## 9. Sanity check (`dev/parity_check.py`)

```bash
python -m dev.parity_check
```

Raw-feature LightGBM baseline via `afe.eval_data.iter_folds`, reporting each
dataset's score on its single fixed-seed split — the fastest way to confirm
the split/encode/fold-iteration path still works after a change, without
running any AutoFE method.

## Known limitations (current state, not final design)

- **AutoFE methods see ordinal-coded categoricals, not native categorical
  types.** `prep_for_generation()` ordinal-encodes before handing data to
  any method, including OpenFE (which can natively distinguish categorical
  columns). This is a simplification to give all three libraries one common
  numeric-only input contract; a future pass could let OpenFE see real
  categorical dtypes.
- **Compute budget default (300s) and OpenFE's `n_new_features=10` /
  featuretools' `max_depth=1` / autofeat's `feateng_steps=1` are reasonable
  first-pass values, not values derived from a tuning study.** Adjust via
  `--budget` and the adapter constructors in `afe/methods.py` as real
  results come in.
- **Model hyperparameters are library defaults** (`LGBMRegressor()`,
  `Ridge()`, `KNeighborsRegressor(n_neighbors=10)`, ...) — no per-dataset
  tuning, consistent with `dev/parity_check.py`'s existing "sanity, not
  a tuned run" precedent.
