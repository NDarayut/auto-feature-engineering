# `afe.meta` — MF-OpenFE

Both halves of MF-OpenFE (`algorithm_plan.md`) live here:

* **Online** (`online.py`, `MFOpenFE`) — runs live, every time, on whatever
  dataframe a caller passes in. This is the public entrypoint, re-exported as
  `afe.MFOpenFE`. See the repo-root [`README.md`](../../README.md) for usage.
* **Offline** (everything else in this package) — a one-time training
  pipeline that produces the `MetaModel` artifact `online.py` loads. Nothing
  here runs at per-dataset usage time.

## Pipeline

```
corpus datasets (OpenML dids, disjoint from benchmark)
        │  corpus_data.load_corpus_dataset()
        ▼
FTGEnvironment ── Feature Transformation Graph over one dataset
        │           state = QSA sketch (meta_features), action = operator,
        │           reward = wrapper Δ from adding the transformed feature
        │  DoubleDQN explores it (ddqn.py)
        ▼
Stage 0  (stage0.py)  ── per dataset: RL search → (sketch, operator, useful) tuples
        │  streamed to results/meta/stage0_tuples.jsonl  (resumable by did)
        ▼
Stage 1  (stage1.py)  ── per-operator RandomForest: sketch → P(useful)
        │  leave-datasets-out eval report
        ▼
models/meta_model.pkl
        │
        ▼  ── online, every new dataset (online.py, MFOpenFE) ──
Stage 3 (generate)       openfe.get_candidate_features() -- OpenFE's own
                         candidate operator library
Stage 4 (meta-filter)    score each candidate whose root operator the
                         meta-model was trained on; pass the rest through
Stage 5+6 (verify+select) one LightGBM fit on raw+survivors, keep by
                         feature importance
```

## Modules
**Offline:**
- `corpus_data.py` — load a corpus dataset by OpenML `did` (cache + numeric prep).
- `operators.py` — frozen unary transformation operator library (the Stage-0 action set).
- `meta_features.py` — LFE-style QSA sketch + scalar meta-features (`SKETCH_DIM`).
- `environment.py` — `FTGEnvironment`: state/step/wrapper-reward for one dataset.
- `ddqn.py` — self-contained numpy Double DQN (no torch).
- `stage0.py` — RL search across the corpus → labeled tuples JSONL.
- `stage1.py` — train + persist the per-operator `MetaModel`.

**Online:**
- `online.py` — `MFOpenFE`: generate (OpenFE) → meta-filter (`MetaModel`) →
  verify+select (LightGBM importance).

## Run

```bash
# Offline, one-time (or to retrain on a different corpus):
pip install "auto-feature-engineering[benchmark]"   # openml/ucimlrepo/kaggle etc.
python -m scripts.run_stage0 --episodes 80          # Stage 0: RL label generation
python -m scripts.train_meta_model                  # Stage 1: train + save the meta-model

# Online, every dataset:
python -c "
from afe import MFOpenFE
mfe = MFOpenFE(task='classification')
X_fe = mfe.fit_transform(X_train, y_train)
"
```

## Test (offline, no network)
```bash
pytest tests/test_meta.py tests/test_online.py -q
```

## Design notes
- **RL is confined to Stage 0** (`algorithm_plan.md` §1): the online path never
  trains or searches. Stage 0's DQN exists only to produce higher-quality
  usefulness labels than a single fixed operator pass would.
- **Scale-invariant state.** The QSA sketch rank-normalizes each feature, so one
  meta-model transfers across datasets of any scale — the LFE premise.
- **Disjointness.** Corpus datasets come from `afe/benchmark/manifests/corpus.json`,
  which `afe.benchmark.manifests` builds with every benchmark dataset removed.
  Training the meta-model on a benchmark dataset would contaminate the
  benchmark (hard rule).
- **Reward = marginal wrapper Δ** of adding one transformed feature to the raw
  set, which is exactly the "is this feature useful" signal Stage 1 learns.
- **Stage 4's meta-filter coverage is partial, by construction.** The trained
  model only has classifiers for `afe.meta.operators`' 8 transformations, of
  which only `log/sqrt/square/sigmoid` also appear as root operators among
  OpenFE's real candidates (verified against the installed `openfe` package).
  Every other candidate passes Stage 4 unfiltered rather than being guessed
  at. See `online.py`'s module docstring and the root README's "How it works"
  section for the full explanation.
- **Stage 2 (gatekeeper) and Stage 5's full FeatureBoost/successive-halving
  are not implemented** — `online.py` always attempts generation, and
  verify+select is a single-fit importance-based simplification. Both are
  documented, deliberate scope cuts, not silent gaps.
