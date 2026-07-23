# `afe.meta` — MF-OpenFE offline meta-learning pipeline

Implements the **offline, one-time** half of MF-OpenFE (`algorithm_plan.md`):
Stage 0 (RL label generation) → Stage 1 (meta-model training). Nothing here
runs at online usage time; the only artifact the online path consumes is the
trained `MetaModel` pickle from Stage 1.

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
models/meta_model.pkl   ← the online Stage-4 filter loads this
```

## Modules
- `corpus_data.py` — load a corpus dataset by OpenML `did` (cache + numeric prep).
- `operators.py` — frozen unary transformation operator library (the action set).
- `meta_features.py` — LFE-style QSA sketch + scalar meta-features (`SKETCH_DIM`).
- `environment.py` — `FTGEnvironment`: state/step/wrapper-reward for one dataset.
- `ddqn.py` — self-contained numpy Double DQN (no torch).
- `stage0.py` — RL search across the corpus → labeled tuples JSONL.
- `stage1.py` — train + persist the per-operator `MetaModel`.

## Run

```bash
# Stage 0 — expensive, one-time; resumable by dataset.
python -m scripts.run_stage0 --limit 5 --episodes 40      # quick corpus slice
python -m scripts.run_stage0 --episodes 80                # full manifest

# Stage 1 — train the meta-model from the tuples.
python -m scripts.train_meta_model
```

## Test (offline, no network)
```bash
pytest tests/test_meta.py -q
```

## Design notes
- **RL is confined to Stage 0** (`algorithm_plan.md` §1): the online path never
  trains or searches. Stage 0's DQN exists only to produce higher-quality
  usefulness labels than a single fixed operator pass would.
- **Scale-invariant state.** The QSA sketch rank-normalizes each feature, so one
  meta-model transfers across datasets of any scale — the LFE premise.
- **Disjointness.** Corpus datasets come from `afe/manifests/corpus.json`, which
  `afe.manifests` builds with every benchmark dataset removed. Training the
  meta-model on a benchmark dataset would contaminate the benchmark (hard rule).
- **Reward = marginal wrapper Δ** of adding one transformed feature to the raw
  set, which is exactly the "is this feature useful" signal Stage 1 learns.
