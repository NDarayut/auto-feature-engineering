# Benchmark results: OpenFE vs. CAFEM

Run: 2026-07-26, `results/benchmark_openfe_cafem.jsonl`, produced by

```bash
python -m scripts.run_benchmark \
    --datasets german-credit heart-disease breast-cancer-wisconsin \
               concrete-strength qsar-biodegradation house-prices \
               telecom-churn california-housing bank-marketing electricity \
    --methods baseline openfe cafem --budget 900 \
    --out results/benchmark_openfe_cafem.jsonl
```

## Setup

- **Methods.** `baseline` (raw prepped features), `openfe`
  (`OpenFEMethod`, default settings: full candidate expansion, keeps top 10
  features, `n_jobs=1`), `cafem` (`CAFEMMethod`, default settings: 40 DQN
  episodes, chain depth ≤ 2, keeps chains with positive wrapper improvement,
  up to 20 features).
- **Suite.** The 10 small/medium datasets of the frozen benchmark (5
  classification-small, 3 medium classification, 2 regression-small/medium +
  house-prices). The large datasets (≥500k rows) were excluded from this run
  for compute reasons; the harness supports them identically.
- **Protocol.** As described in the README: one frozen, seeded 80/20 split
  per dataset, identical for all methods; generation in a spawned subprocess
  under a 900 s budget (no method timed out); scores from the 3-family model
  panel on the held-out fold.
- **Metrics.** AUC for binary classification, macro one-vs-rest AUC for
  heart-disease (multiclass), R² for regression. Means across a column mix
  metrics (AUC and R²), so treat them as a coarse direction signal only —
  per-dataset rows are the real comparison.

## Scores

### Tree panel (LightGBM)

| dataset | metric | baseline | openfe | cafem |
|---|---|---|---|---|
| bank-marketing | auc | 0.9336 | 0.9233 | 0.9336 |
| breast-cancer-wisconsin | auc | 0.9977 | **0.9997** | 0.9983 |
| california-housing | r2 | 0.8547 | **0.8600** | 0.8356 |
| concrete-strength | r2 | 0.9351 | 0.9328 | 0.9351 |
| electricity | auc | 0.9685 | 0.9540 | 0.9685 |
| german-credit | auc | 0.8287 | 0.8120 | **0.8295** |
| heart-disease | auc_ovr | 0.8094 | **0.8169** | 0.8101 |
| house-prices | r2 | 0.5823 | 0.5741 | **0.5943** |
| qsar-biodegradation | auc | 0.9366 | 0.9335 | 0.9366 |
| telecom-churn | auc | 0.8259 | 0.7933 | **0.8300** |
| *mean* | | *0.8672* | *0.8600* | *0.8672* |

### Linear panel (LogisticRegression / Ridge)

| dataset | metric | baseline | openfe | cafem |
|---|---|---|---|---|
| bank-marketing | auc | 0.8627 | **0.8937** | 0.8627 |
| breast-cancer-wisconsin | auc | 0.9936 | **0.9950** | 0.9936 |
| california-housing | r2 | 0.6142 | **0.6818** | 0.6201 |
| concrete-strength | r2 | 0.6474 | **0.6600** | 0.6474 |
| electricity | auc | 0.8159 | **0.8323** | 0.8159 |
| german-credit | auc | 0.8146 | **0.8257** | 0.8135 |
| heart-disease | auc_ovr | 0.7909 | **0.7987** | 0.7898 |
| house-prices | r2 | −0.6589 | **−0.2948** | −0.4774 |
| qsar-biodegradation | auc | 0.9314 | 0.9304 | 0.9314 |
| telecom-churn | auc | 0.8480 | 0.8110 | **0.8494** |
| *mean* | | *0.6660* | *0.7134* | *0.6846* |

### kNN panel (k=10)

| dataset | metric | baseline | openfe | cafem |
|---|---|---|---|---|
| bank-marketing | auc | 0.8495 | **0.8848** | 0.8495 |
| breast-cancer-wisconsin | auc | 0.9993 | **1.0000** | 0.9993 |
| california-housing | r2 | 0.7046 | **0.7903** | 0.7379 |
| concrete-strength | r2 | 0.7156 | **0.7398** | 0.7156 |
| electricity | auc | 0.9050 | 0.8070 | 0.9050 |
| german-credit | auc | 0.7804 | **0.7810** | 0.7590 |
| heart-disease | auc_ovr | 0.7639 | **0.7987** | 0.7537 |
| house-prices | r2 | 0.6554 | **0.6776** | 0.6444 |
| qsar-biodegradation | auc | 0.9054 | 0.9022 | 0.9054 |
| telecom-churn | auc | 0.7902 | **0.8029** | 0.7959 |
| *mean* | | *0.8069* | *0.8184* | *0.8066* |

## Cost (feature-generation wall-clock, features added)

| dataset | openfe | cafem |
|---|---|---|
| bank-marketing | 179.2 s (+10) | 1.0 s (+0) |
| breast-cancer-wisconsin | 15.3 s (+10) | 1.5 s (+6) |
| california-housing | 16.7 s (+10) | 1.0 s (+5) |
| concrete-strength | 7.6 s (+10) | 0.6 s (+0) |
| electricity | 42.3 s (+10) | 1.0 s (+0) |
| german-credit | 24.4 s (+10) | 0.8 s (+1) |
| heart-disease | 8.8 s (+1) | 0.8 s (+1) |
| house-prices | 210.7 s (+10) | 2.7 s (+5) |
| qsar-biodegradation | 45.1 s (+10) | 2.0 s (+0) |
| telecom-churn | 38.7 s (+10) | 1.1 s (+2) |
| **median** | **31.6 s** | **1.0 s** |

Baseline generation is 0 s by definition. No timeouts, no crashes: 90/90
(dataset, method, model-family) cells completed.

## Reading the results

**OpenFE helps weaker model families far more than it helps boosted trees.**
On the linear panel OpenFE improves 8/10 datasets (mean +0.047) and on kNN
7/10 (mean +0.011) — engineered arithmetic/groupby features hand a linear
model non-linearities it cannot represent itself. On the tree panel the
picture inverts: 3 wins, 7 losses/ties (mean −0.007). LightGBM can already
express most of what OpenFE's top-10 features add, and on
telecom-churn/electricity/bank-marketing the fixed "keep 10 features" setting
actively hurt — the selection kept features that didn't survive the holdout.
This mirrors the OpenFE paper's own framing: its gains concentrate where the
downstream learner is weak or the interaction is real (e.g. california-housing,
a known feature-interaction dataset, improves for every family).

**CAFEM is drastically cheaper and much more conservative.** Median
generation cost is ~1 s versus ~32 s (and up to 211 s) for OpenFE, because
the RL search evaluates unary chains on a 2 000-row subsample. Its
positive-delta filter kept *zero* features on 4/10 datasets — on those it
degrades to the baseline by construction, never below it. Where it does add
features, effects are small: it is the only method that beat baseline on
tree-panel telecom-churn (+0.004) and house-prices (+0.012), but it slightly
hurt kNN on german-credit/heart-disease. The gap to OpenFE is structural:
CAFEM's operator set here is unary only (log/sqrt/square/reciprocal/sigmoid/
tanh/zscore/minmax), so it cannot build the feature *interactions* (ratios,
products, groupby aggregates) that drive OpenFE's linear/kNN wins.

**Takeaways.**

1. If the downstream model is a boosted tree, neither method reliably beats
   the raw features on these small/medium datasets — verify before you pay
   the generation cost. (This motivates MF-OpenFE's meta-filter + verify
   stages.)
2. If the downstream model is linear or distance-based, OpenFE is clearly
   worth its cost (linear mean +0.047).
3. CAFEM as a per-dataset method is a cost/safety tradeoff: ~30× cheaper than
   OpenFE and near-harmless by construction, but its unary-only search rarely
   finds large wins. Its real value in this project is offline — the same
   search generates the labeled tuples MF-OpenFE's meta-model trains on.

## Caveats

- One fixed split per dataset — point estimates, no variance/significance
  (the suite's deliberate compute tradeoff; see `draft_plan.md` §5.3).
- Default hyperparameters for both methods; OpenFE's `n_new_features=10` and
  CAFEM's `episodes=40, max_depth=2` were not tuned per dataset.
- Small/medium datasets only; the 6 large benchmark datasets (≥500 k rows)
  were not part of this run.
- Mean rows mix AUC and R² and are directional only.

Raw rows: `results/benchmark_openfe_cafem.jsonl` (one JSON object per
(dataset, method, model-family), including errors had there been any).
Regenerate the aggregate report with
`python -m scripts.report_benchmark results/benchmark_openfe_cafem.jsonl`.
