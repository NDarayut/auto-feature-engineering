# AutoFE Benchmark Plan

Goal: compare automatic feature engineering (AutoFE) methods fairly, so that
observed differences reflect the quality of the generated features and not
an artifact of the benchmark setup (model choice, dataset selection,
encoding, or randomness).

## 1. Design Principles

- **Isolate the variable under test.** The only thing that should differ
  between arms of a comparison is *which method generated the features*.
  Model, preprocessing, CV splits, and compute budget must be held constant
  within a comparison.
- **No leakage.** Every AutoFE method is fit (including any target-aware
  transforms, e.g. target encoding, WoE, meta-learned rankers) only on the
  training fold. Feature generation and feature selection happen inside the
  cross-validation loop, not before it.
- **Report distributions, not points.** Every number in the results is a
  mean ± spread over multiple seeds/folds, not a single run.

## 2. Datasets

Assemble a benchmark suite spanning:

- **Task type:** classification and regression, kept as the top-level split
  for all results (a method that wins on regression may not win on
  classification, and metrics aren't comparable across them).
- **Sector:** at minimum healthcare, finance, and one physical-science
  domain (e.g. chemistry/materials), plus a general/tabular-ML catch-all
  (e.g. UCI/OpenML/Kaggle staples) as a baseline sector. This lets us
  distinguish "good in general" from "good only on sector X."
- **Scale:** include a mix of row counts and feature counts (small/medium/
  large) since several AutoFE methods (e.g. expansion-reduction approaches)
  scale poorly and may time out or degrade on wide/large tables — that is
  itself a result worth capturing, not something to filter out up front.
- **Known ground truth where possible.** Prefer at least a few datasets
  with documented domain-expert feature engineering (e.g. Kaggle
  competition winner writeups) so generated features can be sanity-checked
  against known-useful engineered features.

For each dataset, fix and record: target definition, train/test split
(or CV protocol), and any dataset-specific preprocessing decided once and
reused across all methods.

## 3. Feature Generation Protocol

For each (dataset, AutoFE method) pair:

1. Generate a **baseline** feature set: the raw/minimally-cleaned features
   only (no AutoFE), used as the control arm.
2. Run each AutoFE method under a **fixed compute budget** (wall-clock time
   and/or max evaluations) so methods aren't compared at different effort
   levels. Also record actual wall-clock time and peak memory — this is a
   benchmark output, not just a control.
3. Record, per method: number of features generated, number of features
   surviving the method's own selection step (if any), and generation time.

Do not let a downstream model's own feature importance influence which
AutoFE method "wins" during generation — that judgment happens only at
evaluation time (§5), uniformly across methods.

## 4. Preprocessing & Encoding (applied uniformly per model family)

Different downstream models need different encodings, and using the wrong
one for the wrong model would penalize the *encoding choice*, not the
features. Fix one encoding pipeline per model family, applied identically
across all AutoFE methods and the baseline:

- **Tree-based models (XGBoost/LightGBM/CatBoost, Random Forest):**
  native categorical handling where supported, otherwise ordinal/target
  encoding fit on the training fold only; missing values passed through
  natively or via a sentinel value.
- **Linear/logistic & distance-based models (Logistic/Linear Regression,
  SVM, kNN):** one-hot or WoE encoding for categoricals, standard/robust
  scaling for numerics, explicit missing-value imputation (fit on train
  fold only).
- **Neural network baselines (optional, if included):** embeddings for
  high-cardinality categoricals, standard scaling for numerics.

All encoders/imputers/scalers are fit inside the CV loop (train fold only)
and applied to the held-out fold — never fit on the full dataset.

## 5. Evaluation

### 5.1 Model diversity
Evaluate every generated feature set on a small panel of models spanning
different inductive biases, not just one gradient-boosted tree model:
- One boosted tree model (e.g. XGBoost or LightGBM)
- One linear model (Logistic/Linear Regression, regularized)
- One non-tree, non-linear model (e.g. kNN or a shallow MLP)

A method that only helps tree models isn't "good," it's tree-specific —
report both the per-model results and the aggregate, so this distinction
is visible rather than averaged away.

### 5.2 Metrics
For each (dataset, method, model) combination, report:
- **Predictive performance:** task-appropriate metric(s) — e.g. AUC/
  accuracy/F1 for classification, RMSE/MAE/R² for regression — plus the
  *delta* vs. the no-AutoFE baseline on the same model.
- **Feature yield:** number of features generated and number retained
  after selection.
- **Feature efficiency:** fraction of generated features that are
  individually predictive (e.g. by permutation importance or a fast
  univariate score exceeding a fixed threshold) — this operationalizes
  "many features that are also good," not just "many features."
- **Compute cost:** generation time, and (if applicable) inference-time
  cost of computing the features on new data.

A method is preferred only if it improves performance-per-compute-budget
over the baseline, not merely if it adds volume.

### 5.3 Statistical rigor
- Evaluate every (dataset, method, model) combination across **5
  independent random seeds** (data split/shuffle/init seed) or **k-fold
  cross-validation (k=5)** — pick one protocol per dataset based on its
  size (CV for small datasets, seed-repeats for large ones where CV is
  too slow) and apply it consistently to all methods on that dataset.
- Report mean ± standard deviation, and run a paired significance test
  (e.g. paired t-test or Wilcoxon signed-rank across seeds/folds) between
  each AutoFE method and the baseline, and between top-performing methods,
  rather than declaring a winner from means alone.

## 6. Aggregation & Reporting

- Report results first split by **task type** (classification vs.
  regression), then by **sector**, then aggregated overall — never
  aggregate classification and regression metrics together.
- For each split, show: baseline vs. each method's performance delta,
  feature yield, feature efficiency, and compute cost, plus significance
  markers.
- Explicitly surface sector-specific winners vs. a generalist winner (best
  average rank across all sectors) — these can legitimately be different
  methods, and the report should say so rather than picking one "best"
  method.
- Include failure cases (timeouts, crashes, methods that couldn't run on a
  given dataset) in the report rather than silently excluding them —
  robustness is part of the comparison.

## 7. Known Pitfalls to Guard Against

- **Leakage via feature selection outside CV** — selecting features on the
  full dataset before splitting will inflate every method's score equally
  but hide real differences; always select inside the fold.
- **Budget mismatch** — an unbounded search method will look better than a
  time-boxed one for reasons unrelated to feature quality; enforce the
  same compute budget (§3.2).
- **Encoding mismatch** — comparing a method that natively handles
  categoricals against one that doesn't, without controlling for encoding,
  conflates encoding quality with feature quality (§4).
- **Single-model overfitting to a benchmark** — optimizing the benchmark
  design around one downstream model's quirks; the model panel (§5.1)
  exists specifically to catch this.
