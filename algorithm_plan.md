# AutoFE Algorithm Design: MF-OpenFE (Meta-Filtered OpenFE)

## Context
Goal: design a new AutoFE method, to build and benchmark later against `draft_plan.md`,
optimized for **practicality/usability over novelty**. Constraint: RL is acceptable only in a
**one-time offline** step, never at online usage time (per-dataset inference must be fast and
predictable, no RL search or reward tuning at usage time). Every technique below is attributed
to its source paper. This is a design-only document — no code has been written yet.

---

## 1. Design principle: RL offline, supervised-only online

Split the pipeline into two lifecycles:
- **Offline (one-time, no time pressure, RL acceptable):** build the meta-training corpus and
  train the meta-model. Training instability/slow convergence here is a one-time engineering
  cost, not a per-dataset reliability risk.
- **Online (every time a user runs this on a new dataset, must be fast and predictable):** pure
  supervised-model inference plus OpenFE's existing deterministic machinery. No RL, no search
  loop, no reward tuning at usage time.

This lets the design use CAFEM's RL-based search to produce **higher-quality meta-training
labels** than a cheap heuristic would, while keeping the deployed/online path fully predictable.

## 2. Algorithm specification

**Stage 0 — Offline label generation via RL search** *(one-time, per historical dataset in the
meta-training corpus)*
- Method: per-feature RL agent (Double DQN) navigating a **Feature Transformation Graph**,
  where each node is a candidate feature and each action applies an operator — thoroughly
  explores the transformation space per historical dataset, using wrapper CV performance as
  reward.
- **Paper:** CAFEM — *"Cross-data Automatic Feature Engineering via Meta-learning and
  Reinforcement Learning"*, PAKDD 2020 (`research/Cross-data Automatic Feature.pdf`).
- Output: for each historical dataset, a set of (transformation, meta-features, usefulness
  label) tuples — richer/more thorough than what a fixed-budget statistical filter would find,
  because RL search isn't capped to one pass over a fixed operator library.
- Optional refinement: CAFEM's MAML-style cross-dataset transfer step can be used here too, to
  make the RL policy itself generalize faster across the corpus during offline training — still
  entirely offline, never invoked at usage time.

**Stage 1 — Offline meta-model training** *(one-time, on Stage 0's output)*
- Method: train a supervised classifier/regressor (meta-features → predicted usefulness) on the
  labeled corpus from Stage 0, using a QSA-style distributional feature sketch as input.
- **Paper:** LFE — *"Learning Feature Engineering for Classification"*, IJCAI 2017
  (`research/Learning Feature Engineering for Classification.pdf`) for the meta-feature
  sketch/per-operator classifier design; GELFE — *"Automated Feature Engineering for Automated
  Machine Learning"*, Knowledge-Based Systems 2025
  (`research/_Knowledge_Based_Systems__Automated_Feature_Engineering_for_Automated_Machine_Learning11.pdf`)
  for the multi-classifier meta-target formulation and CIFE-based redundancy handling; the
  2026 meta-learning paper (`research/Automated_Feature_Engineering_Using_Meta-Learning_.pdf`)
  for the general task–transformation–performance matrix framing.
- Output artifact: a plain RF/GBM classifier or regressor — small, fast, inspectable. This is
  the only meta-learning artifact used online; RL never runs again after this point.

**Stage 2 — Offline gatekeeper training** *(one-time, on corpus-level outcomes)*
- Method: supervised model on (dataset-level meta-features) → (did AutoFE help at all,
  historically, and by how much). Motivated by two papers' own reported failure analyses:
  OpenFE found no significant gain on 19/68 of its benchmark datasets, and GELFE reports cases
  of AutoFE underperforming no-AutoFE (e.g. the "electricity" dataset).
- **Papers:** OpenFE — *"OpenFE: Automated Feature Generation with Expert-level Performance"*,
  ICML 2023 (`research/openfe.pdf`) §6 (empirical failure cases); GELFE (KBS 2025, as above)
  §6 (reported underperformance cases) — used here as the motivating evidence, not as a
  technique either paper itself proposes.
- Output: a small classifier that runs before Stage 3, to skip generation entirely on datasets
  predicted not to benefit.

**Stage 3 — Online generation** *(every new dataset, runs live)*
- Method: unchanged — OpenFE's fixed operator library (arithmetic, groupby aggregations, etc.)
  via expansion-reduction candidate generation.
- **Paper:** OpenFE, ICML 2023 (as above), §3.

**Stage 4 — Online meta-filter** *(every new dataset, runs live, no training)*
- Method: apply Stage 1's trained meta-model to rank/prune Stage 3's candidate pool before it
  reaches OpenFE's own evaluation stage — pure inference, milliseconds per candidate.
- **Papers:** LFE (IJCAI 2017) and GELFE (KBS 2025), as above — same artifact as Stage 1,
  applied at inference time.

**Stage 5 — Online verify** *(every new dataset, runs live)*
- Method: unchanged — OpenFE's FeatureBoost residual-fitting (avoids full retraining) plus
  successive-halving pruning over data blocks, now run on the smaller, meta-filtered pool.
- **Paper:** OpenFE, ICML 2023 (as above), §4.

**Stage 6 — Online select** *(every new dataset, runs live)*
- Method: unchanged — OpenFE's MDI/SHAP-based interaction attribution and final pruning.
- **Paper:** OpenFE, ICML 2023 (as above), §4.5.

## 3. Meta-training corpus requirements

- **Breadth:** span multiple sectors and both task types (classification/regression), reusing
  the same sector-diverse dataset pool being assembled for `draft_plan.md`'s benchmark — but
  the meta-training split and the benchmark evaluation split must be **disjoint**. Training the
  meta-model on datasets it's later "graded" on would contaminate the benchmark results. This
  is a hard requirement.
- **Size:** every meta-learning paper in the set (LFE, GELFE, CAFEM, 2026 paper) flags that
  transfer quality is capped by corpus representativeness — prioritize more datasets over more
  exhaustive search per dataset.
- **Cost budget for Stage 0:** RL search per historical dataset is the most expensive offline
  step; since it only runs once per corpus dataset (not per benchmark run), this cost is
  acceptable and bounded by corpus size, not by online usage frequency.

## 4. Why this fits the practicality constraint

- **Online path is 100% deterministic/supervised** (Stages 3–6 are OpenFE's existing code plus
  two trained-and-frozen sklearn-style models) — no training instability at usage time, which
  was the actual concern behind "avoid RL."
- **RL is isolated to Stage 0**, a one-time corpus-labeling step with no latency requirement and
  no per-user-run repetition — the classic case where RL's slow/unstable convergence is an
  acceptable one-time cost rather than a recurring risk.
- **Every technique is attributed to its source paper** (table below) so implementation can
  follow each paper's method directly rather than reinventing details.

| Stage | Technique | Paper | File |
|---|---|---|---|
| 0 | RL search (DDQN, Feature Transformation Graph) | CAFEM, PAKDD 2020 | `research/Cross-data Automatic Feature.pdf` |
| 0 (optional) | MAML cross-dataset policy transfer | CAFEM, PAKDD 2020 | `research/Cross-data Automatic Feature.pdf` |
| 1 | Meta-feature sketch (QSA) + per-operator classifier | LFE, IJCAI 2017 | `research/Learning Feature Engineering for Classification.pdf` |
| 1 | Multi-classifier meta-target + CIFE redundancy handling | GELFE, KBS 2025 | `research/_Knowledge_Based_Systems__Automated_Feature_Engineering_for_Automated_Machine_Learning11.pdf` |
| 1 | Task–transformation–performance matrix framing | Meta-Learning-2026 | `research/Automated_Feature_Engineering_Using_Meta-Learning_.pdf` |
| 2 | Failure-case motivation (gatekeeper) | OpenFE §6; GELFE §6 | `research/openfe.pdf`; KBS file above |
| 3, 5, 6 | Expansion-reduction generation, FeatureBoost, successive halving, MDI/SHAP selection | OpenFE, ICML 2023 | `research/openfe.pdf` |

## 5. Explicitly out of scope for this design pass
- No LLM-based AutoFE (FELIX/CAAFE) — flagged in the synthesis as emerging but no reference
  implementation exists in the research folder.
- No implementation yet — this pass is the algorithm/technique specification only.
