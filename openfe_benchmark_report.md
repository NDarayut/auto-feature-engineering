# Benchmark Report

## Datasets

| abbrev | dataset | task | sector | metric |
|---|---|---|---|---|
| GC | german-credit | classification | finance | auc |
| CS | concrete-strength | regression | physical/materials | r2 |

## Overview

Mean held-out score across the three model families (per-family scores are fold-means; metrics differ per dataset -- see the legend).

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

## Speed (feature-generation wall-time)

| method | GC | CS | median |
|---|---|---|---|
| baseline | 0.0 s | 0.0 s | 0.0 s |
| openfe | 44.9 s | 6.9 s | 25.9 s |

## Feature counts (before -> after)

Number of columns fed into the downstream models before (original, post max-cols-cap) and after a method's generated features are added.

| method | GC | CS |
|---|---|---|
| baseline | 20 -> 20 | 8 -> 8 |
| openfe | 20 -> 30 | 8 -> 18 |

## By task

Mean of the per-dataset overview score (mean across model families) over each dataset's group; a method missing on every dataset in a group shows —.

| method | classification | regression |
|---|---|---|
| baseline | 0.808 | 0.766 |
| openfe | **0.810** | **0.848** |

## By sector

Mean of the per-dataset overview score (mean across model families) over each dataset's group; a method missing on every dataset in a group shows —.

| method | finance | physical/materials |
|---|---|---|
| baseline | 0.808 | 0.766 |
| openfe | **0.810** | **0.848** |

## Failures / timeouts / crashes

(none)