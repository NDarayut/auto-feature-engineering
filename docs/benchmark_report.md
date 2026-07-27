# Benchmark Report

## Datasets

| abbrev | dataset | task | sector | metric |
|---|---|---|---|---|
| QB | qsar-biodegradation | classification | chemistry | auc |
| E | electricity | classification | energy | auc |
| BM | bank-marketing | classification | finance | auc |
| BPC | bnp-paribas-claims | classification | finance | auc |
| GC | german-credit | classification | finance | auc |
| HCD | home-credit-default | classification | finance | auc |
| ICF | ieee-cis-fraud | classification | finance | auc |
| TC | telecom-churn | classification | finance/telco | auc |
| J | jannis | classification | general | auc_ovr |
| N | nomao | classification | general | auc |
| BCW | breast-cancer-wisconsin | classification | healthcare | auc |
| D1 | diabetes-130us | classification | healthcare | auc_ovr |
| HD | heart-disease | classification | healthcare | auc_ovr |
| BM2 | broken-machine | classification | industrial | auc |
| VS | vehicle-sensit | classification | physical-sensor | auc_ovr |
| C | covertype | multiclass | physical-science | auc_ovr |
| MM | microsoft-mslr | regression | general | r2 |
| CH | california-housing | regression | general/real-estate | r2 |
| HP | house-prices | regression | general/real-estate | r2 |
| M | medical | regression | healthcare | r2 |
| CS | concrete-strength | regression | physical/materials | r2 |
| S | superconductivity | regression | physical/materials | r2 |

## Overview

Mean held-out score across the three model families (per-family scores are fold-means; metrics differ per dataset -- see the legend).

| method | QB | E | BM | BPC | GC | HCD | ICF | TC | J | N | BCW | D1 | HD | BM2 | VS | C | MM | CH | HP | M | CS | S |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| cafem | **0.924** | **0.890** | 0.882 | **0.703** | 0.801 | **0.696** | **0.833** | **0.825** | **0.818** | **0.988** | 0.997 | **0.652** | 0.788 | **0.573** | **0.928** | **0.852** | **0.142** | 0.731 | **0.254** | 0.971 | 0.766 | **0.845** |
| openfe | 0.922 | 0.860 | **0.902** | — | **0.806** | — | — | 0.802 | — | — | **0.998** | — | **0.805** | — | — | — | — | **0.777** | — | **0.975** | **0.778** | — |

## Per-method scores

| method | model | QB | E | BM | BPC | GC | HCD | ICF | TC | J | N | BCW | D1 | HD | BM2 | VS | C | MM | CH | HP | M | CS | S |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| cafem | knn | 0.905 | 0.888 | 0.850 | 0.630 | 0.759 | 0.595 | 0.764 | 0.796 | 0.750 | 0.981 | 0.999 | 0.593 | 0.764 | 0.499 | 0.910 | 0.857 | 0.081 | 0.738 | 0.644 | 0.968 | 0.716 | 0.888 |
|  | linear | 0.932 | 0.820 | 0.863 | 0.724 | 0.813 | 0.740 | 0.830 | 0.849 | 0.826 | 0.988 | 0.994 | 0.652 | 0.791 | 0.498 | 0.926 | 0.828 | 0.131 | 0.620 | -0.477 | 0.976 | 0.647 | 0.735 |
|  | tree | 0.936 | 0.964 | 0.934 | 0.755 | 0.830 | 0.752 | 0.906 | 0.830 | 0.877 | 0.994 | 0.998 | 0.712 | 0.809 | 0.723 | 0.949 | 0.872 | 0.215 | 0.836 | 0.594 | 0.970 | 0.935 | 0.911 |
| openfe | knn | 0.902 | 0.801 | 0.889 | — | 0.781 | — | — | 0.803 | — | — | 1.000 | — | 0.799 | — | — | — | — | 0.790 | — | 0.974 | 0.740 | — |
|  | linear | 0.930 | 0.830 | 0.903 | — | 0.826 | — | — | 0.811 | — | — | 0.995 | — | 0.799 | — | — | — | — | 0.682 | — | 0.976 | 0.660 | — |
|  | tree | 0.934 | 0.947 | 0.914 | — | 0.812 | — | — | 0.793 | — | — | 1.000 | — | 0.817 | — | — | — | — | 0.860 | — | 0.976 | 0.933 | — |

## Speed (feature-generation wall-time)

| method | QB | E | BM | BPC | GC | HCD | ICF | TC | J | N | BCW | D1 | HD | BM2 | VS | C | MM | CH | HP | M | CS | S | median |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| cafem | 4.9 s | 2.1 s | 2.3 s | 17.0 s | 1.8 s | 10.5 s | 8.8 s | 2.5 s | 81.8 s | 11.4 s | 3.7 s | 11.1 s | 1.9 s | 15.9 s | 69.8 s | 24.4 s | 10.5 s | 2.3 s | 6.7 s | 1.3 s | 1.3 s | 18.0 s | 7.8 s |
| openfe | 111.8 s | 60.4 s | 4.2 min | — | 61.5 s | — | — | 92.4 s | — | — | 39.8 s | — | 21.7 s | — | — | — | — | 37.9 s | — | 29.5 s | 19.0 s | — | 50.1 s |

## Feature counts (before -> after)

Number of columns fed into the downstream models before (original, post max-cols-cap) and after a method's generated features are added.

| method | QB | E | BM | BPC | GC | HCD | ICF | TC | J | N | BCW | D1 | HD | BM2 | VS | C | MM | CH | HP | M | CS | S |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| cafem | 41 -> 44 | 8 -> 12 | 16 -> 16 | 132 -> 133 | 20 -> 21 | 121 -> 127 | 200 -> 206 | 19 -> 21 | 54 -> 74 | 118 -> 126 | 30 -> 36 | 49 -> 51 | 13 -> 13 | 58 -> 62 | 100 -> 120 | 54 -> 57 | 136 -> 139 | 8 -> 13 | 80 -> 85 | 5 -> 7 | 8 -> 8 | 81 -> 101 |
| openfe | 41 -> 51 | 8 -> 18 | 16 -> 26 | — | 20 -> 30 | — | — | 19 -> 29 | — | — | 30 -> 40 | — | 13 -> 14 | — | — | — | — | 8 -> 18 | — | 5 -> 15 | 8 -> 18 | — |

## By task

Mean of the per-dataset overview score (mean across model families) over each dataset's group; a method missing on every dataset in a group shows —.

| method | classification | multiclass | regression |
|---|---|---|---|
| cafem | 0.820 | **0.852** | 0.618 |
| openfe | **0.871** | — | **0.843** |

## By sector

Mean of the per-dataset overview score (mean across model families) over each dataset's group; a method missing on every dataset in a group shows —.

| method | chemistry | energy | finance | finance/telco | general | general/real-estate | healthcare | industrial | physical-science | physical-sensor | physical/materials |
|---|---|---|---|---|---|---|---|---|---|---|---|
| cafem | **0.924** | **0.890** | 0.783 | **0.825** | **0.649** | 0.492 | 0.852 | **0.573** | **0.852** | **0.928** | **0.805** |
| openfe | 0.922 | 0.860 | **0.854** | 0.802 | — | **0.777** | **0.926** | — | — | — | 0.778 |

## Failures / timeouts / crashes

| dataset | method | status | count |
|---|---|---|---|
| bnp-paribas-claims | openfe | timeout | 1 |
| broken-machine | openfe | timeout | 1 |
| covertype | openfe | timeout | 1 |
| diabetes-130us | openfe | timeout | 1 |
| home-credit-default | openfe | timeout | 1 |
| house-prices | openfe | timeout | 1 |
| ieee-cis-fraud | openfe | timeout | 1 |
| jannis | openfe | timeout | 1 |
| microsoft-mslr | openfe | timeout | 1 |
| nomao | openfe | timeout | 1 |
| superconductivity | openfe | timeout | 1 |
| vehicle-sensit | openfe | timeout | 1 |