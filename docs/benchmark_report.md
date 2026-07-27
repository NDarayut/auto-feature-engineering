# Benchmark Report

## Datasets

| abbrev | dataset | task | metric |
|---|---|---|---|
| BCW | breast-cancer-wisconsin | classification | auc |
| CH | california-housing | regression | r2 |
| CS | concrete-strength | regression | r2 |
| E | electricity | classification | auc |
| GC | german-credit | classification | auc |
| HD | heart-disease | classification | auc_ovr |
| HP | house-prices | regression | r2 |
| J | jannis | classification | auc_ovr |
| M | medical | regression | r2 |
| N | nomao | classification | auc |
| QB | qsar-biodegradation | classification | auc |
| S | superconductivity | regression | r2 |
| TC | telecom-churn | classification | auc |

## Overview

Mean held-out score across the three model families (per-family scores are fold-means; metrics differ per dataset -- see the legend).

| method | BCW | CH | CS | E | GC | HD | HP | J | M | N | QB | S | TC |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| baseline | 0.993 | 0.715 | 0.738 | **0.896** | **0.774** | **0.734** | 0.193 | **0.812** | 0.974 | **0.990** | 0.916 | 0.849 | 0.821 |
| autofeat | 0.993 | 0.756 | 0.852 | — | 0.772 | 0.732 | — | — | 0.974 | — | 0.913 | **0.855** | **0.823** |
| featuretools | **0.995** | **0.768** | **0.857** | — | 0.763 | 0.712 | **0.297** | — | 0.974 | — | **0.917** | -1.22e+21 | 0.717 |
| openfe | 0.992 | 0.767 | 0.776 | 0.864 | 0.771 | 0.724 | — | — | **0.977** | — | 0.915 | — | 0.802 |

## Per-method scores

| method | model | BCW | CH | CS | E | GC | HD | HP | J | M | N | QB | S | TC |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| baseline | knn | 0.989 | 0.694 | 0.677 | 0.905 | 0.756 | 0.729 | 0.655 | 0.742 | 0.973 | 0.985 | 0.903 | 0.887 | 0.790 |
|  | linear | 0.995 | 0.606 | 0.605 | 0.816 | 0.787 | 0.752 | -0.659 | 0.816 | 0.978 | 0.988 | 0.925 | 0.740 | 0.848 |
|  | tree | 0.994 | 0.844 | 0.933 | 0.968 | 0.780 | 0.720 | 0.582 | 0.878 | 0.970 | 0.996 | 0.920 | 0.921 | 0.826 |
| autofeat | knn | 0.990 | 0.736 | 0.791 | — | 0.746 | 0.726 | — | — | 0.974 | — | 0.900 | 0.889 | 0.794 |
|  | linear | 0.996 | 0.688 | 0.831 | — | 0.787 | 0.751 | — | — | 0.978 | — | 0.919 | 0.757 | 0.849 |
|  | tree | 0.994 | 0.844 | 0.933 | — | 0.783 | 0.720 | — | — | 0.970 | — | 0.920 | 0.920 | 0.826 |
| featuretools | knn | 0.992 | 0.747 | 0.786 | — | 0.751 | 0.724 | 0.655 | — | 0.968 | — | 0.912 | 0.889 | 0.804 |
|  | linear | 0.997 | 0.703 | 0.852 | — | 0.745 | 0.690 | -0.353 | — | 0.978 | — | 0.910 | -3.67e+21 | 0.517 |
|  | tree | 0.996 | 0.852 | 0.933 | — | 0.794 | 0.723 | 0.588 | — | 0.977 | — | 0.929 | 0.925 | 0.830 |
| openfe | knn | 0.989 | 0.779 | 0.740 | 0.807 | 0.758 | 0.718 | — | — | 0.975 | — | 0.905 | — | 0.803 |
|  | linear | 0.995 | 0.675 | 0.657 | 0.832 | 0.788 | 0.739 | — | — | 0.978 | — | 0.922 | — | 0.811 |
|  | tree | 0.993 | 0.848 | 0.930 | 0.954 | 0.767 | 0.714 | — | — | 0.978 | — | 0.919 | — | 0.793 |

## Speed (feature-generation wall-time)

| method | BCW | CH | CS | E | GC | HD | HP | J | M | N | QB | S | TC | median |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| baseline | 0.0 s | 0.0 s | 0.0 s | 0.0 s | 0.0 s | 0.0 s | 0.0 s | 0.0 s | 0.0 s | 0.0 s | 0.0 s | 0.0 s | 0.0 s | 0.0 s |
| autofeat | 31.2 s | 10.3 s | 4.3 s | — | 26.9 s | 14.7 s | — | — | 16.9 s | — | 110.3 s | 61.9 s | 2.3 min | 26.9 s |
| featuretools | 2.0 s | 0.7 s | 0.4 s | — | 1.1 s | 0.6 s | 17.5 s | — | 1.4 s | — | 3.7 s | 33.6 s | 2.1 s | 1.7 s |
| openfe | 42.0 s | 35.5 s | 16.2 s | 90.6 s | 60.1 s | 19.3 s | — | — | 34.9 s | — | 117.8 s | — | 107.5 s | 42.0 s |

## Failures / timeouts / crashes

| dataset | method | status | count |
|---|---|---|---|
| bank-marketing | autofeat | timeout | 1 |
| bank-marketing | baseline | pair_crashed | 1 |
| bank-marketing | featuretools | pair_crashed | 1 |
| bank-marketing | openfe | timeout | 1 |
| bnp-paribas-claims | autofeat | timeout | 1 |
| bnp-paribas-claims | baseline | pair_crashed | 1 |
| bnp-paribas-claims | featuretools | crashed | 1 |
| bnp-paribas-claims | openfe | timeout | 1 |
| electricity | autofeat | timeout | 1 |
| electricity | featuretools | pair_crashed | 1 |
| heart-disease | openfe | crashed | 1 |
| house-prices | autofeat | error | 1 |
| house-prices | openfe | timeout | 1 |
| jannis | autofeat | timeout | 1 |
| jannis | openfe | timeout | 1 |
| microsoft-mslr | baseline | pair_crashed | 1 |
| microsoft-mslr | openfe | timeout | 1 |
| nomao | autofeat | timeout | 1 |
| nomao | featuretools | crashed | 1 |
| nomao | openfe | timeout | 5 |
| superconductivity | openfe | timeout | 1 |
| vehicle-sensit | autofeat | timeout | 1 |
| vehicle-sensit | baseline | pair_crashed | 1 |
| vehicle-sensit | featuretools | crashed | 1 |
| vehicle-sensit | openfe | timeout | 1 |