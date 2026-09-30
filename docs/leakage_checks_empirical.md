# Empirical Leakage Checks — Milestone 1 Section G

## Train/test TransactionID overlap

Overlapping TransactionID count: **0**

No row-level contamination between train and test via TransactionID.

## Correlation with `isFraud` (flag threshold: |corr| >= 0.95)

No column crosses the suspicious-correlation threshold.

### Top 15 columns by |correlation| with isFraud (for context, not a ranking of importance)

| column   |   corr_with_isFraud |   abs_corr |
|:---------|--------------------:|-----------:|
| V257     |            0.38306  |   0.38306  |
| V246     |            0.366878 |   0.366878 |
| V244     |            0.364129 |   0.364129 |
| V242     |            0.36059  |   0.36059  |
| V201     |            0.328005 |   0.328005 |
| V200     |            0.318783 |   0.318783 |
| V189     |            0.308219 |   0.308219 |
| V188     |            0.303582 |   0.303582 |
| V258     |            0.297151 |   0.297151 |
| V45      |            0.281832 |   0.281832 |
| V158     |            0.278066 |   0.278066 |
| V156     |            0.275952 |   0.275952 |
| V149     |            0.273282 |   0.273282 |
| V228     |            0.268861 |   0.268861 |
| V44      |            0.260376 |   0.260376 |
