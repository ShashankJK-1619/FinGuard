# Milestone 3 — Chronological Split & Interpretable Fraud Baseline

Module B (Fraud Detection), Milestone 3. Input: Milestone 2's
`data/processed/baseline_features_v1.parquet` (590,540 rows, train-only).
`test_transaction`/`test_identity` are not loaded anywhere in this
milestone and are never used as labeled evaluation data — "holdout" below
is a time-later slice of the same labeled training data, evaluated exactly
once. No `C*`/`D*`/`M*`/`V*`/raw `id_*` columns are used (none are even
present in the Milestone 2 table — re-verified by
`tests/models/test_schema.py`). No tree-based/boosted model is used.

All code: `src/models/schema.py` (column-role policy), `src/models/split.py`
(chronological split), `src/models/preprocessing.py` (sklearn
`ColumnTransformer`), `src/models/train_baseline.py` (fit/evaluate/threshold
analysis/coefficients). Run with
`.venv/bin/python -m src.models.train_baseline`; writes
`data/interim/milestone3_metrics.json` (raw numbers this document is built
from) in addition to printing the same report to stdout.

## 1. What this milestone solves

Milestone 1 established that fraud drifts over time and that the
competition's own train/test split is time-based, not random
(`docs/time_structure.md`). Milestone 2 built point-in-time-safe features
but created no split and no model. Milestone 3 builds the first model —
deliberately a plain, interpretable logistic regression, not because it's
expected to be the best-performing option, but because a fraud decision
needs to be explainable before it needs to be optimal, and a transparent
baseline is the right reference point to measure any later, more complex
model against.

## 2. Chronological split

Split boundaries fall on whole **calendar-day** boundaries
(`day = TransactionDT // 86400`, reusing Milestone 2's `day_index`
definition), not an arbitrary row index — so no single day has some
transactions in one segment and others in another
(`src/models/split.py::chronological_split`, verified by
`tests/models/test_split.py`). Target proportions were 60% train / 20%
validation / 20% holdout: large enough training volume to fit a model
against a ~3.5% fraud prevalence, while keeping both validation (for
threshold analysis) and holdout (for a single final check) at a size
that still contains thousands of fraud cases each.

| Split | Days (elapsed) | Rows | Fraud count | Fraud prevalence | `TransactionDT` range |
|---|---|---|---|---|---|
| Train | 1–101 | 356,216 | 12,039 | 3.380% | 86,400 – 8,812,779 |
| Validation | 102–141 | 118,790 | 4,682 | 3.941% | 8,812,817 – 12,268,796 |
| Holdout | 142–182 | 115,534 | 3,942 | 3.412% | 12,268,813 – 15,811,131 |

Realized proportions: 60.3% / 20.1% / 19.6% of rows (day-boundary snapping
means these aren't exactly 60/20/20, reported exactly here rather than
assumed). Fraud prevalence is not constant across the window (3.38% →
3.94% → 3.41%) — consistent with Milestone 1's non-stationarity concern;
no attempt is made to correct for this drift in this milestone.

## 3. Feature set and preprocessing

Full column-by-column include/exclude policy and reasoning lives in
`src/models/schema.py` (not duplicated here to avoid two sources of truth
that can drift apart). Summary: **26 source columns** feed the model — 16
numeric, 6 categorical, 4 binary passthrough — expanding to **58 matrix
columns** after one-hot encoding and per-column missing indicators. Key
decisions:

- **High-cardinality/masked raw columns already replaced by a Milestone-2
  derived feature are dropped from the model** (`card1`, `addr1`,
  `P_emaildomain`, `R_emaildomain`, `DeviceInfo`) — their frequency/
  recency/missingness proxies are used instead, avoiding a
  several-hundred-dummy one-hot block and unseen-category risk at scoring
  time.
- **`TransactionDT` itself is excluded from the feature matrix** (it is
  only used to build the split): its absolute value has an undisclosed
  reference point, so a coefficient on it has no interpretable unit, and
  `day_index`/`week_index`/`hour_sin`/`hour_cos`/`day_of_week_index`
  already carry the same information at an interpretable granularity.
- **Masked numeric codes with no Milestone-2 derived alternative**
  (`card2`, `card3`, `card5`, `addr2`) are kept as numeric inputs. This
  imposes an **unverified ordinal/interval assumption** on values whose
  true mapping Vesta never disclosed — flagged here as a limitation, not
  hidden, and excluded from the coefficient interpretation in §8.
- **Missing-value handling**: numeric columns are median-imputed with
  `SimpleImputer(add_indicator=True)`, so a first-occurrence
  `seconds_since_last_card1`/`addr1` (`NaN` = "no prior transaction with
  this key", a real state) gets its own indicator column instead of being
  silently averaged away. Categorical columns are imputed to an explicit
  `"missing"` level rather than dropped or merged into an existing
  category.
- **Categorical encoding**: one-hot with `handle_unknown="ignore"`, so a
  category that appears only in validation/holdout (not seen during
  `fit` on train) produces an all-zero row instead of raising or leaking
  information about which unseen value it was.
- **Binary flags are passed through unscaled**, so their coefficients are
  directly a log-odds effect of "flag on vs. off," not an arbitrary
  standardized unit.
- **No future leakage**: the entire preprocessing pipeline (imputer
  medians/most-frequent values, scaler means/variances, one-hot category
  vocabularies) is fit via `sklearn.pipeline.Pipeline.fit` on the
  **train split only**; validation and holdout only ever go through
  `.transform`/`.predict_proba` with those already-fitted statistics
  (`tests/models/test_preprocessing.py::test_preprocessor_fit_statistics_come_only_from_fit_data`,
  `tests/models/test_train_baseline.py::test_model_is_fit_only_on_train_split_not_val_or_holdout`).

## 4. Model

Plain `sklearn.linear_model.LogisticRegression` (`lbfgs`, `max_iter=1000`),
`class_weight="balanced"` — chosen over resampling (SMOTE, undersampling,
etc.) because it rebalances the loss function for the ~3.5% prevalence
without synthesizing rows or adding a resampling step, and the fitted
model is still a plain logistic regression with directly interpretable
coefficients. No random forest, gradient boosting, or other non-linear
model is used, per this milestone's explicit scope.

## 5. Validation metrics (default threshold = 0.5)

| Metric | Value |
|---|---|
| n rows | 118,790 |
| Fraud prevalence | 3.941% |
| Confusion matrix | TN=76,684, FP=37,424, FN=1,225, TP=3,457 |
| Precision | 0.0846 |
| Recall | 0.7384 |
| F1 | 0.1517 |
| ROC-AUC | 0.7652 |
| PR-AUC | 0.1428 |
| Brier score | 0.2599 |

## 6. Holdout metrics (default threshold = 0.5, touched once)

| Metric | Value |
|---|---|
| n rows | 115,534 |
| Fraud prevalence | 3.412% |
| Confusion matrix | TN=62,159, FP=49,433, FN=917, TP=3,025 |
| Precision | 0.0577 |
| Recall | 0.7674 |
| F1 | 0.1073 |
| ROC-AUC | 0.7317 |
| PR-AUC | 0.1202 |
| Brier score | 0.3157 |

Holdout ROC-AUC/PR-AUC are both lower than validation (0.7317 vs. 0.7652;
0.1202 vs. 0.1428) — consistent with genuine future drift rather than a
comparable random split, and a reminder that validation-split performance
should not be quoted as the expected production number.

## 7. Threshold analysis (validation split only)

Computed by sweeping the decision threshold on **validation**, never on
holdout, to avoid tuning against the one split reserved for a final,
single check.

| Threshold | Precision | Recall | False positives | False negatives | Alert volume | Alert rate |
|---|---|---|---|---|---|---|
| 0.05 | 0.0395 | 0.9994 | 113,866 | 3 | 118,545 | 99.79% |
| 0.10 | 0.0396 | 0.9989 | 113,484 | 5 | 118,161 | 99.47% |
| 0.15 | 0.0398 | 0.9972 | 112,650 | 13 | 117,319 | 98.76% |
| 0.20 | 0.0406 | 0.9861 | 109,174 | 65 | 113,791 | 95.79% |
| 0.30 | 0.0470 | 0.9368 | 88,959 | 296 | 93,345 | 78.58% |
| 0.40 | 0.0629 | 0.8481 | 59,211 | 711 | 63,182 | 53.19% |
| 0.50 | 0.0846 | 0.7384 | 37,424 | 1,225 | 40,881 | 34.41% |
| 0.60 | 0.1057 | 0.6606 | 26,168 | 1,589 | 29,261 | 24.63% |
| 0.70 | 0.1205 | 0.5737 | 19,605 | 1,996 | 22,291 | 18.77% |
| 0.80 | 0.1346 | 0.4791 | 14,424 | 2,439 | 16,667 | 14.03% |
| 0.90 | 0.1643 | 0.3206 | 7,635 | 3,181 | 9,136 | 7.69% |

**No threshold above is claimed "best."** Picking one requires a business
cost function — e.g. the relative cost of a missed fraud (false negative)
vs. a false decline or manual-review alert (false positive) — that has
not been defined in this milestone. What the table does establish: at the
literal default of 0.5, over a third of all validation transactions would
be flagged, which is not an operable alert volume for any real review
queue; any deployable threshold would sit well above 0.5, and choosing
exactly where is a business decision, not a modeling one.

## 8. Calibration findings

Brier scores (0.2599 validation, 0.3157 holdout) are high for a binary
outcome with ~3.5% prevalence — a well-calibrated model with that
prevalence would typically score well under 0.05. The calibration curve
confirms why: across both splits, the model's predicted probabilities
(roughly 0.2–0.93 on validation, 0.2–0.97 on holdout, binned by decile)
are **consistently and substantially higher** than the observed fraud
rate in every bin (roughly 0.01–0.15 on validation, 0.01–0.12 on
holdout). This is an expected, direct consequence of `class_weight=
"balanced"`: it reweights the *training loss*, not the output
probabilities, so `predict_proba` no longer returns calibrated fraud
probabilities — it returns a rebalanced decision score that is still
correctly *ordered* (hence a usable ROC-AUC/PR-AUC and threshold sweep)
but must not be read as "this transaction has an X% chance of being
fraud." Anyone consuming these scores downstream (e.g. as an input to a
cost-based decision) would need either a recalibration step (e.g.
Platt scaling / isotonic regression fit on validation) or a model without
`class_weight="balanced"`, both out of scope here.

## 9. Coefficient interpretation — transparent features only

Per this milestone's scope, coefficients are interpreted only for
features whose business meaning is documented. `card2`, `card3`, `card5`,
and `addr2` are masked numeric codes kept in the model (§3) but
**excluded from this interpretation** — their coefficients exist in
`data/interim/milestone3_metrics.json` for completeness, but none ranked
among the strongest positive or negative effects (largest magnitude
among them: `card3`, coefficient 0.273), so excluding them from the
narrative costs little. All numeric coefficients are in "per one
standard deviation" units (features were standard-scaled); binary/
one-hot coefficients are in raw log-odds units (unscaled).

**Strongest positive (higher predicted fraud log-odds):**

| Feature | Coefficient | Odds ratio | Interpretation |
|---|---|---|---|
| `DeviceInfo_missing` | 1.072 | 2.92 | Having an identity record but no device descriptor is associated with higher fraud odds (unscaled binary flag) |
| `DeviceType_mobile` | 0.713 | 2.04 | Mobile device is associated with higher fraud odds than the missing-device baseline |
| `ProductCD_S` | 0.707 | 2.03 | Product code `S` carries higher fraud odds than other product codes (exact product meaning undisclosed) |
| `email_domain_match_MATCH` | 0.507 | 1.66 | Purchaser/recipient email domains matching is associated with *higher*, not lower, fraud odds — counter to a naive "matching = trustworthy" assumption |
| `card6_credit` | 0.412 | 1.51 | Credit (vs. debit) card is associated with higher fraud odds |
| `ProductCD_W` | 0.352 | 1.42 | Product code `W` carries higher fraud odds than the reference product codes |
| `day_index` / `week_index` | 0.350 / 0.349 | 1.42 / 1.42 | A trend term: fraud odds rose over the observed 182-day window — see §10 for why this should not be extrapolated forward |
| `TransactionAmt_log1p` | 0.337 | 1.40 | Larger transaction amounts are associated with modestly higher fraud odds |
| `DeviceType_desktop` | 0.290 | 1.34 | Desktop device is also associated with higher fraud odds than the missing-device baseline (less than mobile) |

**Strongest negative (lower predicted fraud log-odds):**

| Feature | Coefficient | Odds ratio | Interpretation |
|---|---|---|---|
| `ProductCD_R` | -1.104 | 0.33 | Product code `R` carries markedly lower fraud odds |
| `DeviceType_None` (no identity/device record) | -1.066 | 0.34 | No device info at all is associated with lower fraud odds than having a mobile/desktop record — note this is the reference/imputed "missing" level, not a documented device type |
| `DeviceInfo_prior_count` | -0.679 | 0.51 | A device descriptor seen more often before (more "familiar") is associated with lower fraud odds — the frequency/velocity signal working as intended |
| `has_identity_record` | -0.619 | 0.54 | Having any identity/device record at all is associated with lower fraud odds than not having one |
| `R_emaildomain_missing` | -0.393 | 0.67 | A missing recipient email domain (common when purchaser == recipient, per Milestone 1/2) is associated with lower fraud odds |
| `card6_debit` | -0.370 | 0.69 | Debit card is associated with lower fraud odds than the credit-card effect above |
| `email_domain_match_ONE_MISSING` | -0.291 | 0.75 | One of the two email domains missing (not both) is associated with lower fraud odds than a full MATCH |

These are **associations within this model, at this point in time, on
this feature set** — not causal claims, and not claims that any single
feature is a strong fraud signal in isolation.

## 10. Limitations and remaining risks

- **Miscalibrated probabilities** (§8): `predict_proba` outputs are
  ordering-correct but not true probabilities, a direct consequence of
  `class_weight="balanced"`. Do not feed these scores into any downstream
  system that assumes calibrated probability (e.g. an expected-loss
  calculation) without a recalibration step.
- **`day_index`/`week_index` are monotonic-in-time counters used as
  linear trend features.** Any transaction scored after this training
  window has a `day_index` value the model never saw — a linear
  extrapolation outside the training domain, not an interpolation. The
  observed positive coefficient reflects fraud prevalence rising *within*
  this 182-day window; there's no guarantee that trend continues, and a
  model relying on it will degrade as deployment time moves further past
  the training window.
- **`card2`, `card3`, `card5`, `addr2` carry an unverified ordinal/
  interval assumption** (§3, §9) — none ranked as a dominant coefficient
  here, but this doesn't retroactively validate the assumption, only
  observes it didn't matter much for *this* model.
- **No default, deployable threshold has been chosen** (§7) — that
  requires a business cost function not defined in this milestone.
- **`C*`/`D*`/`V*`/`M*` leakage status remains genuinely unresolved**
  (carried over from Milestones 1–2) — this baseline still sidesteps that
  question rather than answering it.
- **Holdout was evaluated once, here.** If any future milestone revisits
  modeling choices and re-checks holdout performance, that holdout is no
  longer a clean final check — it would need to be treated as contaminated
  and a fresh, never-touched slice used instead.
- **Fraud prevalence is not constant across the split** (3.38% → 3.94% →
  3.41%), and the drop in both ROC-AUC and PR-AUC from validation to
  holdout (§6) is consistent with real non-stationarity, not evaluation
  noise — though with one split this can't be fully distinguished from
  a one-off fluctuation.

## 11. Tests

`tests/models/test_schema.py` — model-feature columns never include a
masked-pattern column, the identifier/label/time columns, or overlap
across numeric/categorical/binary groups.
`tests/models/test_split.py` — split covers every row exactly once, is
strictly chronological across segment boundaries, never splits a
calendar day across two segments, is invariant to input row shuffling,
and rejects invalid fraction inputs.
`tests/models/test_preprocessing.py` — non-feature columns are dropped,
missing numeric values are handled without error, unseen categorical
values at transform time are ignored rather than raising, and fit
statistics (e.g. imputation medians) come only from the data passed to
`fit`.
`tests/models/test_train_baseline.py` — pipeline produces valid
probabilities, `evaluate()` returns metrics in valid ranges with a
confusion matrix that sums to the split size, threshold sweeps produce
monotonically non-increasing alert volume, coefficients are extracted for
every transformed feature, the fitted pipeline does not react to
mutations of already-fit-and-transformed splits (leakage check), and (when
`data/processed/baseline_features_v1.parquet` is present) a full
real-data run produces a sane (>0.6) ROC-AUC on both validation and
holdout.

Full suite: 58 passed, 1 skipped (same intentional missing-raw-file
error-path skip from Milestones 1–2).

## 12. Runtime

Fitting the full pipeline (preprocessing + logistic regression) on
356,216 training rows × 58 post-encoding columns completes in a few
seconds on the same machine used for Milestones 1–2. No new dependencies
were added — `scikit-learn` was already in `requirements.txt`.
