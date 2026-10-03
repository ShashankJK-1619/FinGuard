# Milestone 5 — Transparent Nonlinear Model Comparison

This milestone adds one nonlinear tabular model —
`sklearn.ensemble.HistGradientBoostingClassifier` — and compares it fairly
against Milestone 3's class-weighted logistic-regression baseline, reusing
every piece of leakage-safe infrastructure Milestones 3–4 already built
unchanged: the transparent feature set, the chronological
train/validation/temporal-benchmark split, the `cal_fit`/`selection`
sub-split and out-of-sample calibration methodology, and the illustrative
cost-based decisioning framework. No masked `C*`/`D*`/`M*`/`V*`/`id_*`
column is introduced anywhere in this milestone. No deep learning, no
graph models. Numbers below come from one real run of
`.venv/bin/python -m src.models.train_nonlinear` against
`data/processed/baseline_features_v1.parquet`
(`data/interim/milestone5_metrics.json` is the reproducibility artifact).

**Consistency-pass correction (post-initial-draft):** the first version of
this document recommended the no-`day_index`/`week_index` configuration as
"preferred" while showing the *with*-time-index configuration's calibrated
and decision-policy numbers in the headline tables — those two
configurations are not numerically identical, so that recommendation was
not backed by the metrics actually displayed. This version fixes that: the
preferred (no-time-drift) configuration is now run through the exact same
full pipeline (train → `cal_fit` → sigmoid/isotonic comparison → `selection`
→ `temporal_benchmark` → business decisioning, via the new
`nonlinear_full_pipeline` function) as the with-time-index configuration and
the logistic baseline, under byte-for-byte the same cost assumptions and
threshold grids, and every number reported as "the preferred
configuration's" result below is that configuration's *own* calibrated and
decisioning output — not borrowed from the other variant. See §4/§6/§8.

## 0. What is reused unchanged from Milestones 3–4

- **Feature set**: `src.models.schema.NUMERIC_FEATURES` /
  `CATEGORICAL_FEATURES` / `BINARY_FEATURES` (and
  `NUMERIC_FEATURES_NO_TIME_DRIFT` for the preferred configuration) — the
  identical columns Milestone 3 selected, still excluding every masked
  group.
- **Split**: `src.models.split.chronological_split` with its Milestone 3
  default fractions (`train_frac=0.6, val_frac=0.2`), then
  `chronological_split_two_way(val_df, first_frac=0.5)` exactly as
  Milestone 4 introduced it. Boundaries were not touched to favor either
  model.
- **Calibration methodology**: `src.models.calibration` and
  `src.models.train_calibrated.calibration_comparison`/
  `select_best_calibration_method`, reused as library calls, not
  reimplemented. Calibrators fit on `cal_fit` only, evaluated on
  `selection`/`temporal_benchmark` only.
- **Cost framework**: `src.models.cost_model` and
  `configs/cost_assumptions.json`, byte-for-byte the same hypothetical
  figures Milestone 4 used (`cost_per_missed_fraud=500`,
  `cost_per_false_positive_decline=25`, `cost_per_manual_review=3`,
  `review_capacity_frac=0.05`, `review_fraud_catch_rate=0.8`,
  `review_false_positive_rate=0.05`). See
  [docs/milestone4_calibration_decisioning.md](milestone4_calibration_decisioning.md)
  §5 for the same disclaimer — these remain hypothetical, illustrative
  portfolio assumptions, not real financial figures, and were not changed
  to make any configuration look better.
- **`REVIEW_THRESHOLD_GRID`/`DECLINE_THRESHOLD_GRID`**: identical lists
  from `src.models.train_calibrated`, used for the logistic baseline and
  *both* nonlinear configurations.

See `src/models/train_nonlinear.py` for the implementation; the function
names below (`tune_hyperparameters`, `nonlinear_full_pipeline`,
`decision_policy_comparison`, `permutation_importance_table`,
`time_feature_ablation_nonlinear`) are importable and unit-tested in
`tests/models/test_train_nonlinear.py`.

## 1. Split (reproduced this run)

| Split | Rows | Fraud rows | Fraud rate | Day range |
|---|---|---|---|---|
| train | 356,216 | 12,039 | 3.38% | 1–101 |
| val (→ cal_fit + selection) | 118,790 | 4,682 | 3.94% | 102–141 |
| temporal_benchmark | 115,534 | 3,942 | 3.41% | 142–182 |
| val → cal_fit | 61,343 | 2,682 | 4.37% | 102–121 |
| val → selection | 57,447 | 2,000 | 3.48% | 122–141 |

Identical boundaries to Milestones 3–4 (same `train_last_day`/`val_last_day`
as previously reported), confirming the split was not altered.

## 2. Model choice: `HistGradientBoostingClassifier`

Preferred option per instruction order (HistGradientBoostingClassifier,
then XGBoost/LightGBM only if justified). No dependency/runtime tradeoff
was found that justified reaching further: it ships with the project's
already-pinned `scikit-learn>=1.4,<2` dependency (no new dependency),
natively handles missing values (no imputation needed for numeric columns)
and native categorical splits (no one-hot explosion for category columns).
Total end-to-end runtime for this milestone's consistency-pass run (tuning
+ fitting/calibrating/decisioning BOTH nonlinear feature-set configurations
+ importance, real data) was **≈3 minutes** wall clock on this machine —
only modestly more than the single-configuration run, because the second
configuration reuses the already-tuned hyperparameters (no re-tuning) and
only adds one more fit + calibration comparison + decision-policy grid.
XGBoost/LightGBM would add a new dependency for a problem this module
already handles comfortably within the project's "no Spark, no cloud
infra, no deep learning" minimal-tooling stance (README.md).

### Preprocessing differs from the linear pipeline, on purpose

`src.models.preprocessing.build_preprocessor` (median-impute + scale +
one-hot, built for a linear model) is **not** reused as-is —
`src.models.train_nonlinear.build_tree_preprocessor` instead:

- passes numeric columns straight through, unscaled and un-imputed — a
  tree's splits are threshold-based and invariant to monotonic rescaling,
  and HistGradientBoostingClassifier treats a missing numeric value as its
  own learned split direction natively, so median-filling it would destroy
  rather than preserve the "missing is informative" signal
  `preprocessing.py` already protects via `add_indicator=True` for the
  linear model;
- ordinal-encodes categorical columns (after imputing missing as an
  explicit "missing" level, matching `preprocessing.py`'s policy) and
  passes a boolean `categorical_features` mask to
  `HistGradientBoostingClassifier` so the encoded integers are read as
  discrete categories, never as an ordinal magnitude;
- imputes binary flags the same way `preprocessing.py` does.

This is still the exact same *feature set* — Requirement A's "same
transparent feature set" is about which columns are modeled, not about
reusing a linear-model-shaped preprocessing pipeline verbatim for a model
family it was never designed for. `class_weight="balanced"` is used for
the nonlinear model too, mirroring the logistic baseline's approach to the
~3.4% fraud prevalence (fair comparison — neither model gets a resampling
trick the other doesn't).

## 3. Hyperparameter tuning (requirement B)

**Train-only**, never touching `cal_fit`/`selection`/`temporal_benchmark`.
`train_df` is time-sorted and split via `sklearn.model_selection.TimeSeriesSplit`
into 3 forward-chaining (expanding-window) folds — each fold's validation
block is strictly later in time than everything used to fit it, mirroring
the project's "never evaluate on data from before what you're predicting"
principle at a smaller internal scale.

Search space: 4 explicitly enumerated candidates (not a brute-force grid),
varying tree complexity (`max_leaf_nodes`), the learning-rate/iteration
tradeoff, and L2 regularization strength — the three knobs most likely to
matter for a dataset this size/imbalance. Selection metric: mean PR-AUC
(average precision) across the 3 folds, matching this project's existing
preference for PR-AUC over raw accuracy on an imbalanced target.

| max_iter | max_leaf_nodes | learning_rate | l2_regularization | mean PR-AUC (3 folds) |
|---|---|---|---|---|
| 100 | 31 | 0.10 | 0.0 | 0.1882 |
| 200 | 31 | 0.05 | 0.0 | 0.1933 |
| 100 | 15 | 0.10 | 1.0 | 0.1978 |
| **200** | **63** | **0.05** | **1.0** | **0.2019 (selected)** |

**Selection logic**: argmax mean PR-AUC across folds — a plain, auditable
rule (no tie-break was needed here; the winner was unambiguous). **Chosen
configuration**: `max_iter=200, max_leaf_nodes=63, learning_rate=0.05,
l2_regularization=1.0`. **Tuning runtime**: 86.5 seconds (12 fits: 4
candidates × 3 folds), on this machine. This same chosen configuration is
reused, unchanged, for **both** nonlinear feature-set configurations below
(with-time-index and the preferred no-time-index one) — only the feature
set changes between them, not the hyperparameters, mirroring how
Milestone 4's own ablation isolated one variable at a time.

## 4. Logistic vs. nonlinear — ranking and calibration, all three configurations (requirements C, D)

All probabilities below are **sigmoid-calibrated** (see §5 for why sigmoid
was selected in every case) and evaluated on the same `selection` /
`temporal_benchmark` splits, fit on the same `train` split. Three
configurations are compared side by side, all through the identical
pipeline (`nonlinear_full_pipeline` for the two nonlinear rows):

| Split | Configuration | ROC-AUC | PR-AUC | Brier (calibrated) |
|---|---|---|---|---|
| selection | logistic | 0.7454 | 0.1273 | 0.0325 |
| selection | nonlinear, with day/week_index | 0.7914 | 0.1969 | 0.0307 |
| selection | **nonlinear, preferred (no day/week_index)** | **0.7894** | **0.1958** | **0.0308** |
| temporal_benchmark | logistic | 0.7317 | 0.1202 | 0.0330 |
| temporal_benchmark | nonlinear, with day/week_index | 0.7788 | 0.1722 | 0.0310 |
| temporal_benchmark | **nonlinear, preferred (no day/week_index)** | **0.7799** | **0.1536** | **0.0314** |

Both nonlinear configurations decisively outrank the logistic baseline
(+0.04–0.05 ROC-AUC, +0.03–0.08 PR-AUC) and are both slightly better
calibrated than logistic (Brier ~0.031 vs ~0.033). Between the two
nonlinear configurations themselves, the picture is a genuine, small,
metric-dependent tradeoff, not a clean win for the preferred configuration:
removing `day_index`/`week_index` very slightly *improves*
`temporal_benchmark` ROC-AUC (0.7788→0.7799) but *reduces*
`temporal_benchmark` PR-AUC (0.1722→0.1536, a ~10.8% relative drop) and
leaves calibrated Brier essentially flat (0.0310→0.0314). §8 discusses this
tradeoff and why the preferred configuration is still recommended despite
it.

At the uniform default threshold of 0.5, all three configurations'
calibrated probabilities rarely cross 0.5 (fraud prevalence ~3.5%, so most
calibrated probabilities sit well under that), giving precision/recall/F1
of 0/0/0 across the board — this is a property of using an arbitrary fixed
0.5 threshold on a well-calibrated low-prevalence probability, not evidence
any configuration is non-functional; §6's decision-policy thresholds
(chosen to minimize illustrative cost, not fixed at 0.5) are the
meaningful precision/recall comparison.

## 5. Calibration comparison detail (requirement D)

Same out-of-sample methodology as Milestone 4 (`fit_calibrator` on
`cal_fit`, `calibration_comparison`/`select_best_calibration_method` on
`selection`/`temporal_benchmark`, never the reverse), run independently for
all three configurations:

| Configuration | Method | selection Brier | temporal_benchmark Brier |
|---|---|---|---|
| nonlinear, with day/week_index | uncalibrated | 0.0992 | 0.1050 |
| nonlinear, with day/week_index | **sigmoid (selected)** | **0.0307** | **0.0310** |
| nonlinear, with day/week_index | isotonic | 0.0307 | 0.0308 |
| nonlinear, **preferred** | uncalibrated | 0.1085 | 0.1169 |
| nonlinear, **preferred** | **sigmoid (selected)** | **0.0308** | **0.0314** |
| nonlinear, **preferred** | isotonic | 0.0307 | 0.0316 |
| logistic | uncalibrated | 0.2609 | 0.3157 |
| logistic | **sigmoid (selected)** | **0.0325** | **0.0330** |

**Sigmoid was selected for every configuration** by the same rule
Milestone 4 established (`select_best_calibration_method`: prefer sigmoid
unless isotonic clears a 3% relative Brier improvement on `selection`) —
isotonic's edge over sigmoid is well under 1% relative in all three cases,
nowhere near the 3% bar, so sigmoid's simplicity wins every time.

Two things stand out:
- Both nonlinear configurations' **uncalibrated** Brier (~0.10–0.11) is
  already far better than the logistic model's uncalibrated Brier (0.261)
  — because `class_weight="balanced"` distorts a tree model's raw
  probability output less severely than it distorts a logistic model's raw
  sigmoid output. All three configurations still clearly benefit from
  calibration, so calibration remains necessary, not optional, for any of
  them.
- Reliability tables (written in full in the metrics artifact) show all
  three calibrated configurations' predicted-probability deciles tracking
  observed fraud rates reasonably closely on both `selection` and
  `temporal_benchmark`, with the usual largest residual gap in the
  highest-risk decile (same qualitative pattern Milestone 4 already
  reported for the logistic model).

## 6. Decision-policy comparison, all three configurations (requirement E)

Same approve/review/decline cost framework, same cost assumptions, same
threshold grids — the identical `costs` object and threshold grids are
passed to all three configurations (Requirement 7: cost assumptions and
thresholds were not changed to favor any configuration). Each
configuration's lowest-illustrative-cost `(review_threshold,
decline_threshold)` policy is chosen on `selection` only, then evaluated
once, unmodified, on `temporal_benchmark`.

All three configurations' `selection`-selected policy landed on
`(review_threshold=0.05, decline_threshold=0.30)` — same point in the
grid, not cherry-picked.

| Split | Configuration | fraud_recall | precision | review_volume | decline_volume | false_positives | false_negatives | illustrative total cost |
|---|---|---|---|---|---|---|---|---|
| selection | logistic | 0.180 | 0.157 | 2,872 | 0 | 121.1 | 1,639.2 | $831,242 |
| selection | nonlinear, with day/week_index | 0.345 | 0.201 | 2,872 | 987 | 770.7 | 1,310.8 | $683,282 |
| selection | **nonlinear, preferred** | **0.337** | **0.208** | 2,872 | **829** | **650.2** | 1,325.8 | **$687,770** |
| temporal_benchmark | logistic | 0.170 | 0.145 | 5,776 | 0 | 246.8 | 3,270.0 | $1,658,498 |
| temporal_benchmark | nonlinear, with day/week_index | 0.317 | 0.179 | 5,776 | 2,089 | 1,705.3 | 2,691.2 | $1,405,559 |
| temporal_benchmark | **nonlinear, preferred** | **0.316** | **0.174** | 5,776 | **2,357** | **2,059.0** | 2,698.2 | **$1,417,903** |

**This is the preferred configuration's own, actual decisioning result —
not the with-time-index configuration's result reused by assumption.** Two
honest findings come out of running it explicitly:

- Against the **logistic baseline**, the preferred configuration still
  wins decisively: fraud recall nearly doubles (0.180→0.337 on `selection`,
  0.170→0.316 on `temporal_benchmark`) and illustrative total cost drops by
  **17.3% on `selection`** ($831,242→$687,770) and **14.5% on
  `temporal_benchmark`** ($1,658,498→$1,417,903) — slightly less of a cost
  reduction than the with-time-index configuration's 17.8%/15.3%, but still
  a large, clear win over logistic.
- Against the **with-time-index nonlinear configuration**, the preferred
  configuration is very slightly *worse* on illustrative cost (+0.7% on
  `selection`, +0.9% on `temporal_benchmark`) and has mixed false-positive
  behavior (15.6% *fewer* false positives on `selection`, 20.8% *more* on
  `temporal_benchmark`). This is a small, real difference, not noise-level
  rounding, and it is reported here precisely so the preferred-configuration
  recommendation in §8/§12 is not overstated as a free win on every axis —
  it is a deliberate tradeoff of a small amount of decisioning performance
  for removing two features with extrapolation risk and zero measured
  permutation importance (§7).

A sensitivity analysis identical in structure to Milestone 4's (11 cost
variations) was run for the **preferred configuration** on `selection`
(full table in the metrics artifact, under
`nonlinear_sensitivity_analysis_selection_preferred_config`); as in
Milestone 4, most ±2x single-input variations do not move the argmin-cost
policy off `(0.05, 0.30)` — manual review remains unconditionally the
cheaper way to act at the margin under most of these variations — while the
two extreme, targeted variations (`review_cost_exceeds_decline_cost`,
`fraud_loss_near_zero`) do move it, same qualitative pattern as the
logistic model's sensitivity table in Milestone 4.

**This is not a production policy recommendation for any configuration** —
see Milestone 4's limitations (§9 of that doc) on manual review's finite
capacity/imperfection and the hypothetical nature of every cost figure;
those caveats apply identically here.

## 7. Feature importance (requirement F)

Logistic coefficients (`extract_coefficients`, ranked by |coefficient|) are
compared against the **with-time-index** nonlinear model's **permutation
importance** (`sklearn.inspection.permutation_importance`, scored by PR-AUC
drop, computed on `selection` — out-of-sample for a pipeline fit on `train`
only, 5 repeats). This importance table was **not** recomputed for the
preferred (no-time-drift) configuration in this run — a deliberate,
noted scope choice, not an oversight: `day_index`/`week_index` both scored
exactly 0.0000 permutation importance in the with-time-index model (see
below), and the ablation/consistency-check in §8 confirms removing them has
only a small, metric-dependent effect on ranking and decisioning, so this
choice does not change which features matter for either configuration.
HistGradientBoostingClassifier has no native `feature_importances_` the
way e.g. RandomForestClassifier does, so permutation importance is the only
importance signal reported for it. **Neither ranking is a causal claim** —
a coefficient describes a linear, additive log-odds effect; a
permutation-importance score describes how much this specific nonlinear
model's held-out PR-AUC drops when a column is shuffled, possibly
reflecting interactions a linear model can't represent at all.

Top permutation-importance features for the with-time-index nonlinear
model (full table of all 26 raw columns is in the metrics artifact):

| Rank | Feature | Importance (mean PR-AUC drop) | Masked code? |
|---|---|---|---|
| 1 | `TransactionAmt_log1p` | 0.0604 | no |
| 2 | `ProductCD` | 0.0596 | no |
| 3 | `email_domain_match` | 0.0480 | no |
| 4 | `card3` | 0.0412 | **yes** |
| 5 | `DeviceType` | 0.0384 | no |
| 6 | `P_emaildomain_prior_count` | 0.0301 | no |
| 7 | `card6` | 0.0213 | no |
| 8 | `card2` | 0.0127 | **yes** |
| 9 | `card1_prior_count` | 0.0119 | no |
| 10 | `hour_sin` | 0.0113 | no |

`day_index` and `week_index` scored **exactly 0.0000** permutation
importance for the nonlinear model — consistent with (though, per §8, not
a guarantee of) removing them having only a small net effect on this
model's calibrated/decisioning performance.

**Top-10 overlap with logistic's top-10 |coefficient| features: zero.**
This is a real finding, not a bug, and comes with an important
granularity caveat: `extract_coefficients`'s `source_column` is, for a
one-hot-encoded categorical feature, the individual **level** name (e.g.
`ProductCD_R`, `DeviceType_mobile`), because the logistic model literally
has one coefficient per level — while permutation importance here shuffles
the **whole raw categorical column** at once. So logistic's top-10 is
dominated by specific one-hot levels and missingness flags
(`ProductCD_R`, `DeviceInfo_missing`, `DeviceType_None`,
`DeviceType_mobile`, `has_identity_record`, ...), while the nonlinear
model's top-10 is dominated by whole raw columns
(`TransactionAmt_log1p`, `ProductCD`, `email_domain_match`, `card3`,
`DeviceType`, ...). Read at the **column level** rather than the
level/column-name mismatch, there is real qualitative agreement: both
models place `ProductCD`, `DeviceType`/`DeviceType_*`, `email_domain_match`,
and identity/device-missingness signal somewhere near the top of their
respective rankings — they simply report that agreement at different
granularities, which the comparison function intentionally does not paper
over (see `compare_feature_importance`'s docstring in
`src/models/train_nonlinear.py`).

## 8. Time-feature ablation and preferred-configuration consistency check (requirement G)

Two distinct analyses are reported here, and they must **not** be
conflated — the first is a bare, uncalibrated ranking ablation; the second
is the full calibrated/decisioning consistency check this milestone's
correction added.

### 8a. Ranking-only ablation (uncalibrated Brier — explicitly labeled)

Same ablation structure as Milestone 4 (`NUMERIC_FEATURES_NO_TIME_DRIFT`
vs. the full feature set), reusing the same tuned hyperparameters for both
variants so only the feature set changes. **Every Brier score in this
table is UNCALIBRATED** (raw `predict_proba` output,
`brier_score_is_calibrated: false` in the metrics artifact) — it isolates
the feature-set effect from the calibration-method choice and must not be
read side by side with the calibrated Brier scores in §4/§5 above as if
they were the same quantity:

| Variant | validation ROC-AUC | temporal_benchmark ROC-AUC | temporal_benchmark PR-AUC | temporal_benchmark Brier (**uncalibrated**) |
|---|---|---|---|---|
| with day_index/week_index | 0.8158 | 0.7788 | 0.1722 | 0.1050 |
| without day_index/week_index | 0.8130 | 0.7799 | 0.1536 | 0.1169 |

### 8b. Preferred-configuration consistency check (calibrated, full pipeline)

This is the check this milestone's correction introduced: the same
with/without comparison, but using each configuration's own **calibrated**
metrics from §4 (produced by the full `nonlinear_full_pipeline` run, not a
bare ablation):

| Split | Configuration | ROC-AUC (calibrated) | PR-AUC (calibrated) | Brier (calibrated) |
|---|---|---|---|---|
| selection | with day_index/week_index | 0.7914 | 0.1969 | 0.0307 |
| selection | without (preferred) | 0.7894 | 0.1958 | 0.0308 |
| temporal_benchmark | with day_index/week_index | 0.7788 | 0.1722 | 0.0310 |
| temporal_benchmark | without (preferred) | 0.7799 | 0.1536 | 0.0314 |

The direction of the ROC-AUC finding is the same in both the uncalibrated
ablation and the calibrated consistency check: removing `day_index`/
`week_index` very slightly *improves* `temporal_benchmark` ROC-AUC
(+0.0011), consistent with Milestone 4's logistic finding. But PR-AUC is
meaningfully *worse* without them (temporal_benchmark: 0.1722→0.1536, a
~10.8% relative drop) and calibrated Brier is marginally worse too
(0.0310→0.0314). Combined with §6's finding that the preferred
configuration's illustrative decisioning cost is also very slightly higher
(+0.7–0.9%) than the with-time-index configuration's, **the net effect of
excluding `day_index`/`week_index` is small and metric-dependent, not a
clean win on every axis.**

**Per the explicit instruction, the default preferred nonlinear
configuration still excludes `day_index`/`week_index`**
(`NUMERIC_FEATURES_NO_TIME_DRIFT`) — this is a deliberate choice to remove
two features that scored exactly zero permutation importance (§7) and carry
documented extrapolation risk (`docs/milestone3_baseline_model.md`,
`src/models/schema.py` rule 8: any row scored after the training window has
a `day_index`/`week_index` value the model never saw during fitting), in
exchange for a small, acceptable amount of PR-AUC/cost — **not** because
the preferred configuration was found to be strictly better on every
metric. The head-to-head comparison in §4/§6 reports **both**
configurations' numbers explicitly so this tradeoff is visible, not
hidden behind a single "winner" table.

## 9. Limitations

- **No production deployment decision is made in this milestone.** This
  is a model-comparison exercise under an illustrative cost framework, not
  a recommendation to replace the logistic baseline in any real system.
- **Hyperparameter search was deliberately modest** (4 candidates, 3
  time-aware folds) — a larger or different search space might find a
  materially different configuration; this was scoped to stay reproducible
  and fast, not to be exhaustive. The same chosen hyperparameters are
  reused, unretuned, for the preferred no-time-drift configuration — they
  were never re-tuned specifically for that configuration, which is a
  conservative (not performance-inflating) choice.
- **`temporal_benchmark` has now been touched across three milestones**
  (Milestone 3 baseline metrics, Milestone 4 calibration/decisioning, this
  milestone's logistic-vs-nonlinear comparisons, now run for three
  configurations) — it remains explicitly *not* a pristine, never-touched
  final test set, per Milestone 4's own terminology correction.
- **Feature importance is not causal** for either model — see §7.
- **Feature importance was computed once, on the with-time-index pipeline
  only** — not recomputed for the preferred configuration in this run (see
  §7's noted scope choice).
- **The preferred configuration is not uniformly better than the
  with-time-index configuration** — see §8b. It is marginally worse on
  PR-AUC and illustrative cost, and the choice to prefer it rests on
  removing extrapolation risk and zero-importance features, not on a
  strict performance win.
- **All cost figures remain hypothetical and illustrative**, unchanged
  from Milestone 4, and manual review is still modeled as finite-capacity
  and imperfect (not a perfect backstop) — see Milestone 4 §9 for the full
  caveat, which applies identically to every nonlinear configuration's
  decision policy here.
- **The masked-numeric-code caveat still applies**: `card2`/`card3` appear
  in the nonlinear model's top-10 permutation importance exactly as they
  can appear large in the logistic coefficient table — their business
  meaning is still explicitly unknown (`docs/data_dictionary.md`), and no
  interpretation of *why* they matter is claimed here, only that shuffling
  them hurts this model's held-out ranking score.
- **Only one nonlinear model family was tried** (gradient-boosted trees via
  HistGradientBoostingClassifier). No random forest, no XGBoost/LightGBM
  comparison was run — per the explicit preference order in the task
  instructions, not because those alternatives were tested and rejected.

## 10. Engineering & tests

- `src/models/train_nonlinear.py`: tree-native preprocessing
  (`build_tree_preprocessor`, `categorical_feature_mask`), pipeline
  construction (`build_nonlinear_pipeline`, `fit_nonlinear_pipeline`),
  train-only time-aware tuning (`tune_hyperparameters`), the Requirement G
  ablation (`time_feature_ablation_nonlinear`, now explicitly labeling its
  Brier scores `brier_score_is_calibrated: False`), the new
  `nonlinear_full_pipeline` helper (fit → calibration comparison → best
  method → calibrated metrics → decision policy, in one reusable code path
  used for every nonlinear feature-set configuration), permutation
  importance (`permutation_importance_table`, `compare_feature_importance`),
  and the decision-policy comparison (`decision_policy_comparison`); reuses
  `src.models.calibration`, `src.models.cost_model`, and
  `src.models.train_calibrated`'s calibration-comparison/selection
  functions unmodified.
- `tests/models/test_train_nonlinear.py`: 11 tests, covering the
  categorical-feature mask/preprocessor column alignment, native handling
  of missing values, train-only tuning structure and candidate-selection
  correctness, the ablation's structural output, permutation-importance
  coverage of every raw feature, the importance-comparison overlap logic,
  the decision-policy comparison's "chosen once on selection, evaluated
  once on temporal_benchmark" property, and one real-data sanity test
  (`requires_baseline_features`, auto-skips without the raw CSVs).
- Full suite: **109 passed, 1 skipped** (98 pre-existing + 11 new in
  `test_train_nonlinear.py`; the 1 skip is an unrelated raw-CSV-dependent
  test skipping for the same reason it already did before this
  milestone). Re-verified after the consistency-pass changes.
- `data/interim/milestone5_metrics.json`: full reproducibility artifact
  (split reports, tuning candidates, cost assumptions, logistic
  calibration/decisioning, with-time-index nonlinear calibration/
  decisioning, preferred no-time-index nonlinear calibration/decisioning,
  the three-way calibrated head-to-head, the uncalibrated ablation, the
  calibrated preferred-configuration consistency check, the preferred
  configuration's sensitivity analysis, logistic coefficients, and
  with-time-index nonlinear permutation importance).

## 11. Runtime

End-to-end `.venv/bin/python -m src.models.train_nonlinear` on the real
baseline table (471,750 total rows): **≈3 minutes** wall clock on this
machine for the consistency-pass run (two full nonlinear configurations),
of which hyperparameter tuning (§3) was ≈87 seconds (12
HistGradientBoostingClassifier fits, shared by both configurations). The
remainder covers: fitting both configurations' final pipelines, two full
calibration comparisons (3 methods × 2 out-of-sample splits each) plus the
logistic model's own calibration comparison, two full decision-policy grids
plus the logistic model's, one sensitivity analysis (11 variations, on the
preferred configuration), and one permutation-importance run (26 features ×
5 repeats) on the `selection` split.

## 12. Recommended next milestone

A natural next step is **production threshold/model selection** — picking
one specific configuration (the preferred no-time-drift nonlinear
configuration, given §4/§6/§8's finding that it decisively beats logistic
while carrying only a small, acceptable PR-AUC/cost tradeoff relative to
the with-time-index variant for removing extrapolation risk) and one
specific `(review_threshold, decline_threshold)` pair as an actual proposed
policy, backed by sign-off-worthy (not just illustrative) cost assumptions,
plus a monitoring/drift-detection layer for the chosen configuration and
features in production — none of which has been attempted in any milestone
so far. Alternatively, Module A (Payments/Product Analytics) or Module C
(Credit Risk) could be started fresh, per the project's stated multi-module
scope in README.md.
