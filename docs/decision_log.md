# Decision Log

Running record of non-trivial decisions and the reasoning behind them.
Append-only; do not rewrite history, add corrections as new entries.

## 2026-09-30 — Milestone 1 started

- **Dataset**: IEEE-CIS Fraud Detection (Kaggle competition, hosted by IEEE
  CIS, data from Vesta Corporation). Public competition data — explicitly
  not from any employer. Grain: one row per transaction. Target: `isFraud`.
- **Python version**: pinned to 3.12.7 via `pyenv`/`.python-version`, not the
  system default 3.14.7. Reason: 3.14 is too new for guaranteed stable wheels
  across the full pandas/numpy/scikit-learn/statsmodels stack at this time;
  3.12 is a safer choice for a reproducible data science environment.
- **SQL engine**: DuckDB, per the stated tech stack. Used to query raw CSVs
  in place (via `read_csv_auto` + `SUMMARIZE`) rather than loading full wide
  tables into pandas for inspection — the transaction table is ~394 columns
  and loading it fully just to compute null percentages is wasteful.
- **Config-driven paths**: `configs/paths.json` + `src/data/config.py`,
  using stdlib `json` rather than adding a YAML dependency, to keep the
  dependency list minimal per project constraints.
- **`tabulate` added as a dependency**: needed for `DataFrame.to_markdown()`
  used in generated reports. Small, single-purpose addition.
- **Kaggle download not automated**: the dataset requires a Kaggle account
  login and competition rules acceptance. Automating this would mean
  handling the user's credentials, which is out of scope for this
  assistant. The user downloads the four CSVs manually into `data/raw/`.
- **No modeling, no feature engineering, no committed train/test split** in
  this milestone, per explicit scope constraint. Time-based split reasoning
  is documented (`docs/time_structure.md`) but not implemented.
- **Leakage review treats `C`, `D`, `V` feature groups as "unresolved,
  investigate before use"** rather than either trusting them or dropping
  them. Reason: we cannot verify Vesta's computation methodology (point-in-
  time vs. global aggregation) from documentation alone, and the project
  rule is not to fabricate assumptions about undocumented behavior.

## 2026-09-30 — Milestone 1 inspection run against real data

- Raw CSVs placed in `data/raw/` and `python -m src.data.run_all` executed
  successfully, generating `docs/data_profile.md`, `docs/join_validation.md`,
  `docs/time_structure.md`, and `docs/leakage_checks_empirical.md` from the
  actual IEEE-CIS data (not synthetic/placeholder figures).
- Fixed a documentation typo: `docs/data_dictionary.md` listed the amount
  column as `TransactionAMT`; the real column name confirmed via
  `SUMMARIZE` is `TransactionAmt`.
- Full pytest suite run with raw data present: 9 passed, 1 skipped (the
  skipped test only exercises the missing-file error path and correctly
  skips when files are present).
- No column crossed the |corr| >= 0.95 suspicious-correlation threshold
  with `isFraud`. This is evidence against blatant target leakage via a
  single numeric column, but it does **not** resolve the open question
  about whether `C`/`D`/`V` features were computed using future/global
  information — correlation strength and leakage are different questions;
  a leaky feature can have low linear correlation and a non-leaky feature
  can have high correlation. That investigation remains open for a later
  milestone.
- **Found a schema-consistency issue, not in any doc until now**:
  `train_identity`'s `id_*` columns are named with underscores (`id_01`
  ... `id_38`), but `test_identity`'s equivalent 41 columns are named with
  hyphens (`id-01` ... `id-38`). Same columns, same count, different
  delimiter — a well-known IEEE-CIS quirk, confirmed empirically here via
  `information_schema`/`DESCRIBE`, not assumed from outside knowledge. This
  is not leakage, but any future feature-engineering code that reads both
  train and test identity tables must normalize this naming difference
  first, or test-time feature generation will silently produce all-null
  `id_*` columns.

## 2026-09-30 — Milestone 2: point-in-time feature engineering & baseline dataset

- Built `src/features/` (`schema.py`, `amount_time_features.py`,
  `identity_features.py`, `categorical_features.py`,
  `velocity_features.py`, `build_baseline_dataset.py`) and
  `data/processed/baseline_features_v1.parquet` (train only, 590,540 rows,
  38 columns). Full reasoning in `docs/milestone2_feature_design.md`;
  per-feature detail in `docs/feature_dictionary.md`.
- **Applied the `id_*`/`id-*` normalization found in Milestone 1**:
  `src/features/schema.py::normalize_identity_columns` fixes the rename.
  Applied to `train_identity` here even though only train is used this
  milestone, so the same function is safe to reuse on `test_identity` at
  scoring time later without a silent all-null bug.
- **Frequency/recency features use causal expanding-window aggregation**
  (sort by `TransactionDT`, then `groupby(key).cumcount()` /
  `.diff()`), not global aggregation, specifically to avoid the classic
  Kaggle-fraud-competition leakage bug of counting future rows into a
  "how often have I seen this" feature. This is the highest-leakage-risk
  part of this milestone and is covered by adversarial tests
  (`tests/features/test_velocity_features.py`) that mutate a later row and
  assert earlier rows' feature values don't change.
- **Excluded `M1`–`M9` and raw `id_01`–`id_38` values**, even though
  neither was explicitly forbidden the way `C*`/`D*`/`V*` were. Neither was
  requested in the Milestone 2 feature list, and both carry the same
  documented-category-but-undocumented-per-column-meaning problem as the
  explicitly-masked groups. Treated as "not yet in scope," not added just
  because nothing technically forbade it. Full reasoning in
  `docs/milestone2_feature_design.md` §6.
- **Output format: Parquet via DuckDB's native writer** (`COPY ... TO
  ... (FORMAT PARQUET)`), not `pandas.to_parquet` / CSV. Avoids adding a
  `pyarrow`/`fastparquet` dependency (project constraint: minimal
  dependencies) while still getting typed columnar storage; can be read
  back the same way (`duckdb.sql("SELECT * FROM read_parquet(...)")`).
- **No train/validation split created.** Milestone 1 already established
  the split must be time-based; building it now with nothing yet to use it
  for would be scope creep ahead of the modeling milestone.
- Full pytest suite after this milestone: 36 passed, 1 skipped (same
  intentional skip as Milestone 1 — the missing-raw-file error-path test).

## 2026-09-30 — Milestone 3: chronological split & interpretable baseline

- Built `src/models/` (`schema.py`, `split.py`, `preprocessing.py`,
  `train_baseline.py`) on top of Milestone 2's
  `data/processed/baseline_features_v1.parquet`. Full reasoning in
  `docs/milestone3_baseline_model.md`.
- **Split on whole calendar-day boundaries** (`TransactionDT // 86400`),
  not an arbitrary row index, so no single day has transactions in two
  different segments. Target 60/20/20 train/val/holdout; realized 60.3% /
  20.1% / 19.6% after day-boundary snapping (356,216 / 118,790 / 115,534
  rows). Fraud prevalence varies across segments (3.38% / 3.94% / 3.41%),
  confirming Milestone 1's non-stationarity concern empirically rather
  than just structurally.
- **Column-role policy for model inputs is centralized in
  `src/models/schema.py`**, not scattered across preprocessing code, so
  every include/exclude decision has one auditable location. Key calls:
  - `card1`, `addr1`, `P_emaildomain`, `R_emaildomain`, `DeviceInfo` are
    excluded as raw model inputs in favor of their Milestone-2 derived
    frequency/recency/missingness proxies — avoids several-hundred-column
    one-hot blocks and unseen-category risk at scoring time.
  - `TransactionDT` itself is excluded from the feature matrix (used only
    to build the split): its undisclosed reference point makes a raw
    coefficient on it uninterpretable, and `day_index`/`week_index`/
    `hour_sin`/`hour_cos`/`day_of_week_index` already carry the same
    information at an interpretable granularity.
  - `card2`, `card3`, `card5`, `addr2` (masked numeric codes with no
    Milestone-2-built alternative) are kept as numeric inputs with an
    explicitly flagged, unverified ordinal/interval assumption — not
    excluded outright, since Milestone 3 does not retroactively build new
    Milestone-2-style feature engineering (scope discipline), and not
    silently trusted either.
- **Model: plain `LogisticRegression(class_weight="balanced")`**, not
  resampling (SMOTE/undersampling), to handle ~3.5% fraud prevalence
  without synthesizing rows or losing coefficient interpretability. No
  tree-based/boosted model used, per explicit Milestone 3 scope.
- **No future leakage**: the full preprocessing pipeline (imputer
  medians, scaler stats, one-hot vocabularies) is fit on the train split
  only, via a single `sklearn.pipeline.Pipeline`; validation/holdout only
  ever go through `.transform`. Verified by
  `tests/models/test_preprocessing.py` and
  `tests/models/test_train_baseline.py`, not just asserted by
  construction.
- **Threshold analysis run on validation only, never on holdout** — holdout
  is touched exactly once, for final evaluation. No threshold is labeled
  "best": that requires a business cost function (relative cost of a
  missed fraud vs. a false alert) that was not defined in this milestone.
- **Found empirically, not anticipated going in**: `class_weight=
  "balanced"` produces well-ordered scores (ROC-AUC 0.765 validation,
  0.732 holdout) but badly miscalibrated probabilities (Brier 0.260
  validation, 0.316 holdout; calibration curve shows predicted
  probabilities several multiples above observed fraud rates in every
  decile bin). Documented explicitly in
  `docs/milestone3_baseline_model.md` §8 as a limitation — these scores
  must not be treated as true fraud probabilities without a
  recalibration step not built here.
- Full pytest suite after this milestone: 58 passed, 1 skipped (same
  intentional skip as Milestones 1–2).

## 2026-09-30 — Milestone 4: probability calibration & illustrative fraud decisioning

- Built `src/models/calibration.py`, `configs/cost_assumptions.json` +
  `src/models/cost_model.py`, and `src/models/train_calibrated.py` on top
  of Milestone 3's frozen split and baseline pipeline. Full reasoning in
  `docs/milestone4_calibration_decisioning.md`. No boosted-tree/forest/
  neural-network model trained, per explicit scope.
- **Hand-rolled sigmoid/isotonic calibrators instead of
  `CalibratedClassifierCV(cv="prefit")`**. Reason: this project pins
  `scikit-learn>=1.4,<2`, a range where `cv="prefit"` support has shifted
  across releases; a small, directly testable calibrator class (`.fit`/
  `.transform`) avoids depending on a specific sklearn version's internal
  behavior and makes the "fit on validation only" guarantee trivial to
  unit test by mutating holdout after fitting.
- **Calibrators fit on validation raw probabilities + labels only; holdout
  only ever `.transform`-ed, never `.fit`-ed** — same no-future-leakage
  discipline Milestones 2–3 established, now extended to calibration.
  Verified by `tests/models/test_calibration.py` and
  `tests/models/test_train_calibrated.py` (mutating holdout after fitting
  must not change any validation-side number).
- **Isotonic selected over sigmoid** by lowest validation Brier score
  (0.0356 vs. 0.0359) — a narrow margin; both cut Brier score by roughly
  7–9x versus the uncalibrated score on both validation and holdout,
  while ROC-AUC stayed within noise (sigmoid exactly unchanged by
  construction; isotonic moved by at most ~0.005 due to tie-induced flat
  regions) — the clean separation of ranking from calibration quality
  this milestone set out to demonstrate.
- **Three-action decisioning (`approve`/`review`/`decline`) models manual
  review as a perfect fraud backstop** — `fraud_recall` counts review the
  same as decline. Explicit simplification, not hidden: this project has
  no empirical review-team accuracy data to build a more realistic
  partial-catch-rate assumption. Flagged as a limitation in
  `docs/milestone4_calibration_decisioning.md` §9.
- **Illustrative cost assumptions ($500 missed fraud / $25 false-positive
  decline / $3 manual review) are config-driven**
  (`configs/cost_assumptions.json`), carry an inline disclaimer, and are
  explicitly NOT claimed to represent Vesta Corporation, IEEE-CIS, a real
  bank, or any real payments company anywhere in code or docs.
- **Found empirically, not anticipated going in**: at the chosen base
  costs, manual review ($3) is cheap enough relative to both alternatives
  that simple ±2x variations on any single cost input never changed the
  argmin-cost threshold policy — only the reported total cost moved. Two
  more extreme, targeted variations were added specifically to
  demonstrate requirement E's point: raising manual-review cost to $60
  (above the $25 decline cost) flips the policy toward outright decline
  and drops fraud recall from 0.640 to 0.367; dropping missed-fraud cost
  to $5 collapses the policy to all-approve (recall 0). This is treated
  as a genuine finding about this specific cost configuration (locally
  flat near current assumptions, but moves sharply past a break-even
  point), not a flaw in the sensitivity mechanism — see
  `docs/milestone4_calibration_decisioning.md` §7.
- **Found empirically, not anticipated going in**: removing
  `day_index`/`week_index` (the two linear time-drift features Milestone 3
  flagged as an extrapolation risk) did not hurt holdout performance — it
  slightly improved holdout ROC-AUC (0.7317→0.7351) and holdout PR-AUC
  (0.1202→0.1429). Treated as suggestive from a single split, not
  conclusive, but it weakens the case for keeping these features given
  they already carried a documented deployment-risk concern with no
  offsetting performance benefit found here.
- **Threshold/cost exploration run on validation only; holdout touched
  once** for a single evaluation of the validation-selected policy, never
  re-optimized there — holdout has now been touched once each in
  Milestones 3 and 4 and would need to be treated as contaminated if any
  future milestone revisits modeling choices against this same slice.
- Full pytest suite after this milestone: 86 passed, 1 skipped (same
  intentional skip as Milestones 1–3).

## 2026-10-02 — Milestone 4 methodology correction: leakage-safe calibration evaluation + realistic review model

**Supersedes the numbers in the 2026-09-30 Milestone 4 entry above; that
entry is left unedited per the append-only convention, but its calibration
comparison and cost-sensitivity numbers should no longer be cited.**

- **Bug found**: the original Milestone 4 run fit each calibrator on the
  validation period and then reported that calibrator's Brier score /
  reliability table on the SAME validation period — an in-sample,
  optimistically-biased evaluation. This was most visible for isotonic
  regression, whose "validation" reliability table had `mean_predicted_prob`
  exactly equal to `observed_fraud_rate` in every bin, a direct symptom of
  evaluating a non-parametric fit on its own training data.
- **Fix**: `src/models/split.py::chronological_split_two_way` further
  divides Milestone 3's validation period, chronologically, into `cal_fit`
  (earlier half, day ≤121: 61,343 rows, 4.37% fraud) and `selection` (later
  half, day ≥122: 57,447 rows, 3.48% fraud) — the original
  train/validation/holdout day boundaries are untouched. Calibrators are
  `.fit()` on `cal_fit` only; Brier score, reliability tables, and the
  cost-based threshold/cost grid are all evaluated/selected on `selection`,
  which no calibrator or policy ever saw during fitting.
- **Corrected result reverses the original calibrator choice**: evaluated
  out-of-sample, isotonic's Brier score on `selection` (0.03282) is
  *worse*, not better, than sigmoid's (0.03247), and isotonic is worse on
  the temporal benchmark too (0.03519 vs. sigmoid's 0.03298) and on PR-AUC
  on both splits. **Sigmoid is now selected** — confirming the suspicion
  that isotonic's original apparent edge (0.0356 vs. 0.0359, in-sample) was
  an overfitting artifact of evaluating it on its own fit data, not a real
  calibration advantage. `select_best_calibration_method` now defaults to
  preferring sigmoid unless isotonic beats it by at least 3% relative Brier
  improvement on the out-of-sample `selection` split; here isotonic is
  behind, not ahead, so the rule doesn't need to activate.
- **Terminology correction**: the final chronological segment, called
  "holdout" through Milestone 3 and the original Milestone 4 run, is
  renamed **temporal benchmark** everywhere (code, tests, docs). It has now
  been inspected three times during iterative development (Milestone 3
  baseline metrics, original Milestone 4 calibration/policy evaluation,
  this correction's calibration/policy re-evaluation) and must not be
  presented as a pristine, never-touched final unbiased test set — this is
  now stated explicitly in `docs/milestone4_calibration_decisioning.md` and
  `README.md` rather than left implicit.
- **Review-realism correction to `src/models/cost_model.py`**: the original
  cost model treated manual review as having unlimited capacity and a
  perfect fraud-catch rate. `CostAssumptions` now has three new, independently
  configurable fields — `review_capacity_frac`, `review_fraud_catch_rate`,
  `review_false_positive_rate` — all defaulting to the old unlimited/perfect
  behavior so pre-correction callers are unaffected. At the illustrative
  base assumptions (5% capacity, 80% catch rate, 5% review error rate), the
  `selection`-split lowest-cost policy (review≥0.05, decline≥0.3) wants to
  review 12,652 of 57,447 transactions (22.0%) but capacity caps this at
  2,872 (5%): the 2,872 **highest-probability** candidates are kept and the
  other 9,780 fall back to **approve, not decline** (declining them outright
  would be stricter than the threshold pair itself specifies). This capacity
  constraint alone cuts fraud recall from what it would otherwise be;
  tightening capacity further to 1% (`review_capacity_tightened`
  sensitivity variation) drops recall from 0.1804 to 0.0616, and degrading
  the review catch rate to 40% (`review_catch_rate_low`) drops it to 0.0902
  — both demonstrate the realistic-review-model correction has real teeth,
  not just a theoretical one.
- **New, incidental finding from the correction**: under sigmoid
  calibration, no transaction in the `selection` split reaches a calibrated
  probability of 0.3, so `decline_volume` is 0 across the *entire* 43-pair
  threshold/cost grid — every lowest-cost policy found here acts solely
  through (capacity-limited) review, never outright decline, at these
  illustrative costs. This is a property of this specific calibrated-score
  distribution and cost configuration, not a modeling bug.
- Full pytest suite after this correction: 98 passed, 1 skipped (same
  intentional skip carried from Milestones 1–3, plus 12 net new tests for
  `chronological_split_two_way`, the review-realism cost-model parameters,
  and the corrected calibration-comparison/selection-method tests).

## 2026-10-02 — Milestone 5: transparent nonlinear model comparison

- Added `src/models/train_nonlinear.py`
  (`HistGradientBoostingClassifier`) on top of Milestone 3-4's frozen
  split, cal_fit/selection sub-split, calibration methodology, and cost
  framework — all reused unchanged. Full reasoning in
  `docs/milestone5_nonlinear_comparison.md`. No masked `C*`/`D*`/`M*`/`V*`/
  `id_*` feature introduced; no deep learning, no graph model, no Kaggle
  test labels used.
- **`HistGradientBoostingClassifier` chosen over XGBoost/LightGBM**: no
  dependency/runtime tradeoff was found that justified a new dependency —
  it already ships with the pinned `scikit-learn>=1.4,<2` requirement,
  handles missing values and categorical splits natively, and the whole
  milestone (tuning + fitting + calibration + decisioning + importance, on
  the real ~472k-row table) ran in ≈3m12s wall clock.
- **A dedicated tree-native preprocessor
  (`build_tree_preprocessor`) was built instead of reusing
  `src.models.preprocessing.build_preprocessor` as-is**: scaling is
  irrelevant to a tree's threshold splits, and one-hot-encoding a
  categorical column would throw away the native categorical-split
  capability that's one of this model's actual advantages. Numeric columns
  pass through un-imputed (HGB treats missingness as its own learned split
  direction); categorical columns are ordinal-encoded with an explicit
  `categorical_features` boolean mask telling HGB to treat them as
  discrete categories, not magnitudes. Still the exact same transparent
  feature set as Milestones 3-4 — this is a preprocessing-shape choice,
  not a feature-set change.
- **Hyperparameter tuning kept deliberately modest and train-only**: 4
  explicitly enumerated candidate configurations (not a brute-force grid),
  scored by mean PR-AUC across 3 train-only, time-aware
  (`TimeSeriesSplit`, forward-chaining) folds — never touching
  `cal_fit`/`selection`/`temporal_benchmark`. Winner:
  `max_iter=200, max_leaf_nodes=63, learning_rate=0.05,
  l2_regularization=1.0` (mean PR-AUC 0.2019 vs. 0.1882-0.1978 for the
  other three candidates). Tuning itself took ≈78s (12 fits).
- **Nonlinear model clearly outranks the logistic baseline on identical
  splits**: calibrated ROC-AUC 0.7914 vs. 0.7454 (`selection`), 0.7788 vs.
  0.7317 (`temporal_benchmark`); PR-AUC 0.1969 vs. 0.1273 (`selection`),
  0.1722 vs. 0.1202 (`temporal_benchmark`). Both models generalize forward
  in time with only a small drop from `selection` to `temporal_benchmark`
  — neither is dramatically more fragile to the forward time gap.
- **Sigmoid selected over isotonic for the nonlinear model too**, by the
  same out-of-sample rule Milestone 4 established — isotonic's edge on
  `selection` Brier (0.03067 vs. 0.03075) is under 0.3% relative, nowhere
  near the 3% bar. Notably, the nonlinear model's *uncalibrated* Brier
  score (0.0992) was already far better-behaved than the logistic
  baseline's uncalibrated Brier (0.2609) — `class_weight="balanced"`
  distorts this model's raw probability output less severely — but
  calibration still clearly helps both (0.0992→0.0307 nonlinear,
  0.2609→0.0325 logistic), so it remains necessary for both, not optional
  for either.
- **Under identical illustrative cost assumptions, the nonlinear model's
  own selection-chosen policy (same `(0.05, 0.30)` threshold pair as the
  logistic model's) roughly doubles fraud recall** (e.g. 0.180→0.345 on
  `selection`) **and lowers illustrative total cost by 15-18%** on both
  `selection` and `temporal_benchmark` — but with materially more false
  positives (121→771 on `selection`), a visible tradeoff, not a free win.
  Review volume is identical between models (same `review_capacity_frac`
  applied to the same row counts); the nonlinear model's extra recall
  comes from its decline channel and from reviewing a more fraud-dense
  subset within the same capacity, not a larger review budget. Not a
  production policy recommendation — same hypothetical-cost and
  finite/imperfect-review caveats as Milestone 4 apply unchanged.
- **Zero top-10 overlap between logistic |coefficient| and nonlinear
  permutation-importance rankings — a granularity artifact, not a real
  disagreement**: `extract_coefficients`'s `source_column` is a one-hot
  *level* name for categorical features (e.g. `ProductCD_R`), while
  permutation importance shuffles the whole raw column (`ProductCD`).
  Read at the column level, both models independently surface
  `ProductCD`, `DeviceType`, `email_domain_match`, and identity/device
  missingness signal near the top — real qualitative agreement, reported
  at different granularities on purpose rather than papered over.
- **`day_index`/`week_index` ablation repeated for the nonlinear model,
  same qualitative finding as Milestone 4's logistic ablation**: excluding
  them very slightly improves `temporal_benchmark` ROC-AUC (0.7788→0.7799)
  though PR-AUC/Brier move slightly the other way; both features scored
  exactly 0.0000 permutation importance. No evidence here that they earn
  their documented extrapolation risk. Per explicit instruction, the
  default preferred nonlinear configuration excludes them
  (`NUMERIC_FEATURES_NO_TIME_DRIFT`); the head-to-head comparison above
  uses the full feature set to stay apples-to-apples with the
  already-established Milestone 3/4 logistic baseline specifically.
- `temporal_benchmark` has now been touched a fourth time across
  Milestones 3-5 (Milestone 3 baseline metrics, Milestone 4 original
  calibration/policy evaluation, Milestone 4's correction re-evaluation,
  this milestone's logistic-vs-nonlinear comparison) — still explicitly
  not a pristine, never-touched final test set.
- Full pytest suite after this milestone: 109 passed, 1 skipped (98
  carried forward + 11 new tests in `tests/models/test_train_nonlinear.py`;
  same intentional skip carried from Milestones 1-4).

## 2026-10-02 — Milestone 5 consistency-pass correction: preferred-configuration metrics were not actually run

- **Bug found on review**: the original Milestone 5 report recommended the
  no-`day_index`/`week_index` nonlinear configuration as "preferred" but
  the headline calibrated/decisioning tables (§4/§6 of
  `docs/milestone5_nonlinear_comparison.md`) showed the *with*-time-index
  configuration's numbers — those two configurations are not numerically
  identical, so the recommendation was not actually backed by the metrics
  displayed. The `day_index`/`week_index` ablation that *was* run
  (`time_feature_ablation_nonlinear`) only ever reported bare, uncalibrated
  ROC-AUC/PR-AUC/Brier — it never calibrated, never ran decision-policy
  comparison, for either feature-set variant.
- **Fix**: added `nonlinear_full_pipeline` to `src/models/train_nonlinear.py`
  — one shared code path (fit on train → raw probs on cal_fit/selection/
  temporal_benchmark → sigmoid/isotonic comparison fit on cal_fit only →
  `select_best_calibration_method` → calibrated metrics via `evaluate_probs`
  → `decision_policy_comparison` using the SAME `costs` object/threshold
  grids for every caller). `main()` now calls this once for the
  with-time-index feature set and once for the no-time-drift (preferred)
  feature set, so both configurations' calibrated and decisioning numbers
  come from the identical pipeline and the identical cost assumptions —
  neither was changed to favor either configuration.
- **`time_feature_ablation_nonlinear`'s uncalibrated Brier is now
  explicitly labeled**: each summary dict carries
  `brier_score_is_calibrated: False`, and the doc's ablation table (§8a) is
  now clearly separated from a new §8b "preferred-configuration consistency
  check" table that reports the same with/without comparison using
  CALIBRATED metrics from the full pipeline — the two must not be read as
  the same quantity.
- **Corrected finding**: the preferred (no-time-drift) configuration's own
  numbers are very close to, but not identical to, the with-time-index
  configuration's: calibrated temporal_benchmark ROC-AUC is marginally
  *better* without the time features (0.7788→0.7799), but PR-AUC is
  meaningfully *worse* (0.1722→0.1536, ~10.8% relative) and illustrative
  decisioning cost is marginally *higher* (+0.7% on `selection`, +0.9% on
  `temporal_benchmark`) than the with-time-index configuration's own cost.
  Against the **logistic baseline**, the preferred configuration still wins
  decisively (fraud recall 0.180→0.337 on `selection`, 0.170→0.316 on
  `temporal_benchmark`; illustrative cost down 17.3%/14.5%) — slightly less
  of a win than the with-time-index configuration's 17.8%/15.3%, but still
  a clear, large improvement. The net effect of excluding
  `day_index`/`week_index` is small and metric-dependent, not a clean win
  on every axis; it remains the recommended default because both features
  scored exactly 0.0000 permutation importance and carry documented
  extrapolation risk, not because it is strictly better on every metric.
- Hyperparameters were NOT re-tuned for this correction — the same chosen
  configuration (`max_iter=200, max_leaf_nodes=63, learning_rate=0.05,
  l2_regularization=1.0`) is reused unchanged for both feature-set
  configurations, as before. Cost assumptions and threshold grids were not
  changed.
- Full pytest suite after this correction: 109 passed, 1 skipped (same
  count as before the correction — no test needed to change, since
  `main()` itself is not directly unit-tested and all touched function
  signatures (`time_feature_ablation_nonlinear`,
  `decision_policy_comparison`, etc.) kept their existing contracts).

## 2026-10-05 — Module A, Milestone 2: payments experimentation & product decisioning

This is the first Module A entry in this log — Module A Milestone 1's
design decisions (donor-cloned abandonment profiles, KPI/funnel
definitions, rule-based guardrails) are recorded directly in
`docs/payments_data_model.md` rather than retroactively duplicated here.
This entry covers Milestone 2 only.

- **Randomization unit: session, not customer.** IEEE-CIS (and this
  project's existing Module A field inventory, `docs/payments_data_model.md`
  §2a) provides no real, documented customer identifier.
  `card4`/`card6`/email-domain fields are explicitly opaque, undocumented
  categorical keys per Module B's own data dictionary — building a
  customer proxy out of them (e.g. hashing `card4+card6+P_emaildomain`)
  was rejected as fabricating an identity concept this dataset cannot
  actually support, and risking an unaudited rate of hash
  collisions/misses. Session-level randomization was used instead, with
  the resulting contamination risk (the same real person's multiple
  sessions landing in different arms) stated as an explicit, unmeasurable
  limitation rather than hidden. See
  `docs/moduleA_milestone2_experimentation.md` §2.
- **Assignment and treatment-effect draws use deterministic SHA-256
  hashing of `(salt, experiment_seed, session_id)`, not `numpy.random.Generator`.**
  This makes assignment a pure function of `session_id` alone — reorder-
  stable, rerun-stable, and trivially auditable — and avoids the
  row-order sensitivity a stateful RNG would introduce if iterated over a
  DataFrame. Two different salts (`ASSIGNMENT_SALT`, `EFFECT_SALT`) keep
  the assignment draw and the treatment-response draw statistically
  independent despite sharing the same `session_id` and `experiment_seed`.
- **Synthetic treatment-effect mechanism scope, deliberately narrow**:
  only flips `is_synthetic_abandonment` sessions (never real transaction
  history, in either arm) from `abandoned` to `authorized`, only in the
  treatment arm, only when `attempt_count >= eligible_min_attempt_count`,
  only via an independent Bernoulli draw at `recovery_uplift_prob`. This
  was chosen so the mechanism is easy to audit end-to-end and so its
  predicted downstream effects (movement in `retry_recovery_rate` and
  `attempts_per_completed_session`, no movement in
  `first_attempt_authorization_rate`, `retry_rate`, or any guardrail
  metric) are mechanically derivable in advance, not just empirically
  observed after the fact. `payment_attempts` rows are deliberately NOT
  mutated when a session's outcome flips — a stated simplification, not
  an oversight.
- **`recovery_uplift_prob` (0.10) and all guardrail tolerances /
  `min_practical_uplift_abs` (0.01) in
  `configs/payments_experiment_assumptions.json` were fixed BEFORE
  running `src.payments.build_experiment_report` against the real IEEE-
  CIS data, and were not adjusted afterward.** The real-data run's
  `no-ship` result (uplift below the practical threshold, not a
  guardrail breach) is reported as obtained, per the explicit instruction
  not to tune treatment assumptions post-hoc to force significance.
- **`legitimate_decline_proxy_rate` defined as a proxy, not a model
  FPR**: share of real-linked, non-fraudulent sessions with at least one
  `decline_reason_code == 'risk_block'` attempt. Documented as expected
  to show no real between-arm signal, since `decline_reason_code` is
  sampled independently of `isFraud` in Milestone 1's generator — mirrors
  the "methodological artifact" callout style already established for
  Milestone 1's `velocity_flag` backtest.
- **Segment heterogeneity restricted to 4 pre-specified dimensions**
  (`DeviceType`, `ProductCD`, `card6`, `retry_eligible`), enforced by an
  allow-list that raises `ExperimentSegmentError` on any other column —
  deliberately not an open-ended slicing tool. No multiple-comparison
  correction is applied; this is recorded as an accepted limitation
  specific to this small, pre-registered set, not a general license to
  skip correction when slicing more broadly later.
- **SRM implemented as a one-sample `proportions_ztest`** (observed
  treatment count vs. expected fraction) rather than importing a
  separate chi-square goodness-of-fit routine, to avoid adding a new
  explicit dependency beyond the already-used `statsmodels`.
- **Real-data run result** (`.venv/bin/python -m src.payments.build_experiment_report`,
  679,121 sessions, `experiment_seed=20261005`): crossover check clean
  (679,121 unique session IDs, no duplicates); SRM check did not flag
  (observed treatment fraction 0.5009 vs. expected 0.5, p=0.149);
  pre-treatment covariate balance close across arms. Primary metric:
  control completion rate 0.8692 (294,644/338,966), treatment 0.8763
  (298,083/340,155), absolute uplift +0.0071 (relative +0.81%), 95% CI
  [0.0055, 0.0087], p ≈ 2.2e-18 — statistically significant and positive.
  All 4 guardrail metrics within configured tolerance (largest observed
  absolute diff 0.0015, on `legitimate_decline_proxy_rate`, tolerance
  0.005). **Decision: no-ship** — the only failing condition was
  `practical_uplift_clears_threshold` (0.0071 < 0.01), a direct,
  real-data illustration of statistical significance without practical
  significance. Segment heterogeneity broadly echoed the primary result
  across `DeviceType`/`ProductCD`/`card6` values (uplift ≈ 0.005-0.012,
  mostly significant given large per-segment n), and — consistent with
  the treatment mechanism's design (§ above) — the effect concentrated
  almost entirely in `retry_eligible = True` sessions (uplift +0.0214,
  p ≈ 5.7e-38) versus `retry_eligible = False` sessions (uplift
  +0.0004, not significant), exactly as the mechanism predicts since it
  can only ever act on retry-eligible sessions.
- Full pytest suite after this milestone: 181 passed, 1 skipped (109
  Module B tests, unchanged, + 72 Module A tests in `tests/payments/` —
  34 carried forward from Milestone 1 plus 38 new Milestone 2 tests
  across `tests/payments/test_experiment_assignment.py`,
  `test_experiment_effects.py`, `test_experiment_diagnostics.py`,
  `test_experiment_analytics.py`, and additions to `test_config.py`;
  same pre-existing skip carried unchanged from Module B).
