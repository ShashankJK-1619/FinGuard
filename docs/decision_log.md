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
  project. The user downloads the four CSVs manually into `data/raw/`.
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

## 2026-10-12 — Module C Milestone 1: credit-risk data model & target-definition foundation

- **No dataset downloaded, no model trained.** This milestone is schema
  and logic design only, validated against toy fixtures in
  `tests/credit_risk/`. Full reasoning in `docs/credit_risk_data_model.md`,
  `docs/credit_risk_dataset_comparison.md`,
  `docs/credit_risk_target_definition.md`, and the consolidated summary
  `docs/moduleC_milestone1_credit_risk_foundation.md`.
- **Dataset recommendation: LendingClub accepted-loans data**, chosen
  over Freddie Mac's Single-Family Loan-Level Dataset (SFLLD — strongest
  alternative, structurally cleaner point-in-time separation via
  separate origination/monthly-performance tables, but mortgage-specific,
  narrower feature set, and registration-gated) and Home Credit Default
  Risk (pre-baked `TARGET` label with no raw status timeline to derive
  censoring logic from, plus a documented history of subtle feature
  leakage in public Kaggle analyses). LendingClub's `loan_status`
  vocabulary maps directly onto this module's default/delinquency
  severity split, it has the richest documented application-time feature
  set of the three, and it has the lowest access friction (commonly
  CC0-licensed mirrors, no account/competition gate).
- **Censoring, not silent zero-filling, for insufficient follow-up**:
  `compute_default_12m_labels` codes a loan with no recorded default but
  also insufficient elapsed follow-up as `label_status =
  "censored_insufficient_followup"` / `default_12m = NA`, never a
  confirmed `0`. Reasoning: a loan that simply hasn't had time to default
  yet is not evidence it will survive the full window; silently coding
  it `0` would bias label distributions toward recently originated loans
  and understate risk for the newest vintages for a reason that has
  nothing to do with their actual creditworthiness.
- **Inclusive window-boundary convention**: a default event landing
  exactly on `origination_date + outcome_window_months` counts as
  "within the window" (`default_12m = 1`), chosen over an exclusive
  boundary because the window length itself is a planning convention,
  not a natural law, and an exclusive boundary would arbitrarily reward
  boundary-case defaults as "survivors."
- **Delinquency and default modeled as separate severities, never
  conflated**: `delinquency_event_date` is carried for visibility only
  and never consulted when setting `default_12m` — a lesser delinquency
  event must never, by itself, flip the default label. This mirrors the
  lifecycle distinction in `docs/credit_risk_data_model.md` Section A
  (delinquency is recoverable; default is the configured
  severe/terminal state).
- **`grade`/`sub_grade`/`int_rate` left flagged `"unknown"`, deliberately
  unresolved**: these fields are technically fixed at-or-before
  origination (not a timing leak in the literal sense) but are the
  lender's own prior risk assessment/pricing — using them as
  independent borrower-fact features risks a model that reproduces
  LendingClub's existing risk model rather than learning default risk
  from first principles. Whether to exclude them entirely, include them
  as an explicit "replicate the lender's pricing policy" baseline, or
  something in between is deferred as an explicit Milestone 2 decision,
  not decided here.
- **Leakage-keyword safety net in `feature_eligibility.py`**: beyond the
  human-authored `available_at_application` judgment in
  `configs/credit_risk_feature_eligibility.json`, `FeatureEligibilityEntry`
  raises `FeatureEligibilityError` if any field name matches a known
  post-origination leakage pattern (`recover`, `total_pymnt`,
  `last_pymnt`, `out_prncp`, `charged_off`, `settlement`, `hardship`,
  `last_fico`, …) while marked `available_at_application = "yes"` — a
  defense against a future config-editing mistake, not a substitute for
  the human judgment that produced the table in the first place.
- Full pytest suite after this milestone: 208 passed, 1 skipped (181
  prior Module A/B tests unchanged + 27 new Module C tests in
  `tests/credit_risk/` across `test_schema.py`, `test_config.py`,
  `test_target_definition.py`, `test_feature_eligibility.py`; same
  pre-existing skip carried unchanged from Module B).

## 2026-10-12 — Module C Milestone 1: documentation corrections (provenance wording + selection-bias limitation)

Two corrections made to the Milestone 1 docs before commit, per review
feedback. No code, config, or test changed — documentation only; dataset
status (not downloaded) and milestone scope are unchanged.

- **Tightened LendingClub provenance wording.** The original wording
  ("widely mirrored under permissive terms, commonly distributed as
  CC0/public-domain") risked implying the *current, original* LendingClub
  download is itself openly available without restriction. Corrected in
  `docs/credit_risk_dataset_comparison.md` and
  `docs/moduleC_milestone1_credit_risk_foundation.md` to state precisely:
  LendingClub does not currently offer unrestricted public download of
  this historical file from its own platform; this project will use one
  specific, reproducible public mirror of the 2007-2018 accepted-loans
  file, and will record that mirror's exact source URL, stated license,
  and a file checksum **at download time** (Module C Milestone 2) rather
  than asserting licensing/openness now. No mirror has been selected or
  downloaded yet.
- **Added an explicit selection-bias / target-population limitation.**
  The LendingClub accepted-loans file contains only originated (accepted)
  loans -- declined applications never appear in it. Documented in a new
  "Selection bias and target-population limitation" section in
  `docs/credit_risk_dataset_comparison.md`, in `docs/
  moduleC_milestone1_credit_risk_foundation.md` Sections 5 and 7, and in
  `README.md`'s Module C business-problem paragraph and Known
  Limitations list: the resulting `default_12m` PD model estimates
  default risk *conditional on having already been approved and
  originated*, and **must not be presented as a complete approve/decline
  underwriting model for the full applicant population**. Correcting
  this would require reject inference against LendingClub's separately
  published rejected-applications file (a thinner table with no
  origination/repayment columns, since none exist for a declined
  application) -- and even then, declined applicants' true repayment
  outcomes are permanently unobserved, since they were never funded. Any
  reject-inference technique substitutes a modeling assumption about
  that unobserved counterfactual for an observed fact; it narrows this
  bias, it does not eliminate it. This limitation is inherent to any
  accepted-loans-only credit dataset, not specific to LendingClub, and
  is left unresolved/out of scope for this milestone -- not committed to
  any future milestone by default.
- Full pytest suite re-run after these documentation-only edits: 208
  passed, 1 skipped (unchanged from Milestone 1's own run, as expected
  since no code/config/test changed).

## 2026-10-12 — Module C Milestone 2: real LendingClub data validation, target construction & chronological split

- **Mirror selected and downloaded**: DePaul University course-data
  mirror (`bigblue.depaul.edu/jlee141/econdata/LendingClub_LoanData/`),
  `LendingClub_2007_to_2018Q4.csv`, 1,675,133,810 bytes, SHA256
  `3eae03c28fd9d2e8a076ebeb73507e8d4d0f44d90500decdb0936e0933d1f36a`
  (`configs/credit_risk_raw_source_metadata.json`,
  `src/credit_risk/raw_source.py`). No account/API key required.
  **License wording kept deliberately cautious**: this mirror shows no
  explicit license statement of its own, only an origin citation to a
  Dropbox folder; a separate Kaggle listing of apparently the same data
  shows a "CC0: Public Domain" tag, but that tag is Kaggle's own claim
  about its own hosted copy and is explicitly NOT treated as inherited
  by this mirror or attributed to LendingClub itself.
  `RawSourceMetadata.__post_init__` raises if
  `license_statement_is_from_original_publisher=True` is ever set
  without an accompanying human-reviewed doc update -- there is
  currently no case where this repo has found a mirror making that
  claim.
- **Footer-row contamination found and fixed via a rejected-first-
  attempt discipline**: the real file has 33 non-loan footer/
  section-marker text rows (e.g. "Total amount funded in policy code 1:
  ..."). The first fix attempted, DuckDB's `ignore_errors=true` with
  sampled type inference, was tried and REJECTED after testing showed
  it silently nulled out real columns on unrelated rows whenever its
  type-sniffing sample missed a rarer value elsewhere in the 1.6GB file
  -- a worse outcome than the 33 known rows it was meant to fix.
  Replaced with `all_varchar=true` (no reader-level casting at all) plus
  an explicit `WHERE regexp_matches(id, '^[0-9]+$')` filter in
  `src/credit_risk/raw_loader.py::register_loans_view`, which isolates
  exactly the 33 bad rows with no collateral risk. Verified independently
  via Python's own quote-aware `csv` module that the underlying file is
  not malformed CSV (every one of 2,260,701 raw data rows has exactly
  151 fields) -- the issue was purely DuckDB's type-sniffing on this
  specific file, not the CSV structure itself.
- **Confirmed empirically: the real file has no genuine default/
  charge-off event-date column.** Per the explicit project rule against
  fabricating event dates, two distinct, separately-named,
  never-conflated targets were built instead of one
  (`src/credit_risk/target_construction.py`):
  `default_terminal_status` (maturity-date-based, no event date
  required, variable 36/60-month horizon) and
  `default_12m_last_pymnt_proxy` (reuses Milestone 1's unmodified
  `compute_default_12m_labels`, fed a proxy event date --
  `last_pymnt_d`, falling back to `issue_d` for the zero-payment
  charge-off edge case). The proxy's bias direction is documented and
  one-directional: LendingClub's real charge-off policy lags
  `last_pymnt_d` by 120+ days, so the proxy can only pull borderline
  defaults INTO the 12-month window, never push them out. Neither
  target is ever presented as, or merged with, the other.
  `INFERRED_DATA_AS_OF_DATE = 2019-04-01` is derived from the maximum
  observed `last_credit_pull_d` across the file, stated as an inferred
  proxy for the snapshot date, not an asserted fact from the source.
- **Real `loan_status` vocabulary (9 values) mapped explicitly**,
  replacing Milestone 1's secondary-source guess
  (`configs/credit_risk_status_mapping.json`,
  `src/credit_risk/status_mapping.py`). The two "Does not meet the
  credit policy." statuses are mapped to the SAME outcome category as
  their unprefixed counterpart, with policy vintage tracked separately
  via `meets_credit_policy` -- never conflated with outcome severity.
  `default_event_statuses` was updated to include "Does not meet the
  credit policy. Status:Charged Off" (a real charge-off) alongside
  "Charged Off"/"Default"; "Late (31-120 days)" remains deliberately
  excluded from the trigger set despite sharing a category with
  "Default", since it is a recoverable, not-yet-terminal severity state.
- **Leakage audit rebuilt for all 151 real columns**, replacing
  Milestone 1's ~29-entry provisional guess
  (`configs/credit_risk_feature_eligibility.json`): 108 safe / 39
  excluded / 4 unresolved (`grade`, `sub_grade`, `int_rate`, and
  `member_id`, which is present but carries no information in the real
  file). `chargeoff_within_12_mths` was specifically verified safe
  despite its outcome-sounding name -- it describes OTHER accounts'
  bureau history at the application-time credit pull, not this loan's
  own outcome, demonstrating name-pattern-matching alone is not a
  sufficient leakage test.
- **Bug found and self-corrected before presenting results**:
  informational/clarifying notes on several "yes"-classified
  bureau-pull entries had been placed in `leakage_concern` rather than
  `proposed_use`, which made `filter_application_time_safe()` undercount
  safe features (37 instead of 108) since any non-empty
  `leakage_concern` disqualifies an entry regardless of content. Fixed
  by moving all such notes into `proposed_use`; the Milestone 1
  leakage-keyword safety net in `FeatureEligibilityEntry.__post_init__`
  was unaffected and did not need to change.
- **Chronological split boundaries chosen from the real censoring-by-
  origination-year table, not an arbitrary guess**
  (`configs/credit_risk_split_design.json`,
  `src/credit_risk/chronological_split.py`): origination years
  2007-2017 are 0% censored for `default_12m_last_pymnt_proxy` under the
  inferred `data_as_of_date`; only 2018 is censored (68.9%) and is
  excluded from all three splits WHOLESALE, not partially, to avoid a
  survivorship distortion a loan-level filter would introduce within
  that vintage -- directly applying the project rule "do not touch
  later periods merely to optimize performance." Resulting buckets:
  train 2007-06/2014-12 (466,345 rows, 5.178% prevalence), validation
  2015-01/2016-12 (855,502 rows, 6.508%), benchmark 2017 (443,579 rows,
  6.281%), excluded_too_recent 2018 (495,242 total / 153,917 labeled,
  5.751% prevalence among labeled). `build_chronological_split` raises
  if any censored row is ever found inside a labeled train/validation/
  benchmark bucket -- verified empirically (0 in each), not just
  asserted by construction.
- **Empirical edge case found and treated as a positive validation, not
  a bug**: 10 loans in the 2013 vintage are still
  Current/Late/In-Grace-Period in the snapshot despite being past their
  own scheduled maturity date (likely real loan modifications). Per the
  terminal-target design these stay censored, never coded either way --
  exactly the behavior
  `test_terminal_target_past_maturity_but_still_open_is_censored_not_fabricated`
  asserts.
- Full pytest suite after this milestone: 249 passed, 1 skipped (207
  prior tests unchanged + 42 new tests across
  `tests/credit_risk/test_raw_source.py`, `test_status_mapping.py`,
  `test_target_construction.py`, `test_chronological_split.py`,
  `test_raw_loader.py`, `test_real_feature_eligibility.py`; the 1 skip
  is the pre-existing, unrelated Module B missing-raw-file guard,
  unchanged by this milestone). The 3 real-data-gated tests in
  `test_raw_loader.py` run (not skipped) on this machine since the raw
  file is present and checksum-verified.

## 2026-10-12 — Module C Milestone 2 target-governance correction (pre-commit review)

**Supersedes the target-governance framing in the 2026-10-12 "Module C
Milestone 2" entry above; that entry is left unedited per the
append-only convention, but `default_terminal_status`/
`default_12m_last_pymnt_proxy`'s primary/secondary roles described there
should no longer be cited -- see this entry instead.** Issued before any
commit of Milestone 2, in response to an explicit review request, with
nine numbered requirements; none of Milestone 2's empirical findings
(real schema, status mapping, censoring tables) changed, only the
target-governance designation, `member_id`'s classification, and the
chronological split built from it.

- **Primary/secondary target designation corrected.** The original
  draft implicitly treated `default_12m_last_pymnt_proxy` as the
  presumptive primary PD target. That is wrong and is corrected here:
  the renamed `terminal_adverse_outcome`
  (`compute_terminal_adverse_outcome_target`, was
  `default_terminal_status`/`compute_terminal_status_target`) is now the
  explicit **PRIMARY** modeling target for Milestone 3 -- built only
  from explicitly mapped resolved terminal `loan_status` categories
  (`charged_off`, `fully_paid`), requiring no event date. It estimates
  eventual adverse credit outcome among originated loans over a
  variable, loan-specific (36- or 60-month) horizon and **does not
  represent a 12-month probability of default**; this must never be
  stated otherwise. `default_12m_last_pymnt_proxy` is retained only as a
  **SECONDARY**, sensitivity-analysis-only target, with its documented
  one-directional bias preserved unchanged and an explicit restatement
  that `last_pymnt_d` is NOT a default-event timestamp (it is the date
  of the loan's most recent received payment, repurposed only as an
  explicitly-labeled proxy).
- **New machine-readable governance source of truth**:
  `configs/credit_risk_target_governance.json`
  (`src/credit_risk/target_governance.py`, new module). Mirrors the
  project's existing fail-loud config-validation pattern
  (`RawSourceMetadata`, `FeatureEligibilityEntry`, `SplitDesign`):
  `TargetGovernance.__post_init__` raises `TargetGovernanceError` if the
  primary and secondary target columns are ever the same, if the
  primary target is ever marked a fixed-horizon PD, if the secondary
  target's proxy event date is ever marked a genuine timestamp, or if
  either target's interpretation string is left blank -- a future config
  edit that contradicts this milestone's governance decision fails
  loudly rather than being silently accepted.
- **`member_id` reclassified from `"unknown"` (unresolved) to `"no"`
  (excluded identifier)**, in `configs/credit_risk_feature_eligibility.json`.
  It is a pure identifier with no defensible predictive business role
  and is always 100% NULL in this public release; "unknown" is reserved
  for genuinely ambiguous-availability fields, which `member_id` is not.
  Leakage-audit counts move from 108 safe / 39 excluded / 4 unresolved
  to **108 safe / 40 excluded / 3 unresolved** (the remaining 3
  unresolved are `grade`/`sub_grade`/`int_rate`, explicitly re-confirmed,
  not revisited, by this correction).
- **Chronological split redesigned for the primary target, not reused
  from the secondary split.** Because `terminal_adverse_outcome`
  resolves on each loan's own 36-/60-month maturity rather than a fixed
  12-month window, the original (now secondary) split's boundaries are
  invalid for it: under `terminal_adverse_outcome`, its benchmark year
  (2017) is 100% censored and its validation years (2015-2016) are
  33-75% censored. `SplitDesign.__post_init__`'s existing
  `benchmark_end + outcome_window_months <= data_as_of_date` validation
  rule required no code change -- setting `outcome_window_months=60`
  (the real empirical maximum of LendingClub's `term` column; confirmed
  distinct values `[36, 60]` only) was sufficient to generalize it
  correctly. The single `configs/credit_risk_split_design.json` was
  replaced by two explicit files:
  `configs/credit_risk_split_design_primary_terminal.json` (new default,
  loaded by `chronological_split.py`'s `DEFAULT_SPLIT_DESIGN_FILE`: train
  2007-06/2011-12 -- 42,535 rows, 15.119% prevalence; validation
  2012 -- 53,367 rows, 16.197%; benchmark 2013 -- 134,814 rows, 10
  censored, 15.596% of 134,804 labeled; excluded_too_recent 2014+ --
  2,029,952 total / 572,677 labeled) and
  `configs/credit_risk_split_design_secondary_proxy.json` (unchanged
  boundaries/counts from the original entry, now loaded explicitly via
  `SECONDARY_PROXY_SPLIT_DESIGN_FILE`). As with the original split, the
  entire under-resolved tail (2014 onward) is excluded wholesale, not
  partially, per the project rule against touching later periods to
  optimize performance. The primary split's labeled sample
  (train+validation+benchmark = 230,716 rows) is substantially smaller
  than the secondary split's (1,765,426 rows) -- a direct, foreseeable
  consequence of correctly designating the longer-horizon target as
  primary, stated here so Milestone 3 is not surprised by it.
- **Documentation, configs, and tests updated accordingly**: `README.md`,
  `docs/credit_risk_split_design.md` (restructured into explicit Primary/
  Secondary sections), `docs/moduleC_milestone2_real_data_validation.md`
  (Sections D, E, F, G, I, J, K revised), `docs/credit_risk_status_mapping.md`,
  `configs/credit_risk_feature_eligibility.json`'s two `proposed_use`
  strings naming the old target, `src/credit_risk/status_mapping.py`'s
  docstring, `tests/credit_risk/test_target_construction.py` (renamed
  imports/calls), `tests/credit_risk/test_real_feature_eligibility.py`
  (`member_id`/excluded-count assertions), `tests/credit_risk/
  test_chronological_split.py` (added a secondary-split-loads-correctly
  test), and a new `tests/credit_risk/test_target_governance.py`. No
  model was trained; no commit or push was made as part of this
  correction.

## 2026-10-12 — Module C Milestone 3: Model A / Model B logistic PD baselines on the frozen primary split

- **No split boundary changed.** Milestone 3 reuses the primary
  chronological split exactly as frozen in Milestone 2 (train
  2007-06/2011-12 — 42,535 rows, 15.119%; validation 2012 — 53,367
  rows, 16.197%; benchmark 2013 — 134,814 total/134,804 labeled,
  15.596%; excluded_too_recent 2014+ — 2,029,952 total/572,677
  labeled). Re-running the split against the real file reproduced
  these exact counts, confirming no drift before any model was fit.
- **Model A / Model B feature lists were predeclared in
  `configs/credit_risk_model_feature_sets.json` and written up in
  `docs/credit_risk_model_feature_sets.md` before either model was
  fit**, per this project's "predeclare, never adjust after seeing
  evaluation-split performance" discipline. Model A: 19 raw numeric +
  4 derived numeric (`term_months`, `emp_length_years`,
  `fico_avg_at_application`, `credit_history_months_at_application`) +
  7 categorical fields, every raw field independently cross-checked
  against `credit_risk_feature_eligibility.json`'s
  `is_application_time_safe` property at load time
  (`feature_selection.py::load_feature_sets`). Model B is defined
  **structurally**, not as an independently edited list: Model A plus
  exactly `{grade, sub_grade, int_rate}` — `load_feature_sets` raises
  `FeatureSelectionError` if Model B's additional-fields config ever
  drifts from that exact set or overlaps Model A's own list.
- **Empirically-verified exclusions from the already-eligibility-
  cleared pool**, each with one stated rationale (not assumed from
  column names): `funded_amnt`/`funded_amnt_inv` dropped for
  near-duplication with `loan_amnt` (correlation 0.9998/0.9990,
  verified on the full real file, not assumed);
  `fico_range_low`/`fico_range_high` superseded by the single derived
  `fico_avg_at_application` (correlation 0.99999998, a constant
  4-5-point gap on every row); 48 vintage-introduced fields (e.g.
  `open_acc_6m`, `bc_util`) excluded because they are verified
  100%-missing in the train bucket specifically (direct query against
  `loans_raw` restricted to 2007-06/2011-12), not merely assumed
  missing from their names — an all-NaN-in-train column contributes no
  trainable signal under the train-only-fit imputer and would leak
  information about what "train" means if imputed from later
  vintages; joint-application fields (`annual_inc_joint`,
  `sec_app_*`, etc., ~95% NULL) excluded as a candidate for a future
  dedicated joint-application sub-model rather than naively imputed
  into the two baseline models; `zip_code` excluded as high-cardinality
  geography (`addr_state` retained as the baseline's geographic
  control) — noted as a separate, stricter decision from
  `subgroup_diagnostics.py`'s exclusion of `addr_state`/`zip_code` from
  the Section J governance diagnostic specifically.
- **Uniform missingness policy, no per-field special-casing**: even
  though `mths_since_last_delinq` (63% missing in train) and
  `mths_since_last_record` (91% missing in train) carry *meaningful*
  missingness (NULL means "never observed," not random),
  `preprocessing.py` applies `SimpleImputer(strategy="median",
  add_indicator=True)` uniformly to every numeric feature rather than
  hand-coding semantics for just these two fields — any column with
  train-set missingness automatically gets its own
  `missingindicator_<field>` column.
- **Logistic regression only, modest untuned regularization, fit once
  each, train-only**: `LogisticRegression(penalty="l2", C=1.0,
  solver="lbfgs", max_iter=1000, class_weight=None)` for both models —
  `C=1.0` used *because* it is scikit-learn's neutral default, not
  because a grid search against validation/benchmark favored it (that
  would be threshold/hyperparameter snooping). `class_weight=None`
  (not `"balanced"`) is deliberate so predicted probabilities stay
  comparable to train's real ~15.1% prevalence, which Section F's
  calibration comparison depends on.
- **Decision threshold named and fixed before touching
  evaluation splits**: `train_prevalence_matched` = train's own
  observed label mean (0.151193), chosen specifically to avoid
  threshold-snooping (picking whatever threshold happens to flatter a
  split's numbers after seeing them).
- **Calibration method choice governed by predeclared, not
  post-hoc, rules** (`calibration.py`): isotonic is only eligible
  when train has at least `MIN_TRAIN_ROWS_FOR_ISOTONIC` (1,000) rows,
  and is only preferred over sigmoid if it beats sigmoid's Brier score
  on validation by at least `ISOTONIC_PREFERENCE_BRIER_IMPROVEMENT`
  (0.0005) — both margins fixed before the comparison was run. Fit on
  train only via fresh clones of the base model (never reusing the
  already-trained `base` estimator), assessed on validation only.
  Result for both models: the *uncalibrated* base model already had
  the lowest Brier score of all three candidates, so neither
  calibrated variant was adopted — reported as-is rather than forcing
  a calibration step the data didn't support.
- **Ablation framing is structurally non-droppable**: every
  `AblationResult` (`ablation.py`) carries a `framing_note` restating
  that Model B's lift over Model A quantifies "signal embedded in the
  lender's prior underwriting decision," never a deployment
  recommendation — `framing_note` is included in `as_dict()` so it
  cannot be silently dropped from any downstream serialization.
- **`excluded_too_recent`'s labeled subset is used only for Section
  I's descriptive vintage-stability analysis**, never for training,
  threshold selection, or benchmark metrics — `vintage_analysis.py`
  separately tags each origination year's `evaluation_eligible`
  status (false only when a year is *purely*
  `excluded_too_recent`), so this distinction is explicit in the
  output, not just in prose.
- **Subgroup governance diagnostics are restricted, at runtime, to
  four predeclared non-demographic fields**: `NON_PROTECTED_
  SUBGROUP_FIELDS = ("home_ownership", "verification_status",
  "application_type", "purpose")` in `subgroup_diagnostics.py`.
  `compute_subgroup_diagnostics` raises `SubgroupDiagnosticsError` if
  asked to group by any other column — in particular `addr_state`/
  `zip_code`, which are excluded here as documented geographic
  protected-class proxies, a stricter standard applied only to this
  one diagnostic (both remain legitimate Model A features elsewhere).
- **10 new `src/credit_risk/` modules** (`feature_selection.py`,
  `feature_loader.py`, `preprocessing.py`, `modeling.py`,
  `evaluation.py`, `calibration.py`, `interpretation.py`,
  `ablation.py`, `vintage_analysis.py`, `subgroup_diagnostics.py`) plus
  `run_milestone3_baseline.py` (orchestration) and **10 new test
  files** covering every correctness property enumerated above
  (feature-set structural enforcement, train-only-fit preprocessing,
  unseen-category handling, metric correctness, calibration
  leakage-safety, ablation sign conventions, vintage eligibility
  tagging, subgroup field allowlist enforcement). Full suite:
  `pytest tests/credit_risk -q` → 137 passed, no regressions.
- **New docs**: `docs/credit_risk_model_feature_sets.md` (Section B
  companion doc, written before any model was fit) and
  `docs/moduleC_milestone3_credit_baseline.md` (the full Section A-M
  write-up with every real computed number). `README.md`'s Module C
  sections and this entry were updated to match.
- **No model was selected for deployment, no production threshold was
  set, and no commit or push was made as part of this milestone** —
  the deliverable stops here for review, per explicit instruction not
  to start a further Module C milestone without the user's review.

## 2026-10-13 — Module C Milestone 4: expected loss, decisioning & probability-governance diagnostics

- **Model A and Model B were refit, never retuned, by directly calling
  Milestone 3's own fitting function**
  (`run_milestone3_baseline._fit_and_evaluate_one_model`), not copied or
  reimplemented — guaranteeing Milestone 4's analyses run against
  exactly the same specification, split, and preprocessing pipeline
  Milestone 3 committed, with no silent drift.
- **Section A: kept raw, uncalibrated probabilities, per the milestone's
  own "don't force calibration" constraint.** Re-checked Milestone 3's
  existing finding that neither sigmoid nor isotonic improves on the
  uncalibrated base model's Brier score on validation (true for both
  models), then explicitly decided every Milestone 4 calculation (risk
  bands, expected loss, decisioning, cost model, governance
  diagnostics) consumes the **raw** predicted probability, not a
  recalibrated variant — a deliberate reading of "keep uncalibrated if
  best," not an oversight.
- **New Cox calibration-slope/intercept diagnostic added additively to
  `calibration.py`**, Milestone 3's own functions untouched. Found a
  real, honest, somewhat concerning result: both models' calibration
  slope is reasonable on validation (Model A 0.95, Model B 0.85) but
  degrades sharply on benchmark (Model A 0.42, Model B 0.50) with a
  large negative intercept — reported as-is, not smoothed over, even
  though Brier score and PSI (below) both look stable on the same
  split. This is explicitly flagged in the milestone doc as two
  genuinely different properties (distribution shape vs. calibration
  accuracy), not a contradiction to be resolved away.
- **Risk-band boundaries (`risk_bands.py`) are fit on train-only
  quantiles and never refit** when bucketing validation/benchmark —
  mirrors the preprocessing pipeline's own train-only-fit discipline
  from Milestone 3. Zero-row bands are reported with explicit `None`
  fields rather than fabricated zeros (`summarize_risk_bands`).
- **Expected-loss framework (`expected_loss.py`) uses a single
  configurable LGD scalar (default 0.8,
  `configs/credit_risk_expected_loss_assumptions.json`) and
  `loan_amnt` as the EAD proxy** (`EAD_FIELD` constant) — chosen for
  consistency with Milestone 3's own feature-selection decision to drop
  `funded_amnt`/`funded_amnt_inv` as near-duplicates of `loan_amnt`.
  Neither LGD nor the EAD definition is derived from the real
  `recoveries`/`total_rec_prncp` fields; full rationale in the new
  companion doc, `docs/credit_risk_expected_loss_framework.md`.
- **Decision-cost module (`decision_cost_model.py`) deliberately departs
  from Module B's `cost_model.py` precedent on its largest cost term**:
  rather than a flat per-event scalar, realized adverse-outcome loss is
  computed per loan as `lgd * ead` (the same lgd/ead Section C uses,
  passed in by the caller, not duplicated as a separate config value) —
  so a bigger loan that goes bad costs more than a smaller one. Flat
  scalars are kept only for `cost_per_manual_review` and
  `opportunity_cost_per_good_loan_declined` ($50/$150,
  `configs/credit_risk_decisioning_cost_assumptions.json`), which really
  are flat per-event costs.
- **Three-way decisioning (`decisioning.py`) reuses Section B's risk-
  band boundaries directly as its review/decline thresholds**
  (`low_medium_boundary`→review, `medium_high_boundary`→decline) rather
  than independently fitting a second threshold pair — avoids duplicate
  threshold-fitting logic and ties the "approve-like/review/decline-
  like" policy to the "low/medium/high" bands by construction. Labels
  are deliberately neutral (`approve_like`/`manual_review`/
  `decline_like`), never "approve"/"decline," to avoid overstating
  underwriting authority for any real applicant.
- **Section G's sensitivity analysis never labels any row "optimal"**
  (`sensitivity_analysis` only reports how adverse-outcome capture,
  good-loan rejection rate, review volume, and estimated illustrative
  cost move as LGD/threshold/cost assumptions move) — verified by test
  (`"optimal" not in [c.lower() for c in out.columns]`).
- **Selection-bias constraint embedded machine-readably, not only in
  prose**: `SELECTION_BIAS_DISCLAIMER` is written as a top-level
  `selection_bias_disclaimer` key in
  `data/interim/credit_risk_milestone4_metrics.json`, alongside a
  `target_framing_note` restating that `terminal_adverse_outcome` is
  not a 12-month PD — satisfying the requirement that both disclaimers
  be present in the generated, machine-readable artifact itself.
- **PSI drift heuristic (`score_stability.py`) fits bin edges on the
  reference distribution only** (train), mirroring the train-only-fit
  discipline used throughout this project, with an explicit
  `PSI_HEURISTIC_CAVEAT` that its 0.10/0.25 bands are an industry rule
  of thumb, not a calibrated statistical test on this dataset. Result:
  both models land in the heuristic "no significant change" band for
  train vs. validation and train vs. benchmark, despite the calibration
  degradation found in Section A/I — an intentional, documented tension
  rather than an inconsistency to paper over.
- **Section J governance discussion concludes Model B's higher raw
  discrimination is not evidence of better governance**: Model A and
  Model B's benchmark expected-loss rankings agree almost completely
  (Spearman ρ=0.9667), consistent with Milestone 3's existing finding
  that Model B's lender-grade lift is real but modest, and consistent
  with the circularity/interpretability/historical-dependency concerns
  about `grade`/`sub_grade`/`int_rate` already raised in Milestone 3.
- **Pandas `None`→`NaN` coercion pitfall hit again** (same family as the
  numpy `bool_` pitfall from Milestone 3): a test asserting a zero-row
  risk band's fields `is None` failed because a DataFrame column mixing
  floats and `None` silently coerces `None` to `NaN` — fixed by testing
  with `pd.isna()` instead, the module's actual return value (a literal
  `None`) was already correct.
- **6 new `src/credit_risk/` modules** (`expected_loss.py`,
  `risk_bands.py`, `decisioning.py`, `decision_cost_model.py`,
  `score_stability.py`, `run_milestone4_decisioning.py` orchestration)
  plus one existing module additively extended (`calibration.py`,
  Milestone 3's functions untouched) and **6 new/extended test files**.
  Full repository suite: **371 passed, 1 skipped**
  (`pytest -q`), zero regressions against the pre-Milestone-4 baseline
  of 318 passed.
- **New docs**: `docs/credit_risk_expected_loss_framework.md` (Section
  C/D LGD/EAD companion doc) and
  `docs/moduleC_milestone4_credit_decisioning.md` (the full Section A-M
  write-up with every real computed number). `README.md`'s Module C
  sections and this entry were updated to match.
- **No model was selected for deployment, no production decisioning
  policy or threshold was set, and no commit or push was made as part
  of this milestone** — the deliverable stops here for review, per
  explicit instruction not to start a further Module C milestone
  without the user's review.

## 2026-10-13 — Portfolio Production Step 1: repository audit corrections

Per append-only convention, correcting rather than rewriting the two
entries above rather than editing them in place.

- **Primary split labeled-row total was miscomputed in prose.** The
  2026-10-12 "governance correction" entry above lists the primary
  split's per-bucket counts correctly (train 42,535; validation 53,367;
  benchmark 134,814 total rows / 10 censored / **134,804 labeled**) but
  then states the labeled-row sum as "230,716 rows" two lines later.
  230,716 = 42,535 + 53,367 + 134,814 (sum of *total* rows, including
  the 10 censored benchmark rows); the correct sum of *labeled* rows is
  **230,706** (42,535 + 53,367 + 134,804), confirmed against
  `data/interim/credit_risk_milestone3_metrics.json`'s
  `model_a_application_time_safe.benchmark_metrics.n = 134804`, which is
  the actual row count used for model evaluation. The historical entry
  above is left unedited per this file's append-only rule; `README.md`,
  `docs/credit_risk_split_design.md`, and
  `docs/moduleC_milestone2_real_data_validation.md` (all of which had
  copied the 230,716 figure) have been corrected to 230,706.
- **Sensitivity-analysis variation count corrected in
  `docs/milestone4_calibration_decisioning.md`.** Two passages said
  `build_cost_variations` produces "10 labeled variations" / "11
  sensitivity variations (base + 10 labeled)"; the function actually
  returns **11** labeled variations (confirmed against
  `tests/models/test_train_calibrated.py::test_build_cost_variations_preserves_unvaried_fields`'s
  explicit 11-label set), and `sensitivity_analysis` is called with
  exactly those 11 variations and no separate base row appended. Both
  passages corrected to "11 labeled variations" / "11 labeled
  sensitivity variations" (dropping the inaccurate "base +" framing).
