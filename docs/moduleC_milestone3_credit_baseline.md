# Module C, Milestone 3 — Credit-Risk Baseline Logistic Regression (Model A vs. Model B)

## 0. Scope boundary

This milestone fits and evaluates the **first real credit-risk model**
in this project: two predeclared logistic-regression specifications
(Model A, Model B), on the frozen PRIMARY chronological split, against
the real LendingClub 2007-2018Q4 file. It is a baseline, not a
deployment candidate — see the repeated framing notes throughout this
doc and in code (`src/credit_risk/ablation.py::FRAMING_NOTE`). No
boosted trees/random forest are used (logistic regression only, per the
milestone's own design constraint — nonlinear comparison is reserved for
a later milestone, mirroring Module B's structure). No approve/decline
recommendation is made for any applicant. Every number below comes from
one real, reproducible run of
`src/credit_risk/run_milestone3_baseline.py` against the real 2,260,668-row
file; none is estimated. Full stdout + the JSON reproducibility artifact
(`data/interim/credit_risk_milestone3_metrics.json`) back every figure
quoted here.

Reproduce with:
```bash
.venv/bin/python -m src.credit_risk.run_milestone3_baseline
```

## A. Primary split — frozen, unaltered

The split is reused exactly as designed in Milestone 2
(`configs/credit_risk_split_design_primary_terminal.json`,
`src/credit_risk/chronological_split.py`) — no boundary was touched for
this milestone:

| split_bucket | total_rows | censored_excluded | labeled_rows | default_count | prevalence | origination range |
|---|---|---|---|---|---|---|
| train | 42,535 | 0 | 42,535 | 6,431 | 15.119% | 2007-06 to 2011-12 |
| validation | 53,367 | 0 | 53,367 | 8,644 | 16.197% | 2012-01 to 2012-12 |
| benchmark | 134,814 | 10 | 134,804 | 21,024 | 15.596% | 2013-01 to 2013-12 |
| excluded_too_recent | 2,029,952 | 1,457,275 | 572,677 | 88,228 | 15.406% | 2014-01 to 2018-12 |

These figures match Milestone 2's already-verified numbers exactly
(`docs/moduleC_milestone2_real_data_validation.md`). Only train is ever
fit on; validation and benchmark are evaluation-only; excluded_too_recent
never enters any evaluation-eligible metric (its labeled subset is used
*only* in Section I's explicitly descriptive-only vintage analysis).

## B. Model A / Model B feature sets

Full rationale: `docs/credit_risk_model_feature_sets.md` and
`configs/credit_risk_model_feature_sets.json`. In short:

- **Model A** (`model_a_application_time_safe`): 19 raw numeric + 4
  derived numeric (`term_months`, `emp_length_years`,
  `fico_avg_at_application`, `credit_history_months_at_application`) +
  7 categorical fields, every one independently cross-checked against
  `configs/credit_risk_feature_eligibility.json`'s
  `is_application_time_safe` property.
- **Model B** (`model_b_with_lender_grade_signal`): Model A plus exactly
  `grade`, `sub_grade`, `int_rate` — LendingClub's own prior risk
  assessment/pricing, structurally enforced to be nothing more or less
  than that by `feature_selection.py`.

## C. Preprocessing pipeline (leakage-safe, shared by both models)

`src/credit_risk/preprocessing.py`: numeric columns get
`SimpleImputer(strategy="median", add_indicator=True)` →
`StandardScaler`; categorical columns get
`SimpleImputer(strategy="constant", fill_value="__missing__")` →
`OneHotEncoder(handle_unknown="ignore", drop="first")`. The whole
`ColumnTransformer` is `.fit()` on train only
(`fit_preprocessing_pipeline`); validation/benchmark are only ever
`.transform()`-ed. `handle_unknown="ignore"` is exercised for real during
this run — the actual sklearn warning fired
(`Found unknown categories in columns [...] during transform`) when
validation/benchmark/excluded_too_recent carried `addr_state`/
`disbursement_method`/other category values never seen in the 2007-2011
train window; these are encoded as all-zero dummies rather than raising.
Model A resolves to 48 transformed columns (30 input + 2 missing
indicators for `mths_since_last_delinq`/`mths_since_last_record` +
one-hot columns); Model B adds the `grade`/`sub_grade`/`int_rate`
columns on top.

## D. Model fitting

`src/credit_risk/modeling.py::fit_logistic_regression`: `LogisticRegression(penalty="l2", C=1.0, solver="lbfgs", max_iter=1000, class_weight=None)` for **both** models — `C=1.0` is scikit-learn's own untuned default, used *because* it's a neutral off-the-shelf value, not because a grid search against validation/benchmark favored it (that would be threshold/hyperparameter snooping). `class_weight=None` (not `"balanced"`) is deliberate: Section F's calibration comparison needs predicted probabilities comparable to train's real ~15.1% prevalence, and class reweighting would shift them away from it. Each model is fit **exactly once**, on train only.

## E. Evaluation (validation + benchmark)

**Threshold**: `train_prevalence_matched` = train's own observed label
mean = **0.151193** — decided from train alone, before any
validation/benchmark metric was computed, specifically to avoid
threshold-snooping (choosing whatever threshold happens to make a split
look best).

| | Model A — validation | Model A — benchmark | Model B — validation | Model B — benchmark |
|---|---|---|---|---|
| n | 53,367 | 134,804 | 53,367 | 134,804 |
| prevalence | 16.197% | 15.596% | 16.197% | 15.596% |
| TN / FP / FN / TP | 29,174 / 15,549 / 3,486 / 5,158 | 65,839 / 47,941 / 7,338 / 13,686 | 26,872 / 17,851 / 2,982 / 5,662 | 60,562 / 53,218 / 5,954 / 15,070 |
| precision | 0.2491 | 0.2221 | 0.2408 | 0.2207 |
| recall | 0.5967 | 0.6510 | 0.6550 | 0.7168 |
| F1 | 0.3515 | 0.3312 | 0.3521 | 0.3375 |
| ROC-AUC | 0.6731 | 0.6566 | 0.6758 | 0.6712 |
| PR-AUC | 0.2814 | 0.2537 | 0.2816 | 0.2632 |
| Brier | 0.12864 | 0.12667 | 0.12847 | 0.1265 |

**Reading these honestly**: ROC-AUC in the 0.66-0.68 range and PR-AUC
around 0.25-0.28 (against a ~15-16% base rate) indicate modest, real,
non-trivial discrimination — far from a strong classifier, consistent
with a baseline logistic regression on a long-horizon, noisy credit
outcome. At the `train_prevalence_matched` threshold, precision is low
(~0.22-0.25) because the threshold is deliberately set near the *base
rate*, not tuned for any particular precision/recall tradeoff — this is
a diagnostic choice, not an operational recommendation.

## F. Calibration (sigmoid vs. isotonic, train-only CV, validation-only assessment)

`src/credit_risk/calibration.py`: both candidate recalibration maps are
fit via `CalibratedClassifierCV(cv=5)` **entirely within train** (fresh
`LogisticRegression` clones per fold, never the already-fit model
reused directly); calibration **quality** is assessed only on
validation (data neither the base model nor the calibration maps ever
saw).

| | Model A | Model B |
|---|---|---|
| n_train | 42,535 | 42,535 |
| Uncalibrated Brier (validation) | 0.128637 | 0.128466 |
| Sigmoid Brier (validation) | 0.129002 | 0.128497 |
| Isotonic Brier (validation) | 0.128686 | 0.128351 |
| **Chosen method** | **sigmoid** | **sigmoid** |

For both models, isotonic's Brier improvement over sigmoid
(0.000316 for Model A, 0.000146 for Model B) falls short of the
predeclared `ISOTONIC_PREFERENCE_BRIER_IMPROVEMENT = 0.0005` floor, so
sigmoid is chosen by the predeclared rule
(`compare_calibration`'s `chosen_method_reason`, reproduced verbatim in
the JSON artifact). Note honestly: for both models, the **un-calibrated**
base model already has the lowest Brier of the three candidates on
validation — this baseline logistic regression on this split is already
reasonably well-calibrated out of the box, and neither recalibration
map materially improves it. `compare_calibration` only ever chooses
between sigmoid and isotonic (uncalibrated is reported for reference,
never a selectable option), consistent with the milestone's requirement
to produce *a* calibration recommendation, not necessarily the single
best-scoring one of all three.

Reliability curves (10-bin, train-fit model, validation-evaluated) are
stored in full in the JSON artifact
(`models.<name>.reliability_curve_uncalibrated`); both models track
observed frequency reasonably closely through the bulk of the
probability range and diverge somewhat at the sparsely-populated top
bins (e.g. Model A's bin 6, n=18, predicted 0.630 vs. observed 0.778) —
expected given how few validation rows fall that high.

## G. Coefficient interpretation (odds ratios, never causal language)

All reference levels (the alphabetically-first category `OneHotEncoder(drop="first")` drops per field) for Model A/B:
`home_ownership`→MORTGAGE, `verification_status`→Not Verified,
`purpose`→car, `addr_state`→AK, `initial_list_status`→f,
`application_type`→Individual, `disbursement_method`→Cash (Model B
additionally: `grade`→A, `sub_grade`→A1). Every coefficient below is
read as "associated with," never "causes."

**Model A intercept**: -2.166 (log-odds for an all-reference-level,
zero-standardized-numeric-feature borrower).

**Model A — top 10 by |coefficient|**:

| feature | coefficient | odds ratio | direction |
|---|---|---|---|
| purpose_small_business | 1.0431 | 2.838 | HIGHER default odds |
| addr_state_WY | -0.8656 | 0.421 | LOWER default odds |
| purpose_educational | 0.5946 | 1.812 | HIGHER default odds |
| addr_state_DC | -0.5865 | 0.556 | LOWER default odds |
| home_ownership_OTHER | 0.5462 | 1.727 | HIGHER default odds |
| purpose_renewable_energy | 0.5405 | 1.717 | HIGHER default odds |
| addr_state_NV | 0.5398 | 1.716 | HIGHER default odds |
| addr_state_DE | -0.4952 | 0.609 | LOWER default odds |
| purpose_moving | 0.4569 | 1.579 | HIGHER default odds |
| purpose_other | 0.4358 | 1.546 | HIGHER default odds |

`purpose_small_business` carrying the single largest coefficient is a
defensible, intuitive result (small-business loans are widely
understood to carry concentrated repayment risk relative to e.g. debt
consolidation); several `addr_state` dummies also rank highly, which is
expected given 51 low-base-rate geographic dummies relative to train's
42,535 rows — individual state coefficients should be read with that
small-per-state-sample caveat in mind, not as strong standalone
findings.

**Model B's `grade`/`sub_grade`/`int_rate` coefficients** (41 dummy +
numeric rows total; largest few shown, full table in the JSON
artifact's `coefficients_top20`):

| feature | coefficient | odds ratio |
|---|---|---|
| sub_grade_F5 | 0.4919 | 1.635 |
| sub_grade_A5 | 0.3850 | 1.470 |
| grade_B | 0.3405 | 1.406 |
| grade_C | 0.3250 | 1.384 |
| grade_D | 0.3243 | 1.383 |
| int_rate | 0.3062 | 1.358 |
| grade_F | 0.2756 | 1.317 |

**Model B intercept**: -2.530.

Several `grade`/`sub_grade` coefficients do not monotonically increase
with worsening letter grade once the other 30 Model-A features are
already in the model (e.g. `grade_B` > `grade_C` ≈ `grade_D` >
`grade_F`, and several `sub_grade` dummies are slightly negative) — this
is the expected signature of circularity/collinearity with
`int_rate` and the many Model-A features grade is itself derived from,
not a sign of a coding error. This is exactly the phenomenon Model B
exists to surface, not resolve.

## H. Lender-signal ablation (Model A → Model B)

`src/credit_risk/ablation.py`. Every result below repeats
`AblationResult.framing_note` verbatim: *"This lift quantifies signal
already embedded in LendingClub's own prior underwriting decision
(grade/sub_grade/int_rate), not a recommendation to use these fields in
a deployed PD model."*

| | validation | benchmark |
|---|---|---|
| ROC-AUC lift (B − A) | +0.0027 | +0.0146 |
| PR-AUC lift (B − A) | +0.0001 | +0.0095 |
| Brier improvement (A − B) | +0.00017 | +0.00017 |

On validation, Model B's lift over Model A is small (ROC-AUC +0.003,
PR-AUC near-zero). On benchmark, the lift is larger (ROC-AUC +0.015,
PR-AUC +0.0095) but still modest in absolute terms. Read honestly: the
lender's own grade/sub_grade/int_rate assignment carries *some*
additional predictive signal beyond the 30 application-time-safe Model A
features, but it is not a dramatic jump — Model A alone already
captures most of the discrimination Model B achieves. This is evidence
about signal overlap, **not** a recommendation to deploy Model B.

## I. Vintage / stability analysis (descriptive only)

`src/credit_risk/vintage_analysis.py`, Model A applied (never refit) to
every labeled row across **all** origination years 2007-2016, including
`excluded_too_recent`'s already-resolved labeled subset — per
`docs/credit_risk_split_design.md`'s explicit framing of that subset as
"a useful empirical cross-check," never an evaluation-eligible split.

| year | split_bucket(s) | eval-eligible | n | prevalence | ROC-AUC | Brier |
|---|---|---|---|---|---|---|
| 2007 | train | True | 603 | 26.202% | 0.7336 | 0.17934 |
| 2008 | train | True | 2,393 | 20.727% | 0.6793 | 0.15408 |
| 2009 | train | True | 5,281 | 13.691% | 0.6655 | 0.11292 |
| 2010 | train | True | 12,537 | 14.015% | 0.7188 | 0.11106 |
| 2011 | train | True | 21,721 | 15.179% | 0.7093 | 0.11936 |
| 2012 | validation | True | 53,367 | 16.197% | 0.6731 | 0.12864 |
| 2013 | benchmark | True | 134,804 | 15.596% | 0.6566 | 0.12667 |
| 2014 | excluded_too_recent | **False** | 180,416 | 15.087% | 0.6304 | 0.12522 |
| 2015 | excluded_too_recent | **False** | 283,026 | 14.886% | 0.6129 | 0.12550 |
| 2016 | excluded_too_recent | **False** | 109,235 | 17.280% | 0.6186 | 0.14276 |

Read honestly: there is a visible, gradual **downward drift** in ROC-AUC
from the earliest train years (0.68-0.73) through validation/benchmark
(0.65-0.67) into the descriptive-only 2014-2016 years (0.61-0.63). Early
years (2007-2008) have very small n (603, 2,393) so their high ROC-AUC
should not be over-read. The 2014-2016 numbers are reported **only** as
a descriptive cross-check per the split design's own stated purpose —
they are never used to retrain, re-tune a threshold, or report a
headline benchmark metric, and the apparent drift there is not
investigated further in this milestone (a candidate follow-up question,
not resolved here).

## J. Subgroup / governance diagnostics (non-demographic)

`src/credit_risk/subgroup_diagnostics.py`, Model A on validation. **No
race, sex, age, or other protected-class field exists anywhere in the
real file** — this is explicitly *not* a demographic-lending-
optimization exercise. The four grouping fields are predeclared,
non-protected, loan/application-structural facts
(`NON_PROTECTED_SUBGROUP_FIELDS`); `addr_state`/`zip_code` are
deliberately excluded as documented geographic protected-class proxies,
even though they are otherwise-eligible modeling features.

**`home_ownership`**:

| group | n | missingness% | observed default% | mean predicted% | calib. gap (pts) | Brier | FPR | FNR |
|---|---|---|---|---|---|---|---|---|
| MORTGAGE | 24,947 | 6.736 | 15.040 | 14.431 | -0.609 | 0.12016 | 0.3137 | 0.4123 |
| RENT | 24,129 | 7.124 | 17.249 | 15.541 | -1.708 | 0.13648 | 0.3797 | 0.3993 |
| OWN | 4,203 | 7.196 | 16.988 | 15.294 | -1.693 | 0.13346 | 0.3660 | 0.3810 |
| OTHER | 46 | 6.805 | 19.565 | 23.454 | +3.889 | 0.15733 | 0.8378 | 0.0000 |
| NONE | 42 | 5.487 | 16.667 | 14.779 | -1.888 | 0.14679 | 0.3143 | 0.7143 |

**`verification_status`**:

| group | n | missingness% | observed default% | mean predicted% | calib. gap | Brier | FPR | FNR |
|---|---|---|---|---|---|---|---|---|
| Verified | 21,246 | 6.944 | 19.161 | 18.187 | -0.974 | 0.14624 | 0.4968 | 0.2700 |
| Not Verified | 19,869 | 6.927 | 12.864 | 11.905 | -0.959 | 0.10841 | 0.2104 | 0.5943 |
| Source Verified | 12,252 | 6.984 | 16.463 | 14.531 | -1.932 | 0.13091 | 0.3297 | 0.4303 |

**`application_type`**: 100% Individual in validation (53,367 rows,
16.197% default, Brier 0.12864) — `Joint App` does not appear at all in
the validation year, so no cross-group comparison is possible for this
field on this split (reported honestly rather than fabricated).

**`purpose`** (13 distinct values; smallest groups have very low n and
should be read cautiously, e.g. `renewable_energy` n=71): the two
largest groups, `debt_consolidation` (n=30,811, 17.377% observed
default) and `credit_card` (n=10,365, 12.726%), are both close to
calibrated (gaps of -1.4 and -0.9 points respectively); several small
groups show larger gaps (e.g. `house` n=418, gap -4.4 points;
`renewable_energy` n=71, gap -5.8 points) that are plausibly sampling
noise given their size, not necessarily a systematic miscalibration —
full table in the JSON artifact.

**Reading this section honestly**: this is a **descriptive monitoring**
exercise only — it flags that `RENT`/`OWN` home-ownership groups and the
`Verified`/`Source Verified` verification groups show somewhat larger
calibration gaps and higher false-positive rates than `MORTGAGE`/`Not
Verified` respectively, at the `train_prevalence_matched` threshold.
This is **not** a recommendation to adjust underwriting by segment, and
none of these four fields is a protected class.

## K. New reusable modules + tests

Ten new modules under `src/credit_risk/`: `feature_selection.py`,
`feature_loader.py`, `preprocessing.py`, `modeling.py`, `evaluation.py`,
`calibration.py`, `interpretation.py`, `ablation.py`,
`vintage_analysis.py`, `subgroup_diagnostics.py`, plus the orchestration
script `run_milestone3_baseline.py`. Ten matching test files under
`tests/credit_risk/` (`test_feature_selection.py`,
`test_feature_loader.py`, `test_preprocessing.py`, `test_modeling.py`,
`test_evaluation.py`, `test_calibration.py`, `test_interpretation.py`,
`test_ablation.py`, `test_vintage_analysis.py`,
`test_subgroup_diagnostics.py`) cover: Model A excludes
grade/sub_grade/int_rate and Model B includes exactly those three;
preprocessing is fit on train only and never re-fit on transform;
unseen categories at transform time are handled gracefully
(all-zero dummies, no raise); the uniform missing-indicator policy
fires only for columns with missingness in train; confusion-matrix/
precision/recall/F1 arithmetic is correct at an arbitrary non-0.5
threshold; calibration's train-only-CV / validation-only-assessment
separation and its isotonic sample-size/improvement-margin rules;
odds ratios are `exp(coefficient)` and reference levels are reported
correctly; the ablation lift is `model_b − model_a` with the documented
Brier-improvement sign convention and the framing note is never
droppable; vintage analysis correctly tags `evaluation_eligible` per
origination year; and subgroup diagnostics refuses any grouping field
outside the predeclared non-protected list. Full suite: **137 passed**
(`pytest tests/credit_risk -q`), no regressions against the
pre-existing 67-test suite.

## L. Documentation

This document (Sections A-J, M), `docs/credit_risk_model_feature_sets.md`
(Section B detail), `README.md`'s Module C status section, and
`docs/decision_log.md`'s Milestone 3 entry.

## M. Consolidated summary

A predeclared, application-time-safe baseline (Model A, 30 fields) and
its lender-grade-augmented counterpart (Model B, 33 fields) were fit
once each via unregularized-by-design (`C=1.0`) L2 logistic regression
on the frozen, 42,535-row primary-split train bucket, through a shared
leakage-safe preprocessing pipeline (train-only fit, uniform numeric
missing-indicator policy, graceful unseen-category handling). Both
models show modest, real discrimination (ROC-AUC 0.657-0.676, PR-AUC
0.25-0.28) against a ~15-16% base rate on validation and benchmark, at
a named, train-prevalence-matched decision threshold decided before any
evaluation-split metric was touched. Both models are already reasonably
well-calibrated out of the box; the predeclared isotonic-vs-sigmoid
comparison selects sigmoid for both, since isotonic's improvement never
clears the predeclared 0.0005 Brier margin. Model A's coefficients
(odds ratios, explicit reference levels, no causal language) surface
`purpose` category and several `addr_state` dummies as the largest
magnitude associations; Model B's addition of `grade`/`sub_grade`/
`int_rate` yields a modest ROC-AUC lift (+0.003 on validation, +0.015 on
benchmark) that is reported strictly as signal-overlap evidence, never a
deployment recommendation. A descriptive-only vintage analysis shows a
gradual ROC-AUC drift downward from early train years into the
non-evaluation-eligible 2014-2016 tail — flagged, not investigated
further here. Non-demographic subgroup diagnostics across
`home_ownership`/`verification_status`/`application_type`/`purpose`
surface modest calibration-gap and false-positive-rate differences
across a few structural segments, reported as descriptive monitoring
only. Ten new, independently tested library modules extend
`src/credit_risk/` with this milestone's capabilities; no model is
recommended for deployment, no boosted trees/random forest were used,
and no approve/decline framing is made for any applicant. **This
milestone's deliverable stops here for review — no further Module C
milestone is started.**
