# FinGuard — Payments, Fraud & Risk Decisioning Platform

A portfolio-quality FinTech data science project built to learn payments,
fraud, and credit risk data science from first principles. Every technical
decision is documented so it can be explained, not just demonstrated.

**Current scope: Module B (Fraud Detection) through Milestone 5, and
Module A (Payments/Product Analytics) through Milestone 2.** Module C
(Credit Risk) is planned but not started. Later Module B milestones
(production threshold/model selection, monitoring) are also not started.
Module A Milestone 3 (split + baseline model) is not started.

## Business problem (for Module B)

Fraud teams must decide, per transaction, whether to approve, decline, or
route to manual review — under class imbalance, adversarial drift, and a
hard requirement that decisions be explainable and auditable. This module
builds toward a transaction-level fraud classifier and decisioning logic,
starting from rigorous data understanding before any model is trained.

## Business problem (for Module A)

Payments/product teams must decide whether a proposed checkout change
(e.g. a revised retry experience after a declined authorization) is
worth shipping — balancing a completion-rate improvement against fraud
and risk guardrails, under finite sample sizes and genuine statistical
uncertainty. This module builds a synthetic payments session/funnel
layer (Milestone 1) and an experimentation/decisioning layer on top of
it (Milestone 2), **entirely on top of the same public IEEE-CIS
transaction data Module B uses** — there is no real checkout funnel, no
real A/B test, and no real company's experiment anywhere in this module;
see `docs/payments_data_model.md` and
`docs/moduleA_milestone2_experimentation.md` for exactly what is real
data vs. invented, and why.

## Dataset

[IEEE-CIS Fraud Detection](https://www.kaggle.com/competitions/ieee-fraud-detection),
a **public Kaggle competition dataset** hosted by the IEEE Computational
Intelligence Society, with data provided by Vesta Corporation.

**This is public competition data. It is not, and must not be represented
as, data from any company the author has worked for.**

- Grain: one row per transaction (`TransactionID`).
- Tables: `train_transaction`, `train_identity`, `test_transaction`,
  `test_identity`. Transaction and identity tables join on `TransactionID`.
- Target: `isFraud` (binary), present only in `train_transaction`.
- Raw CSVs are **not committed to Git** (see `.gitignore`) and must be
  downloaded manually from Kaggle into `data/raw/`:
  - `data/raw/train_transaction.csv`
  - `data/raw/train_identity.csv`
  - `data/raw/test_transaction.csv`
  - `data/raw/test_identity.csv`

## Project status: calibrated logistic + nonlinear model comparison, not a production model

Milestone 1 covered data inspection, join validation, a data dictionary,
and a leakage review. Milestone 2 added a point-in-time-safe feature set
and a transparent baseline modeling table
(`data/processed/baseline_features_v1.parquet`, train only). Milestone 3
added a chronological train / validation / holdout split (at the time,
called "holdout") and a first interpretable logistic-regression baseline
with full evaluation, threshold analysis, and coefficient interpretation
— see [docs/milestone3_baseline_model.md](docs/milestone3_baseline_model.md).
Milestone 4 recalibrated that baseline's probabilities (sigmoid/Platt and
isotonic) and built an explicit illustrative fraud-decision cost
framework (approve/review/decline) on top of it, then underwent a
**methodology correction**: calibrators must never be evaluated on the
same data used to fit them, so Milestone 3's validation period is now
chronologically sub-split into a `cal_fit` slice (fit only) and a
`selection` slice (evaluation and threshold/cost selection, never fit
on); and the final chronological segment — previously called "holdout" —
is renamed **temporal benchmark**, because it has by now been inspected
during iterative development across Milestones 3 and 4 and is no longer
a pristine, never-touched final test set. See
[docs/milestone4_calibration_decisioning.md](docs/milestone4_calibration_decisioning.md)
for the corrected methodology and current numbers. Milestone 5 added one
nonlinear tabular model (`HistGradientBoostingClassifier`) and compared
it fairly against the logistic baseline — same transparent feature set,
same chronological split, same cal_fit/selection calibration methodology,
same illustrative cost framework — on ranking, calibration, business-
decision performance, and feature importance; the nonlinear model
clearly outranks the logistic baseline on this dataset, but no production
threshold or production model has been chosen (comparison only, per
explicit scope), and no monitoring/decisioning layer exists. See
[docs/milestone5_nonlinear_comparison.md](docs/milestone5_nonlinear_comparison.md)
for the full write-up. See [docs/decision_log.md](docs/decision_log.md)
for the full reasoning trail,
[docs/feature_dictionary.md](docs/feature_dictionary.md) for per-feature
documentation, and
[docs/milestone2_feature_design.md](docs/milestone2_feature_design.md) for
the Milestone 2 design rationale.

## Project status (Module A): synthetic payments funnel + experimentation layer, no models trained

Module A Milestone 1 built a **synthetic payment session/attempt funnel**
on top of the same real IEEE-CIS transactions Module B uses — IEEE-CIS
has no native concept of a checkout session, decline, or retry, so a
synthetic session/attempt layer was added on top of the real transaction
rows, with every synthetic field explicitly labeled as such. It produced
KPIs (authorization rate, retry rate, recovery rate), a funnel, segment
breakdowns, and small rule-based fraud guardrails (not a fitted model).
See [docs/payments_data_model.md](docs/payments_data_model.md) for the
full real-vs-synthetic field inventory and design rationale.

Module A Milestone 2 built a **synthetic A/B-test experimentation layer**
on top of Milestone 1's session table: deterministic, seed-fixed,
session-level randomization into control/treatment arms; a small,
transparent, configurable synthetic treatment-response mechanism (never
touching real transaction history); a two-proportion statistical test
with confidence intervals; a power/MDE reference table; guardrail
metrics evaluated alongside the primary completion-rate metric with an
explicit ship/no-ship decision rule; pre-specified segment heterogeneity
(4 dimensions only, no data dredging); and experiment-health diagnostics
(crossover/duplicate-unit detection, Sample Ratio Mismatch, pre-treatment
covariate balance). **No real experiment occurred and no new ML model
was trained** — see
[docs/moduleA_milestone2_experimentation.md](docs/moduleA_milestone2_experimentation.md)
for the full scenario, design rationale, and limitations, and
[docs/payments_experiment_metric_dictionary.md](docs/payments_experiment_metric_dictionary.md)
for the per-metric numerator/denominator/direction/tier dictionary.

## Repository structure

```
configs/          Configuration (paths.json, cost_assumptions.json,
                  payments_synthetic_assumptions.json,
                  payments_guardrail_rules.json,
                  payments_experiment_assumptions.json) — no hardcoded
                  paths, cost values, or experiment assumptions in code
data/raw/         Raw Kaggle CSVs (gitignored, not committed)
data/interim/     Intermediate artifacts (gitignored)
data/processed/   baseline_features_v1.parquet (gitignored, not committed)
docs/             Generated reports + written analysis (data profile, join
                  validation, time structure, data dictionary, leakage
                  review, feature dictionary, milestone 2/3/4/5 design
                  docs, Module A payments data model + Milestone 2
                  experimentation docs + metric dictionary, decision log)
notebooks/        Exploratory notebooks (none yet)
src/data/         Config, DuckDB loaders, inspection/validation scripts
src/features/     Point-in-time-safe feature functions + baseline dataset builder
src/models/       Chronological split, preprocessing pipeline,
                  logistic-regression baseline, calibration, illustrative
                  cost-based decisioning, and nonlinear model comparison
                  (schema.py, split.py, preprocessing.py,
                  train_baseline.py, calibration.py, cost_model.py,
                  train_calibrated.py, train_nonlinear.py)
src/payments/     Module A: synthetic session/attempt generator, KPI/
                  funnel/retry-recovery/segmentation/guardrails
                  (Milestone 1), and experiment assignment/synthetic
                  treatment effects/stats/power/guardrails/segments/
                  diagnostics (Milestone 2) — all DuckDB-SQL-first,
                  config-driven, no fitted models
tests/            pytest suite (incl. tests/features/ and tests/models/
                  leakage-safety tests, and tests/payments/ for Module A)
```

## Environment setup

```bash
pyenv local 3.12.7          # already configured via .python-version
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Dependencies are intentionally minimal: pandas, numpy, scikit-learn,
statsmodels, matplotlib, duckdb, pytest, tabulate. No Spark, no cloud
infra, no deep learning — none of the later-stage tooling is needed yet
and won't be added until a milestone actually requires it.

## Running Milestone 1 inspection

After placing the four raw CSVs in `data/raw/`:

```bash
source .venv/bin/activate
python -m src.data.run_all      # runs all four scripts below in sequence
```

Or individually:

```bash
python -m src.data.inspect_dataset     # docs/data_profile.md
python -m src.data.validate_join       # docs/join_validation.md
python -m src.data.analyze_time        # docs/time_structure.md
python -m src.data.leakage_checks      # docs/leakage_checks_empirical.md
```

Run tests:

```bash
pytest -v
```

Tests that require the raw CSVs auto-skip with a clear reason if the data
isn't present yet (see `tests/conftest.py`).

## Running Milestone 2 feature engineering

```bash
.venv/bin/python -m src.features.build_baseline_dataset
```

Writes `data/processed/baseline_features_v1.parquet` (train only, one row
per `TransactionID`). See `docs/feature_dictionary.md` for what's in it.

## Running Milestone 3 baseline model

```bash
.venv/bin/python -m src.models.train_baseline
```

Requires `data/processed/baseline_features_v1.parquet` (Milestone 2)
already built. Prints the split report, validation/temporal-benchmark
metrics, threshold analysis, and coefficient table to stdout, and writes
`data/interim/milestone3_metrics.json`. See
[docs/milestone3_baseline_model.md](docs/milestone3_baseline_model.md) for
the full write-up.

## Running Milestone 4 calibration & decisioning

```bash
.venv/bin/python -m src.models.train_calibrated
```

Requires `data/processed/baseline_features_v1.parquet` (Milestone 2).
Reuses Milestone 3's chronological train/validation/temporal-benchmark
split unchanged, then further sub-splits the validation period
chronologically into `cal_fit` (earlier half) and `selection` (later
half). Sigmoid and isotonic calibrators are fit on `cal_fit` only;
calibration quality (Brier score, reliability tables) and the cost-based
threshold/cost grid + sensitivity analysis are evaluated on `selection` —
data the calibrator was never fit on — and, separately, once each on the
temporal benchmark. Also runs the day_index/week_index ablation and
writes `data/interim/milestone4_metrics.json`. Cost assumptions,
including manual-review capacity, review fraud-catch rate, and review
false-positive rate, are read from `configs/cost_assumptions.json` — all
are **hypothetical, illustrative values**, not real financial figures
(see the file's own `_disclaimer` field). See
[docs/milestone4_calibration_decisioning.md](docs/milestone4_calibration_decisioning.md)
for the full write-up.

## Running Milestone 5 nonlinear model comparison

```bash
.venv/bin/python -m src.models.train_nonlinear
```

Requires `data/processed/baseline_features_v1.parquet` (Milestone 2).
Reuses Milestone 3's chronological split and Milestone 4's cal_fit/
selection sub-split and calibration methodology unchanged. Trains a
`HistGradientBoostingClassifier` on the same transparent feature set as
the logistic baseline (tree-native preprocessing — no scaling, ordinal-
not one-hot-encoded categoricals — see
`src/models/train_nonlinear.py`), tunes 4 hyperparameter candidates on
TRAIN-ONLY time-aware folds, calibrates it with the same sigmoid/
isotonic out-of-sample methodology, runs the same illustrative cost-based
decision-policy comparison and sensitivity analysis, computes permutation
feature importance, and repeats the day_index/week_index ablation for
this model. Prints a structured comparison to stdout and writes
`data/interim/milestone5_metrics.json`. See
[docs/milestone5_nonlinear_comparison.md](docs/milestone5_nonlinear_comparison.md)
for the full write-up.

## Running Module A Milestone 1 (payments funnel + KPIs)

```bash
.venv/bin/python -m src.payments.build_payments_report
```

Requires the same raw CSVs as Module B (`data/raw/train_transaction.csv`,
`data/raw/train_identity.csv`) — Module A reads them directly and does
**not** depend on Module B's processed features. Builds the synthetic
session/attempt layer (seeded, deterministic — see
`configs/payments_synthetic_assumptions.json`), computes KPIs, the
funnel, retry/recovery analysis, segmentation, and rule-based fraud
guardrails (thresholds in `configs/payments_guardrail_rules.json`),
prints a structured report to stdout, and writes
`data/interim/payments_milestone1_metrics.json`. See
[docs/payments_data_model.md](docs/payments_data_model.md) for the full
write-up.

## Running Module A Milestone 2 (payments experimentation)

```bash
.venv/bin/python -m src.payments.build_experiment_report
```

Requires Milestone 1's raw-data dependency (above) — rebuilds the same
synthetic session/attempt layer, then deterministically assigns each
session to a control/treatment arm, applies the synthetic
treatment-response mechanism, runs crossover/SRM diagnostics and
pre-treatment covariate balance, computes primary/secondary/guardrail
metrics (DuckDB SQL), the two-proportion test and confidence interval,
the power/MDE reference table, the ship/no-ship decision rule, and
pre-specified segment heterogeneity. All assumptions/thresholds are read
from `configs/payments_experiment_assumptions.json` (see its own
`_disclaimer` field — nothing in it is a measured or claimed real-world
effect size). Prints a structured report to stdout and writes
`data/interim/payments_milestone2_experiment_metrics.json`. See
[docs/moduleA_milestone2_experimentation.md](docs/moduleA_milestone2_experimentation.md)
for the full write-up and
[docs/payments_experiment_metric_dictionary.md](docs/payments_experiment_metric_dictionary.md)
for the metric dictionary.

## Known limitations (as of Module B Milestone 5 / Module A Milestone 2)

- Dataset inspection scripts have been run against the actual Kaggle data.
  See `docs/data_profile.md`, `docs/join_validation.md`,
  `docs/time_structure.md`, and `docs/leakage_checks_empirical.md` for the
  generated figures, and the Milestone 1 report (shared in the assistant's
  conversation summary) for the headline numbers.
- The business meaning of every masked feature group (`C*`, `D*`, `M*`,
  `V*`, `id_*`) is **explicitly unknown** — see
  [docs/data_dictionary.md](docs/data_dictionary.md). No interpretation is
  assumed anywhere in this project for those columns.
- Whether `C`/`D`/`V` features were computed by Vesta using only
  point-in-time information (safe) or using global/future aggregation
  (leakage risk) is **unresolved** — see
  [docs/leakage_review.md](docs/leakage_review.md). No decision to use or
  exclude these features has been made.
- The Milestone 2 baseline deliberately excludes `C*`, `D*`, `M*`, `V*`,
  and raw `id_01`–`id_38` values. See
  [docs/milestone2_feature_design.md](docs/milestone2_feature_design.md)
  §6 for why each group was excluded — most are excluded because their
  masked/undocumented meaning makes them indefensible in a *transparent*
  baseline, not because they're necessarily unusable later.
- `data/processed/baseline_features_v1.parquet` is gitignored (generated
  data, not committed) and must be regenerated by running
  `python -m src.features.build_baseline_dataset` after Milestone 1's raw
  CSVs are in place.
- **Milestone 3's logistic-regression baseline has well-ordered but
  miscalibrated probabilities** (a direct effect of
  `class_weight="balanced"`) — fixed in Milestone 4 via post-hoc
  calibration fit on a `cal_fit` slice carved out of the validation
  period, and evaluated out-of-sample on a disjoint `selection` slice and
  on the temporal benchmark (Brier score cut by roughly 8x on both;
  ROC-AUC essentially unchanged, as expected for a ranking-preserving
  transform). **Sigmoid was selected**, not isotonic: once evaluated
  strictly out-of-sample, isotonic no longer beats sigmoid on Brier score
  (it is in fact slightly worse on both the `selection` split and the
  temporal benchmark) — isotonic's earlier apparent edge was an artifact
  of being evaluated on its own fit data, which this correction removed.
  See [docs/milestone4_calibration_decisioning.md](docs/milestone4_calibration_decisioning.md)
  §3–4.
- **No production deployment threshold has been chosen.** Milestone 4
  built an illustrative, config-driven cost framework (now including
  configurable manual-review capacity, review fraud-catch rate, and
  review false-positive rate) and explored a threshold/cost tradeoff grid
  plus a 10-variation sensitivity analysis on the `selection` split only
  (§6–7 of the same doc), but every cost figure is explicitly
  hypothetical and no threshold is labeled a final production policy.
- **Manual review has finite capacity and is not a perfect fraud
  backstop** in the Milestone 4 cost framework (corrected from an earlier
  unlimited/perfect-review assumption) — `review_capacity_frac`,
  `review_fraud_catch_rate`, and `review_false_positive_rate` are all
  configurable in `configs/cost_assumptions.json` and remain illustrative,
  not measured from a real review operation. When desired review volume
  exceeds capacity, the highest-risk transactions are prioritized and
  the rest fall back to **approve, not decline** — declining a
  review-band transaction outright would be stricter than what the
  threshold pair itself specifies. See the same doc, §5 and §9.
- **The final chronological segment is now called "temporal benchmark,"
  not "holdout."** It has been inspected multiple times across
  Milestones 3 and 4 (baseline metrics, calibration comparison, policy
  evaluation) and must not be treated as a pristine, never-touched final
  test set.
- **`day_index`/`week_index` (linear time-drift proxies) were found to
  slightly *hurt* temporal-benchmark ranking performance when included**
  (Milestone 4 ablation, same doc §8) — suggestive from one split, not
  conclusive, but it reinforces Milestone 3's extrapolation-risk concern
  about these two features with no offsetting benefit found here.
- **Milestone 5 added one nonlinear model (`HistGradientBoostingClassifier`)
  and compared it fairly against the logistic baseline** — same
  transparent feature set, split, calibration methodology, and cost
  framework. It clearly outranks the logistic baseline on this dataset
  (calibrated ROC-AUC 0.79 vs. 0.75, PR-AUC 0.20 vs. 0.13 on `selection`)
  and, under the same illustrative cost assumptions, its own
  selection-chosen decision policy roughly doubles fraud recall at a
  15-18% lower illustrative total cost than the logistic model's own
  policy — at the cost of materially more false positives. No random
  forest, XGBoost, or LightGBM comparison has been run — only
  `HistGradientBoostingClassifier`, per the explicit preference order
  given for this milestone. See
  [docs/milestone5_nonlinear_comparison.md](docs/milestone5_nonlinear_comparison.md).
- **Feature importance for the nonlinear model is permutation importance
  only** (`HistGradientBoostingClassifier` has no native
  `feature_importances_`), computed on the `selection` split, and is
  explicitly not a causal measure — same caveat the project already
  applies to logistic coefficients. Top-10 overlap with logistic
  |coefficient| rankings was zero at the literal feature-name level, but
  this is a granularity artifact (one-hot level names vs. whole raw
  columns), not a real disagreement — read at the column level, both
  models independently surface `ProductCD`, `DeviceType`,
  `email_domain_match`, and identity/device-missingness signal near the
  top. See `docs/milestone5_nonlinear_comparison.md` §7.
- **The `day_index`/`week_index` ablation was repeated for the nonlinear
  model and found the same qualitative result as Milestone 4's logistic
  ablation**: excluding them very slightly improved temporal-benchmark
  ROC-AUC, and both features scored exactly zero permutation importance.
  The nonlinear model's default preferred configuration excludes them
  per this evidence (and per explicit instruction), though the
  Milestone 5 head-to-head comparison against the logistic baseline above
  uses the full feature set, to stay apples-to-apples with that
  already-established baseline specifically.
- **Still no production deployment threshold or production model has
  been chosen** — Milestone 5 is a comparison exercise, not a selection
  decision. A likely next milestone is picking one specific model and one
  specific threshold pair as an actual proposed policy, plus a
  monitoring/drift-detection layer — neither attempted in any milestone so
  far.
- **Module A's entire session/attempt/experiment layer is synthetic**,
  built on top of the same real IEEE-CIS transactions Module B uses — see
  [docs/payments_data_model.md](docs/payments_data_model.md) for the
  real-vs-synthetic field inventory. No real payments funnel, checkout
  experience, or A/B test is described anywhere in this module.
- **Module A Milestone 2's randomization unit is the session, not the
  customer** — this dataset has no real, documented customer identifier,
  so a session-level unit was used instead of the generally-preferred
  customer-level unit, with the resulting contamination risk (the same
  real person appearing in both arms across multiple sessions) stated as
  an unmeasurable limitation rather than hidden. See
  [docs/moduleA_milestone2_experimentation.md](docs/moduleA_milestone2_experimentation.md)
  §2.
- **Module A Milestone 2's synthetic treatment effect is a configured
  modeling choice (`recovery_uplift_prob` in
  `configs/payments_experiment_assumptions.json`), not a measured or
  claimed real-world effect size**, and was fixed before running against
  real data and never adjusted afterward to change the outcome.
- **Module A Milestone 2's real-data run returned a `no-ship` decision**:
  the primary metric (`payment_completion_rate`) showed a statistically
  significant, positive absolute uplift of 0.0071 (control 0.8692 vs.
  treatment 0.8763, p ≈ 2.2e-18), with no guardrail breach — but the
  uplift fell below the predeclared minimum practical-significance
  threshold of 0.01, so the decision rule correctly withheld a "ship"
  recommendation despite statistical significance. This is a direct,
  real-data demonstration of the statistical-vs-practical-significance
  distinction the decision rule is designed to enforce, not evidence
  about any real product. See
  [docs/moduleA_milestone2_experimentation.md](docs/moduleA_milestone2_experimentation.md)
  for full results.
- **No multiple-comparison correction is applied to Module A Milestone
  2's segment heterogeneity table** (4 pre-registered dimensions only,
  not an open-ended search) — individual segment-level p-values should
  not be read as independently confirmed discoveries.
- **Module A Milestone 3 (split + baseline model) has not been started.**

## License / provenance note

This project uses only the public IEEE-CIS Fraud Detection dataset under
Kaggle's competition terms. No proprietary, employer, or confidential data
is used anywhere in this repository.
