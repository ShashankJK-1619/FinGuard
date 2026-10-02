# FinGuard — Payments, Fraud & Risk Decisioning Platform

A portfolio-quality FinTech data science project built to learn payments,
fraud, and credit risk data science from first principles. Every technical
decision is documented so it can be explained, not just demonstrated.

**Current scope: Module B (Fraud Detection) only, through Milestone 4.**
Modules A (Payments/Product Analytics) and C (Credit Risk) are planned but
not started. Later Module B milestones (a non-linear model, production
threshold selection, monitoring) are also not started.

## Business problem (for Module B)

Fraud teams must decide, per transaction, whether to approve, decline, or
route to manual review — under class imbalance, adversarial drift, and a
hard requirement that decisions be explainable and auditable. This module
builds toward a transaction-level fraud classifier and decisioning logic,
starting from rigorous data understanding before any model is trained.

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

## Project status: calibrated probabilities + illustrative cost decisioning, not a production model

Milestone 1 covered data inspection, join validation, a data dictionary,
and a leakage review. Milestone 2 added a point-in-time-safe feature set
and a transparent baseline modeling table
(`data/processed/baseline_features_v1.parquet`, train only). Milestone 3
added a chronological train/validation/holdout split and a first
interpretable logistic-regression baseline with full evaluation,
threshold analysis, and coefficient interpretation — see
[docs/milestone3_baseline_model.md](docs/milestone3_baseline_model.md).
Milestone 4 recalibrated that baseline's probabilities (sigmoid/Platt and
isotonic, fit on validation only, evaluated on holdout), built an
explicit illustrative fraud-decision cost framework (approve/review/
decline), explored threshold/cost tradeoffs and their sensitivity to cost
assumptions, and ran a time-feature ablation — see
[docs/milestone4_calibration_decisioning.md](docs/milestone4_calibration_decisioning.md).
No production threshold has been chosen (threshold/cost exploration only,
per explicit scope), no non-linear model has been tried, and no
monitoring/decisioning layer exists. See
[docs/decision_log.md](docs/decision_log.md) for the full reasoning
trail, [docs/feature_dictionary.md](docs/feature_dictionary.md) for
per-feature documentation, and
[docs/milestone2_feature_design.md](docs/milestone2_feature_design.md) for
the Milestone 2 design rationale.

## Repository structure

```
configs/          Configuration (paths.json, cost_assumptions.json) — no
                  hardcoded paths or cost values in code
data/raw/         Raw Kaggle CSVs (gitignored, not committed)
data/interim/     Intermediate artifacts (gitignored)
data/processed/   baseline_features_v1.parquet (gitignored, not committed)
docs/             Generated reports + written analysis (data profile, join
                  validation, time structure, data dictionary, leakage
                  review, feature dictionary, milestone 2/3/4 design docs,
                  decision log)
notebooks/        Exploratory notebooks (none yet)
src/data/         Config, DuckDB loaders, inspection/validation scripts
src/features/     Point-in-time-safe feature functions + baseline dataset builder
src/models/       Chronological split, preprocessing pipeline,
                  logistic-regression baseline, calibration, and
                  illustrative cost-based decisioning (schema.py, split.py,
                  preprocessing.py, train_baseline.py, calibration.py,
                  cost_model.py, train_calibrated.py)
tests/            pytest suite (incl. tests/features/ and tests/models/
                  leakage-safety tests)
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
already built. Prints the split report, validation/holdout metrics,
threshold analysis, and coefficient table to stdout, and writes
`data/interim/milestone3_metrics.json`. See
[docs/milestone3_baseline_model.md](docs/milestone3_baseline_model.md) for
the full write-up.

## Running Milestone 4 calibration & decisioning

```bash
.venv/bin/python -m src.models.train_calibrated
```

Requires `data/processed/baseline_features_v1.parquet` (Milestone 2).
Reuses Milestone 3's chronological split and baseline pipeline, fits
sigmoid and isotonic calibrators on validation only, evaluates calibration
on holdout, runs the illustrative cost-based threshold grid and
sensitivity analysis (validation only), runs the day_index/week_index
ablation, and writes `data/interim/milestone4_metrics.json`. Cost
assumptions are read from `configs/cost_assumptions.json` — all three are
**hypothetical, illustrative values**, not real financial figures (see
the file's own `_disclaimer` field). See
[docs/milestone4_calibration_decisioning.md](docs/milestone4_calibration_decisioning.md)
for the full write-up.

## Known limitations (as of Milestone 4)

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
  `class_weight="balanced"`) — fixed in Milestone 4 via sigmoid/isotonic
  post-hoc calibration fit on validation only (Brier score cut roughly
  7–9x on both validation and holdout; ROC-AUC essentially unchanged, as
  expected for a ranking-preserving transform). See
  [docs/milestone4_calibration_decisioning.md](docs/milestone4_calibration_decisioning.md)
  §3–4.
- **No production deployment threshold has been chosen.** Milestone 4
  built an illustrative, config-driven cost framework and explored a
  44-pair threshold/cost tradeoff grid plus an 8-variation sensitivity
  analysis (§6–7 of the same doc), but every cost figure is explicitly
  hypothetical and no threshold is labeled a final production policy.
- **Manual review is modeled as a perfect fraud backstop** in the
  Milestone 4 cost framework — no real review-team accuracy data exists
  in this project to build a more realistic partial-catch-rate
  assumption. See the same doc, §9.
- **`day_index`/`week_index` (linear time-drift proxies) were found to
  slightly *hurt* holdout ranking performance when included** (Milestone
  4 ablation, same doc §8) — suggestive from one split, not conclusive,
  but it reinforces Milestone 3's extrapolation-risk concern about these
  two features with no offsetting benefit found here.
- No non-linear model (random forest, gradient boosting, etc.) has been
  tried — Milestones 3–4 were scoped to an interpretable linear baseline
  only, by explicit instruction.

## License / provenance note

This project uses only the public IEEE-CIS Fraud Detection dataset under
Kaggle's competition terms. No proprietary, employer, or confidential data
is used anywhere in this repository.
