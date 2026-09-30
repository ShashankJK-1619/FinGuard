# FinGuard — Payments, Fraud & Risk Decisioning Platform

A portfolio-quality FinTech data science project built to learn payments,
fraud, and credit risk data science from first principles. Every technical
decision is documented so it can be explained, not just demonstrated.

**Current scope: Module B (Fraud Detection) only, through Milestone 2.**
Modules A (Payments/Product Analytics) and C (Credit Risk) are planned but
not started. Later Module B milestones (modeling, monitoring) are also not
started.

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

## Project status: modeling has NOT begun

Milestone 1 covered data inspection, join validation, a data dictionary,
and a leakage review. Milestone 2 added a point-in-time-safe feature set
and a transparent baseline modeling table
(`data/processed/baseline_features_v1.parquet`, train only). No model has
been trained yet, and no train/validation/test split has been created or
committed. See [docs/decision_log.md](docs/decision_log.md) for the full
reasoning trail, [docs/feature_dictionary.md](docs/feature_dictionary.md)
for per-feature documentation, and
[docs/milestone2_feature_design.md](docs/milestone2_feature_design.md) for
the Milestone 2 design rationale.

## Repository structure

```
configs/          Configuration (e.g. paths.json) — no hardcoded paths in code
data/raw/         Raw Kaggle CSVs (gitignored, not committed)
data/interim/     Intermediate artifacts (gitignored)
data/processed/   baseline_features_v1.parquet (gitignored, not committed)
docs/             Generated reports + written analysis (data profile, join
                  validation, time structure, data dictionary, leakage
                  review, feature dictionary, milestone 2 design, decision log)
notebooks/        Exploratory notebooks (none yet)
src/data/         Config, DuckDB loaders, inspection/validation scripts
src/features/     Point-in-time-safe feature functions + baseline dataset builder
src/models/       Modeling code (empty — not started)
tests/            pytest suite (incl. tests/features/ leakage-safety tests)
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

## Known limitations (as of Milestone 2)

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

## License / provenance note

This project uses only the public IEEE-CIS Fraud Detection dataset under
Kaggle's competition terms. No proprietary, employer, or confidential data
is used anywhere in this repository.
