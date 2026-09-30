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
