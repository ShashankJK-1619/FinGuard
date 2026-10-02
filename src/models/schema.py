"""Milestone 3: model-input column roles for the interpretable baseline.

Centralizes every include/exclude decision for logistic regression in one
auditable place, the same pattern as
src/features/categorical_features.py::RAW_CATEGORY_COLUMNS. Applies to
data/processed/baseline_features_v1.parquet (Milestone 2 output) only.
C1-C14, D1-D15, M1-M9, V1-V339, and raw id_01-id_38 are not present in that
table at all (Milestone 2 excluded them), so there is nothing to exclude
here for those groups beyond re-verifying their absence (see
tests/models/test_schema.py).

## Column-role policy

1. Identifiers/label (TARGET_COL, ID_COL, TIME_COL): never model inputs.
   TIME_COL (TransactionDT) is the split key and is additionally excluded
   from the feature matrix itself: it is an offset from an undisclosed
   reference point (docs/time_structure.md), so a raw coefficient on it
   ("one more second of unknown-origin offset changes log-odds by X") is
   not interpretable, and the day/week/hour/day-of-week features below
   already carry the same information at an interpretable granularity.

2. Raw high-cardinality or masked-code columns that already have a
   Milestone-2-built derived summary (frequency count, recency gap, or
   missingness flag) are excluded as direct raw inputs in favor of that
   summary. Rationale: including the raw value would mean either a
   several-hundred-column one-hot block with unseen-category risk at
   scoring time (card1: 13,553 distinct; addr1: hundreds), or a masked
   free-text field with no defensible transform (DeviceInfo). The
   engineered proxy already captures the "familiarity/velocity" signal the
   project decided was defensible in Milestone 2.
   Excluded this way: card1, addr1, P_emaildomain, R_emaildomain,
   DeviceInfo.

3. Genuinely low-cardinality documented categoricals are one-hot encoded.
   Their missing values are imputed as an explicit "missing" category
   (not dropped, not silently zero) so absence is itself a visible level.

4. Masked numeric-coded columns with no Milestone-2 derived summary
   (card2, card3, card5, addr2) are kept as numeric model inputs. This
   imposes an unverified ordinal/interval assumption on values whose true
   mapping is undisclosed — flagged explicitly here and in
   docs/milestone3_baseline_model.md as a limitation, not hidden. They are
   not excluded outright because Milestone 2 did not build an alternative
   representation for them and retroactively engineering one is out of
   Milestone 3's scope (no new feature engineering, only modeling on top
   of the existing baseline table).

5. Genuinely continuous documented numeric columns (dist1, dist2) are kept
   as numeric inputs (units undisclosed per docs/feature_dictionary.md,
   but "distance" is inherently continuous, unlike an arbitrary code).

6. TransactionAmt is excluded in favor of TransactionAmt_log1p (same
   information, already built in Milestone 2 specifically so a linear
   model isn't dominated by a long right tail) — including both would add
   a redundant, highly collinear pair of columns for no benefit.
   hour_of_day is excluded in favor of hour_sin/hour_cos for the same
   reason: a raw 0-23 integer implies a false discontinuity between hour
   23 and hour 0 that the sin/cos pair exists specifically to avoid.

7. Binary 0/1 flags are passed through unscaled, so each coefficient is
   directly a log-odds effect of "flag on vs. off" rather than an
   arbitrary standardized unit. Only missingness flags for columns that
   are NOT otherwise in the model (see rule 2) are included, to avoid
   duplicating the same missing/not-missing bit that a one-hot "missing"
   category would already encode for DeviceType (see rule 3): DeviceType
   is a model input, so DeviceType_missing is excluded as redundant with
   DeviceType's own "missing" one-hot level.

8. day_index and week_index are ordinal, monotonically-increasing-in-time
   counters. They are kept as numeric drift/trend inputs — a standard,
   legitimate way to let an interpretable model capture "fraud prevalence
   shifted over the observed window" — but this is flagged as a specific
   generalization risk in docs/milestone3_baseline_model.md: any row
   scored in the future has a day_index/week_index value the model never
   saw in training, i.e. a linear extrapolation outside the training
   domain. This is different from TIME_COL's exclusion (TransactionDT is
   excluded for lack of an interpretable unit and pure redundancy with
   day/week/hour features, not for this extrapolation reason).
"""
from __future__ import annotations

import re

TARGET_COL = "isFraud"
ID_COL = "TransactionID"
TIME_COL = "TransactionDT"

# Regenerated from Milestone 2's baseline table — asserted absent by
# tests/models/test_schema.py, not assumed.
MASKED_COLUMN_PATTERN = re.compile(r"^(C\d+|D\d+|M\d+|V\d+|id_\d+)$")

# Rule 2: raw columns replaced by their Milestone-2-built derived proxy.
EXCLUDED_RAW_WITH_PROXY = ["card1", "addr1", "P_emaildomain", "R_emaildomain", "DeviceInfo"]

# Rule 6: raw columns replaced by an already-built transformed version.
EXCLUDED_RAW_REDUNDANT = ["TransactionAmt", "hour_of_day"]

# Rule 1: split key, excluded from the feature matrix itself.
EXCLUDED_TIME_KEY = [TIME_COL]

# Rule 7: DeviceType is a model input (see CATEGORICAL_FEATURES), so its
# missingness flag would duplicate DeviceType's own "missing" one-hot level.
EXCLUDED_REDUNDANT_MISSING_FLAG = ["DeviceType_missing"]

EXCLUDED_COLUMNS = (
    [ID_COL, TARGET_COL]
    + EXCLUDED_TIME_KEY
    + EXCLUDED_RAW_WITH_PROXY
    + EXCLUDED_RAW_REDUNDANT
    + EXCLUDED_REDUNDANT_MISSING_FLAG
)

# Rule 4 + rule 5 + rule 8 + engineered frequency/recency counts: numeric
# inputs, median-imputed then standard-scaled. seconds_since_last_* get an
# automatic missing-indicator from SimpleImputer(add_indicator=True)
# (src/models/preprocessing.py) because NaN there means "no prior
# transaction with this key" (first occurrence), a real state, not a
# measurement gap to be silently averaged away.
NUMERIC_FEATURES = [
    "TransactionAmt_log1p",
    "card2",  # masked code, unverified ordinal assumption — see rule 4
    "card3",  # masked code, unverified ordinal assumption — see rule 4
    "card5",  # masked code, unverified ordinal assumption — see rule 4
    "addr2",  # masked code, unverified ordinal assumption — see rule 4
    "dist1",
    "dist2",
    "day_index",  # drift/trend term — extrapolation risk, see rule 8
    "week_index",  # drift/trend term — extrapolation risk, see rule 8
    "hour_sin",
    "hour_cos",
    "card1_prior_count",
    "addr1_prior_count",
    "P_emaildomain_prior_count",
    "DeviceInfo_prior_count",
    "seconds_since_last_card1",
    "seconds_since_last_addr1",
]

# Masked numeric codes within NUMERIC_FEATURES carrying the rule-4 caveat —
# used to exclude them from the "transparent feature" coefficient
# interpretation in docs/milestone3_baseline_model.md, not from the model
# itself.
MASKED_CODE_NUMERIC_FEATURES = ["card2", "card3", "card5", "addr2"]

# Rule 3: low-cardinality documented categoricals, one-hot encoded with an
# explicit "missing" level (src/models/preprocessing.py).
CATEGORICAL_FEATURES = [
    "ProductCD",
    "card4",
    "card6",
    "DeviceType",
    "email_domain_match",
    "day_of_week_index",  # phase-unknown bucket identity, not a magnitude — see docs/feature_dictionary.md
]

# Rule 7: binary 0/1 flags, passed through unscaled.
BINARY_FEATURES = [
    "has_identity_record",
    "DeviceInfo_missing",
    "P_emaildomain_missing",
    "R_emaildomain_missing",
]

ALL_MODEL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES + BINARY_FEATURES

# Milestone 4 (rule 8 follow-up): day_index/week_index are the two model
# inputs flagged with the extrapolation risk above — any row scored after
# the training window ends has a day_index/week_index value outside
# everything the model saw during fitting, which is different from every
# other numeric feature here (those are bounded quantities/counts, not a
# monotonically increasing clock). This ablation set removes exactly those
# two columns so Milestone 4 can measure how much of the model's ranking
# performance depends on temporal drift proxies versus genuinely
# behavioral signal, and how that gap changes on the holdout period.
TIME_DRIFT_FEATURES = ["day_index", "week_index"]
NUMERIC_FEATURES_NO_TIME_DRIFT = [f for f in NUMERIC_FEATURES if f not in TIME_DRIFT_FEATURES]
