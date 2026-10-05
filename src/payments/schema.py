"""Module A, Milestone 1: column-name constants for the real transaction
table and the two synthetic tables (`payment_sessions`, `payment_attempts`).

Centralizing names here (rather than scattering string literals across
generate_sessions.py/kpi.py/funnel.py/...) is the same auditability pattern
src/models/schema.py uses for Module B. See docs/payments_data_model.md
Section 2 for which of these are real vs. synthetic.
"""
from __future__ import annotations

# --- Real entity (unmodified IEEE-CIS columns; see docs/payments_data_model.md Sec 2a) ---
ID_COL = "TransactionID"
TIME_COL = "TransactionDT"
AMOUNT_COL = "TransactionAmt"
TARGET_COL = "isFraud"

REAL_CATEGORICAL_COLUMNS = ["ProductCD", "card4", "card6", "DeviceType"]
REAL_EMAIL_COLUMNS = ["P_emaildomain", "R_emaildomain"]

# Minimal real-column set Module A reads from the raw CSVs. Deliberately
# does not include any masked C*/D*/M*/V*/id_* column (see
# docs/payments_data_model.md Sec 2a) and does not depend on Module B's
# processed baseline_features_v1.parquet (Module A reads raw CSVs directly
# via src.data.loaders, keeping the two modules independent).
REAL_TRANSACTION_COLUMNS = (
    [ID_COL, TIME_COL, AMOUNT_COL, TARGET_COL] + REAL_CATEGORICAL_COLUMNS + REAL_EMAIL_COLUMNS
)

# --- Synthetic entity: payment_sessions (grain: one row per session_id) ---
SESSION_ID_COL = "session_id"
SESSION_COLUMNS = [
    SESSION_ID_COL,
    "transaction_id",  # FK to real TransactionID; NULL for abandonment-only sessions
    "is_synthetic_abandonment",
    "final_outcome",  # 'authorized' | 'abandoned'
    "attempt_count",
    "session_start_offset_seconds",
    "ProductCD",
    "card4",
    "card6",
    "DeviceType",
    "TransactionAmt",
    "email_domain_match",
    TARGET_COL,  # real isFraud, NULL for abandonment-only sessions (no fraud determination possible)
]

FINAL_OUTCOME_AUTHORIZED = "authorized"
FINAL_OUTCOME_ABANDONED = "abandoned"

# --- Synthetic entity: payment_attempts (grain: one row per (session_id, attempt_number)) ---
ATTEMPT_COLUMNS = [
    "attempt_id",
    SESSION_ID_COL,
    "attempt_number",  # 1-indexed
    "attempt_offset_seconds",
    "outcome",  # 'declined' | 'authorized'
    "decline_reason_code",  # NULL iff outcome == 'authorized'
]

ATTEMPT_OUTCOME_DECLINED = "declined"
ATTEMPT_OUTCOME_AUTHORIZED = "authorized"

# --- Module A, Milestone 2: experimentation column/label constants ---
# See docs/moduleA_milestone2_experimentation.md for the full design. These
# columns are added ON TOP OF `payment_sessions` by the experiment layer
# (src.payments.experiment_assignment / experiment_effects) — they never
# exist in the Milestone 1 `payment_sessions` table itself, so the
# baseline-vs-experiment boundary is visible in the schema, not just prose.
ASSIGNMENT_COL = "assignment"
ASSIGNMENT_CONTROL = "control"
ASSIGNMENT_TREATMENT = "treatment"

# True iff attempt_count >= the configured eligibility threshold (Sec E) —
# i.e. the session actually went through at least one retry, so a
# retry-experience treatment could plausibly have affected it at all.
RETRY_ELIGIBLE_COL = "retry_eligible"

# True iff this specific session's outcome was changed by the synthetic
# treatment-response mechanism (Sec E). Always False for control sessions
# and for any real-linked session (real transaction history is never
# altered).
TREATMENT_EFFECT_APPLIED_COL = "treatment_effect_applied"

# The outcome the experiment's metrics are computed on: equal to the
# Milestone 1 `final_outcome` everywhere EXCEPT the sessions where
# `treatment_effect_applied` is True, where it is 'authorized' instead of
# 'abandoned'. Keeping this as a separate column (rather than overwriting
# `final_outcome`) preserves the Milestone 1 baseline table unchanged and
# auditable.
EXPERIMENT_OUTCOME_COL = "experiment_final_outcome"

EXPERIMENT_SESSION_COLUMNS = SESSION_COLUMNS + [
    ASSIGNMENT_COL,
    RETRY_ELIGIBLE_COL,
    TREATMENT_EFFECT_APPLIED_COL,
    EXPERIMENT_OUTCOME_COL,
]
