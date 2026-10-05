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
