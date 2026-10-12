"""Module C, Milestone 1: logical table/column constants for the proposed
credit-risk data model.

This is schema **scaffolding only**. No loader in this module reads a real
credit-risk dataset yet -- per the explicit Milestone 1 scope, no dataset
has been downloaded, and no modeling pipeline exists. These constants give
the rest of Module C (target-definition logic, feature-eligibility rules,
and their tests) a single, documented source of truth for table/column
names, instead of hardcoding strings ad hoc, and they mirror exactly the
grain/relationships documented in `docs/credit_risk_data_model.md` -- read
that file first for the "why", this file is just the "what's the literal
column name."
"""
from __future__ import annotations

# --- Primary key / join columns ---
BORROWER_ID_COL = "borrower_id"
APPLICATION_ID_COL = "application_id"
LOAN_ID_COL = "loan_id"

# --- Key lifecycle dates (see docs/credit_risk_data_model.md Sec A) ---
APPLICATION_DATE_COL = "application_date"
UNDERWRITING_DECISION_DATE_COL = "underwriting_decision_date"
ORIGINATION_DATE_COL = "origination_date"

# --- Table grain: one row per borrower_id ---
BORROWER_COLUMNS = [
    BORROWER_ID_COL,
    "first_observed_date",
    "stated_annual_income",
    "employment_length_years",
    "home_ownership",
]

# --- Table grain: one row per application_id (one borrower may have many) ---
LOAN_APPLICATION_COLUMNS = [
    APPLICATION_ID_COL,
    BORROWER_ID_COL,
    APPLICATION_DATE_COL,
    "requested_amount",
    "stated_purpose",
    UNDERWRITING_DECISION_DATE_COL,
    "decision",  # categorical: approved / declined
    "decline_reason",  # populated only when decision == declined
]

# --- Table grain: one row per loan_id (one row per APPROVED+FUNDED application) ---
LOAN_COLUMNS = [
    LOAN_ID_COL,
    APPLICATION_ID_COL,
    BORROWER_ID_COL,
    ORIGINATION_DATE_COL,
    "funded_amount",
    "term_months",
    "interest_rate",
    "installment_amount",
    "risk_grade",
]

# --- Table grain: one row per (loan_id, scheduled_payment_number) ---
SCHEDULED_PAYMENT_NUMBER_COL = "scheduled_payment_number"
SCHEDULED_PAYMENTS_COLUMNS = [
    LOAN_ID_COL,
    SCHEDULED_PAYMENT_NUMBER_COL,
    "scheduled_due_date",
    "scheduled_amount",
    "scheduled_principal",
    "scheduled_interest",
]

# --- Table grain: one row per (loan_id, payment_date) actually received ---
ACTUAL_PAYMENTS_COLUMNS = [
    LOAN_ID_COL,
    "payment_date",
    "amount_paid",
    "principal_paid",
    "interest_paid",
    "days_past_due_at_payment",
]

CREDIT_RISK_TABLES = [
    "borrowers",
    "loan_applications",
    "loans",
    "scheduled_payments",
    "actual_payments",
]

# --- Target-definition columns (see docs/credit_risk_target_definition.md) ---
DATA_AS_OF_DATE_COL = "data_as_of_date"
DEFAULT_EVENT_DATE_COL = "default_event_date"
DELINQUENCY_EVENT_DATE_COL = "delinquency_event_date"
OBSERVATION_WINDOW_END_COL = "observation_window_end"
SUFFICIENT_FOLLOWUP_COL = "sufficient_followup"
DEFAULT_LABEL_COL = "default_12m"
LABEL_STATUS_COL = "label_status"

LABEL_STATUS_DEFAULT = "default"
LABEL_STATUS_NON_DEFAULT = "non_default"
LABEL_STATUS_CENSORED = "censored_insufficient_followup"
