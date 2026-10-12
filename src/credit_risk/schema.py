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

# --- Module C, Milestone 2: real-data target columns ---
# `default_12m` (above) remains the Milestone-1-defined true-event-date
# target shape. Milestone 2 found the real LendingClub file has no genuine
# default/charge-off *event-date* column (see
# docs/moduleC_milestone2_real_data_validation.md Section D) -- so two
# distinct, separately-named targets are constructed instead of silently
# reusing `default_12m` for something it was never validated to mean:
#
# GOVERNANCE (post-Milestone-2 correction -- see
# docs/decision_log.md "Milestone 2 target-governance correction" and
# configs/credit_risk_target_governance.json, the machine-readable source
# of truth):
#
# 1. `terminal_adverse_outcome` -- PRIMARY modeling target for Milestone 3.
#    An "ever defaulted/charged-off by final maturity" target requiring no
#    event date at all, resolved only once a loan's own scheduled term
#    (36 or 60 months) has elapsed. Built only from explicitly mapped
#    resolved terminal `loan_status` categories (`charged_off`,
#    `fully_paid`). It estimates eventual adverse credit outcome among
#    originated loans over a variable, loan-specific horizon -- it is
#    NOT a 12-month probability-of-default estimate and must never be
#    described as one.
# 2. `default_12m_last_pymnt_proxy` -- SECONDARY, sensitivity-analysis-only
#    target. The Milestone-1 compute_default_12m_labels logic, fed a
#    *proxy* event date (last_pymnt_d, or origination_date for the rare
#    zero-payment case). `last_pymnt_d` is NOT a default-event timestamp --
#    it is a last-activity date, so this target carries a documented
#    one-directional bias (see docs/moduleC_milestone2_real_data_validation.md
#    Section D/J) and must never be promoted to primary-target status.
#    Column name makes the proxy nature impossible to miss; never rename
#    this to `default_12m` downstream.
MATURITY_DATE_COL = "maturity_date"
DEFAULT_12M_PROXY_LABEL_COL = "default_12m_last_pymnt_proxy"
LABEL_STATUS_12M_PROXY_COL = "label_status_12m_proxy"
TERMINAL_ADVERSE_OUTCOME_LABEL_COL = "terminal_adverse_outcome"
LABEL_STATUS_TERMINAL_ADVERSE_OUTCOME_COL = "label_status_terminal_adverse_outcome"
SUFFICIENT_FOLLOWUP_TERMINAL_ADVERSE_OUTCOME_COL = "sufficient_followup_terminal_adverse_outcome"
DEFAULT_EVENT_DATE_IS_PROXY_COL = "default_event_date_is_proxy"

# --- Status-mapping categories (Section C) ---
STATUS_CATEGORY_CURRENT = "current_performing"
STATUS_CATEGORY_DELINQUENT = "delinquent"
STATUS_CATEGORY_SEVERE_DEFAULT = "severe_delinquency_or_default"
STATUS_CATEGORY_CHARGED_OFF = "charged_off"
STATUS_CATEGORY_FULLY_PAID = "fully_paid"
STATUS_CATEGORY_AMBIGUOUS_OTHER = "ambiguous_other"

STATUS_CATEGORIES = (
    STATUS_CATEGORY_CURRENT,
    STATUS_CATEGORY_DELINQUENT,
    STATUS_CATEGORY_SEVERE_DEFAULT,
    STATUS_CATEGORY_CHARGED_OFF,
    STATUS_CATEGORY_FULLY_PAID,
    STATUS_CATEGORY_AMBIGUOUS_OTHER,
)
