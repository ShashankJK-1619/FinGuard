"""Pure unit tests for Section D's two real-data targets, against small
constructed toy fixtures (never the real file) -- mirrors Milestone 1's
toy-fixture-only testing discipline for target-definition logic."""
import pandas as pd
import pytest

from src.credit_risk.config import TargetDefinitionAssumptions
from src.credit_risk.schema import (
    DEFAULT_12M_PROXY_LABEL_COL,
    TERMINAL_ADVERSE_OUTCOME_LABEL_COL,
    LABEL_STATUS_12M_PROXY_COL,
    LABEL_STATUS_CENSORED,
    LABEL_STATUS_DEFAULT,
    LABEL_STATUS_NON_DEFAULT,
    LABEL_STATUS_TERMINAL_ADVERSE_OUTCOME_COL,
    LOAN_ID_COL,
    MATURITY_DATE_COL,
    ORIGINATION_DATE_COL,
)
from src.credit_risk.target_construction import (
    compute_default_12m_proxy_target,
    compute_terminal_adverse_outcome_target,
)

ASSUMPTIONS = TargetDefinitionAssumptions(
    outcome_window_months=12,
    default_event_statuses=("Charged Off", "Default"),
    delinquency_event_statuses=("In Grace Period",),
)

DATA_AS_OF = pd.Timestamp("2019-04-01")


def _base_row(**overrides):
    row = dict(
        loan_id=1,
        issue_d="n/a",
        term=" 36 months",
        loan_status="Current",
        last_pymnt_d="n/a",
        origination_date=pd.Timestamp("2015-01-01"),
        last_pymnt_date=pd.NaT,
        term_months=36,
        maturity_date=pd.Timestamp("2018-01-01"),
        status_category="current_performing",
        meets_credit_policy=True,
    )
    row.update(overrides)
    return row


# --- Terminal-status target ---

def test_terminal_target_charged_off_past_maturity_is_default():
    df = pd.DataFrame(
        [_base_row(loan_id=1, status_category="charged_off", maturity_date=pd.Timestamp("2018-01-01"))]
    )
    out = compute_terminal_adverse_outcome_target(df, data_as_of_date=DATA_AS_OF)
    assert out.loc[0, TERMINAL_ADVERSE_OUTCOME_LABEL_COL] == 1
    assert out.loc[0, LABEL_STATUS_TERMINAL_ADVERSE_OUTCOME_COL] == LABEL_STATUS_DEFAULT


def test_terminal_target_fully_paid_past_maturity_is_non_default():
    df = pd.DataFrame(
        [_base_row(loan_id=1, status_category="fully_paid", maturity_date=pd.Timestamp("2018-01-01"))]
    )
    out = compute_terminal_adverse_outcome_target(df, data_as_of_date=DATA_AS_OF)
    assert out.loc[0, TERMINAL_ADVERSE_OUTCOME_LABEL_COL] == 0
    assert out.loc[0, LABEL_STATUS_TERMINAL_ADVERSE_OUTCOME_COL] == LABEL_STATUS_NON_DEFAULT


def test_terminal_target_not_yet_matured_is_censored_regardless_of_status():
    df = pd.DataFrame(
        [_base_row(loan_id=1, status_category="charged_off", maturity_date=pd.Timestamp("2020-01-01"))]
    )
    out = compute_terminal_adverse_outcome_target(df, data_as_of_date=DATA_AS_OF)
    assert pd.isna(out.loc[0, TERMINAL_ADVERSE_OUTCOME_LABEL_COL])
    assert out.loc[0, LABEL_STATUS_TERMINAL_ADVERSE_OUTCOME_COL] == LABEL_STATUS_CENSORED


def test_terminal_target_past_maturity_but_still_open_is_censored_not_fabricated():
    """The empirically-found edge case: a handful of real loans are still
    'Current'/'Late' even after their scheduled maturity date has passed
    (likely restructured). Their terminal outcome is genuinely unknown and
    must stay censored, never coded either way."""
    df = pd.DataFrame(
        [_base_row(loan_id=1, status_category="current_performing", maturity_date=pd.Timestamp("2018-01-01"))]
    )
    out = compute_terminal_adverse_outcome_target(df, data_as_of_date=DATA_AS_OF)
    assert pd.isna(out.loc[0, TERMINAL_ADVERSE_OUTCOME_LABEL_COL])
    assert out.loc[0, LABEL_STATUS_TERMINAL_ADVERSE_OUTCOME_COL] == LABEL_STATUS_CENSORED


# --- 12-month proxy target ---

def test_12m_proxy_default_exactly_on_window_boundary_is_inclusive():
    origination = pd.Timestamp("2015-01-01")
    window_end = origination + pd.DateOffset(months=12)  # 2016-01-01
    df = pd.DataFrame(
        [
            _base_row(
                loan_id=1,
                loan_status="Charged Off",
                status_category="charged_off",
                origination_date=origination,
                last_pymnt_date=window_end,
            )
        ]
    )
    out = compute_default_12m_proxy_target(df, ASSUMPTIONS, data_as_of_date=DATA_AS_OF)
    assert out.loc[0, DEFAULT_12M_PROXY_LABEL_COL] == 1
    assert out.loc[0, LABEL_STATUS_12M_PROXY_COL] == LABEL_STATUS_DEFAULT


def test_12m_proxy_default_one_day_after_window_is_non_default_not_default():
    origination = pd.Timestamp("2015-01-01")
    window_end = origination + pd.DateOffset(months=12)
    df = pd.DataFrame(
        [
            _base_row(
                loan_id=1,
                loan_status="Charged Off",
                status_category="charged_off",
                origination_date=origination,
                last_pymnt_date=window_end + pd.Timedelta(days=1),
            )
        ]
    )
    out = compute_default_12m_proxy_target(df, ASSUMPTIONS, data_as_of_date=DATA_AS_OF)
    assert out.loc[0, DEFAULT_12M_PROXY_LABEL_COL] == 0
    assert out.loc[0, LABEL_STATUS_12M_PROXY_COL] == LABEL_STATUS_NON_DEFAULT


def test_12m_proxy_zero_payment_charge_off_falls_back_to_origination_date():
    """A charged-off loan with no last_pymnt_date (no payment ever made)
    must use origination_date as the proxy event date, not be silently
    dropped or treated as censored."""
    origination = pd.Timestamp("2015-01-01")
    df = pd.DataFrame(
        [
            _base_row(
                loan_id=1,
                loan_status="Charged Off",
                status_category="charged_off",
                origination_date=origination,
                last_pymnt_date=pd.NaT,
            )
        ]
    )
    out = compute_default_12m_proxy_target(df, ASSUMPTIONS, data_as_of_date=DATA_AS_OF)
    assert out.loc[0, DEFAULT_12M_PROXY_LABEL_COL] == 1


def test_12m_proxy_insufficient_followup_is_censored_not_zero():
    origination = pd.Timestamp("2019-01-01")  # window end 2020-01-01, after DATA_AS_OF
    df = pd.DataFrame(
        [
            _base_row(
                loan_id=1,
                loan_status="Current",
                status_category="current_performing",
                origination_date=origination,
                last_pymnt_date=pd.NaT,
            )
        ]
    )
    out = compute_default_12m_proxy_target(df, ASSUMPTIONS, data_as_of_date=DATA_AS_OF)
    assert pd.isna(out.loc[0, DEFAULT_12M_PROXY_LABEL_COL])
    assert out.loc[0, LABEL_STATUS_12M_PROXY_COL] == LABEL_STATUS_CENSORED


def test_12m_proxy_never_treats_delinquency_alone_as_default():
    origination = pd.Timestamp("2015-01-01")
    df = pd.DataFrame(
        [
            _base_row(
                loan_id=1,
                loan_status="In Grace Period",
                status_category="delinquent",
                origination_date=origination,
                last_pymnt_date=origination,
            )
        ]
    )
    out = compute_default_12m_proxy_target(df, ASSUMPTIONS, data_as_of_date=DATA_AS_OF)
    assert out.loc[0, DEFAULT_12M_PROXY_LABEL_COL] == 0
    assert out.loc[0, LABEL_STATUS_12M_PROXY_COL] == LABEL_STATUS_NON_DEFAULT
