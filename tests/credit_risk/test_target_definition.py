from __future__ import annotations

import pandas as pd
import pytest

from src.credit_risk.config import TargetDefinitionAssumptions
from src.credit_risk.schema import (
    DEFAULT_LABEL_COL,
    LABEL_STATUS_CENSORED,
    LABEL_STATUS_COL,
    LABEL_STATUS_DEFAULT,
    LABEL_STATUS_NON_DEFAULT,
)
from src.credit_risk.target_definition import TargetDefinitionError, compute_default_12m_labels


@pytest.fixture
def assumptions() -> TargetDefinitionAssumptions:
    return TargetDefinitionAssumptions(
        outcome_window_months=12,
        default_event_statuses=("Charged Off", "Default"),
        delinquency_event_statuses=("Late (31-120 days)",),
    )


def _toy_loan(origination_date, data_as_of_date, default_event_date=None, delinquency_event_date=None, loan_id="l1"):
    return {
        "loan_id": loan_id,
        "origination_date": pd.Timestamp(origination_date),
        "data_as_of_date": pd.Timestamp(data_as_of_date),
        "default_event_date": pd.Timestamp(default_event_date) if default_event_date else pd.NaT,
        "delinquency_event_date": pd.Timestamp(delinquency_event_date) if delinquency_event_date else pd.NaT,
    }


def test_default_within_window_is_labeled_default(assumptions):
    df = pd.DataFrame([_toy_loan("2020-01-01", "2020-12-01", default_event_date="2020-08-01")])
    out = compute_default_12m_labels(df, assumptions)
    assert out.loc[0, DEFAULT_LABEL_COL] == 1
    assert out.loc[0, LABEL_STATUS_COL] == LABEL_STATUS_DEFAULT


def test_default_exactly_at_window_boundary_is_inclusive_default(assumptions):
    df = pd.DataFrame([_toy_loan("2020-01-01", "2021-01-01", default_event_date="2021-01-01")])
    out = compute_default_12m_labels(df, assumptions)
    assert out.loc[0, DEFAULT_LABEL_COL] == 1
    assert out.loc[0, LABEL_STATUS_COL] == LABEL_STATUS_DEFAULT


def test_default_after_window_with_sufficient_followup_is_non_default(assumptions):
    df = pd.DataFrame([_toy_loan("2020-01-01", "2021-06-01", default_event_date="2021-03-01")])
    out = compute_default_12m_labels(df, assumptions)
    assert out.loc[0, DEFAULT_LABEL_COL] == 0
    assert out.loc[0, LABEL_STATUS_COL] == LABEL_STATUS_NON_DEFAULT


def test_no_default_with_sufficient_followup_is_non_default(assumptions):
    df = pd.DataFrame([_toy_loan("2020-01-01", "2021-01-01")])
    out = compute_default_12m_labels(df, assumptions)
    assert out.loc[0, DEFAULT_LABEL_COL] == 0
    assert out.loc[0, LABEL_STATUS_COL] == LABEL_STATUS_NON_DEFAULT


def test_no_default_with_insufficient_followup_is_censored_not_coded_zero(assumptions):
    """Core requirement of Sec D: a loan that hasn't defaulted yet but also
    hasn't been observed long enough must never be silently coded as a
    confirmed non-default."""
    df = pd.DataFrame([_toy_loan("2020-01-01", "2020-06-01")])
    out = compute_default_12m_labels(df, assumptions)
    assert pd.isna(out.loc[0, DEFAULT_LABEL_COL])
    assert out.loc[0, LABEL_STATUS_COL] == LABEL_STATUS_CENSORED


def test_delinquency_event_alone_does_not_trigger_default(assumptions):
    """Delinquency and default are tracked as separate severities -- a
    lesser delinquency event must never, by itself, flip default_12m."""
    df = pd.DataFrame(
        [_toy_loan("2020-01-01", "2021-01-01", delinquency_event_date="2020-05-01")]
    )
    out = compute_default_12m_labels(df, assumptions)
    assert out.loc[0, DEFAULT_LABEL_COL] == 0
    assert out.loc[0, LABEL_STATUS_COL] == LABEL_STATUS_NON_DEFAULT


def test_rejects_data_as_of_date_before_origination_date(assumptions):
    df = pd.DataFrame([_toy_loan("2020-06-01", "2020-01-01")])
    with pytest.raises(TargetDefinitionError):
        compute_default_12m_labels(df, assumptions)


def test_rejects_missing_required_column(assumptions):
    df = pd.DataFrame([{"loan_id": "l1", "origination_date": pd.Timestamp("2020-01-01")}])
    with pytest.raises(TargetDefinitionError):
        compute_default_12m_labels(df, assumptions)


def test_multiple_loans_labeled_independently(assumptions):
    df = pd.DataFrame(
        [
            _toy_loan("2020-01-01", "2020-06-01", loan_id="censored"),
            _toy_loan("2020-01-01", "2020-08-01", default_event_date="2020-08-01", loan_id="defaulted"),
            _toy_loan("2020-01-01", "2021-01-01", loan_id="survived"),
        ]
    )
    out = compute_default_12m_labels(df, assumptions).set_index("loan_id")
    assert out.loc["censored", LABEL_STATUS_COL] == LABEL_STATUS_CENSORED
    assert out.loc["defaulted", LABEL_STATUS_COL] == LABEL_STATUS_DEFAULT
    assert out.loc["survived", LABEL_STATUS_COL] == LABEL_STATUS_NON_DEFAULT
