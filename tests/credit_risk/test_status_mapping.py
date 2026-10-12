import pandas as pd
import pytest

from src.credit_risk.status_mapping import (
    StatusMappingError,
    load_status_mapping_table,
    map_loan_status,
    map_status_series,
)

REAL_STATUS_VALUES = [
    "Current",
    "In Grace Period",
    "Late (16-30 days)",
    "Late (31-120 days)",
    "Default",
    "Charged Off",
    "Fully Paid",
    "Does not meet the credit policy. Status:Fully Paid",
    "Does not meet the credit policy. Status:Charged Off",
]


def test_every_real_status_value_is_mapped():
    table = load_status_mapping_table()
    assert set(table.keys()) == set(REAL_STATUS_VALUES)


def test_map_loan_status_returns_correct_category():
    assert map_loan_status("Charged Off").category == "charged_off"
    assert map_loan_status("Fully Paid").category == "fully_paid"
    assert map_loan_status("Current").category == "current_performing"


def test_credit_policy_flag_does_not_change_category():
    """'Does not meet the credit policy.' statuses must map to the SAME
    outcome category as their unprefixed counterpart -- underwriting-policy
    vintage must never be conflated with repayment outcome severity."""
    assert (
        map_loan_status("Does not meet the credit policy. Status:Fully Paid").category
        == map_loan_status("Fully Paid").category
    )
    assert (
        map_loan_status("Does not meet the credit policy. Status:Charged Off").category
        == map_loan_status("Charged Off").category
    )
    assert map_loan_status("Does not meet the credit policy. Status:Fully Paid").meets_credit_policy is False
    assert map_loan_status("Fully Paid").meets_credit_policy is True


def test_lesser_delinquency_is_not_severe_category():
    assert map_loan_status("In Grace Period").category == "delinquent"
    assert map_loan_status("Late (16-30 days)").category == "delinquent"
    assert map_loan_status("Late (31-120 days)").category == "severe_delinquency_or_default"


def test_unmapped_status_raises_rather_than_silently_dropping():
    with pytest.raises(StatusMappingError):
        map_loan_status("Some New Status LendingClub Never Published Before")


def test_map_status_series_raises_on_any_unmapped_value():
    series = pd.Series(["Current", "Fully Paid", "Totally Unknown Status"])
    with pytest.raises(StatusMappingError):
        map_status_series(series)


def test_map_status_series_vectorized_matches_scalar():
    series = pd.Series(REAL_STATUS_VALUES)
    result = map_status_series(series)
    for status, category in zip(REAL_STATUS_VALUES, result["status_category"]):
        assert category == map_loan_status(status).category
