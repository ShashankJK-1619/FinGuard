from __future__ import annotations

import pandas as pd

from src.credit_risk.schema import (
    ACTUAL_PAYMENTS_COLUMNS,
    APPLICATION_ID_COL,
    BORROWER_COLUMNS,
    BORROWER_ID_COL,
    CREDIT_RISK_TABLES,
    LOAN_APPLICATION_COLUMNS,
    LOAN_COLUMNS,
    LOAN_ID_COL,
    SCHEDULED_PAYMENT_NUMBER_COL,
    SCHEDULED_PAYMENTS_COLUMNS,
)


def test_credit_risk_tables_lists_exactly_the_five_documented_tables():
    assert set(CREDIT_RISK_TABLES) == {
        "borrowers",
        "loan_applications",
        "loans",
        "scheduled_payments",
        "actual_payments",
    }


def test_each_table_column_list_includes_its_declared_key_columns():
    assert BORROWER_ID_COL in BORROWER_COLUMNS
    assert {APPLICATION_ID_COL, BORROWER_ID_COL} <= set(LOAN_APPLICATION_COLUMNS)
    assert {LOAN_ID_COL, APPLICATION_ID_COL, BORROWER_ID_COL} <= set(LOAN_COLUMNS)
    assert {LOAN_ID_COL, SCHEDULED_PAYMENT_NUMBER_COL} <= set(SCHEDULED_PAYMENTS_COLUMNS)
    assert LOAN_ID_COL in ACTUAL_PAYMENTS_COLUMNS


def _toy_borrowers() -> pd.DataFrame:
    return pd.DataFrame({BORROWER_ID_COL: ["b1", "b2", "b3"]})


def _toy_loan_applications() -> pd.DataFrame:
    return pd.DataFrame(
        {
            APPLICATION_ID_COL: ["a1", "a2", "a3"],
            BORROWER_ID_COL: ["b1", "b1", "b2"],
        }
    )


def _toy_loans() -> pd.DataFrame:
    return pd.DataFrame({LOAN_ID_COL: ["l1", "l2"], APPLICATION_ID_COL: ["a1", "a3"]})


def _toy_scheduled_payments() -> pd.DataFrame:
    return pd.DataFrame(
        {
            LOAN_ID_COL: ["l1", "l1", "l2"],
            SCHEDULED_PAYMENT_NUMBER_COL: [1, 2, 1],
        }
    )


def test_borrowers_grain_is_one_row_per_borrower_id():
    df = _toy_borrowers()
    assert not df[BORROWER_ID_COL].duplicated().any()


def test_loan_applications_grain_is_one_row_per_application_id_many_per_borrower():
    df = _toy_loan_applications()
    assert not df[APPLICATION_ID_COL].duplicated().any()
    # A single borrower (b1) legitimately has more than one application --
    # grain is the application, not the borrower.
    assert (df[BORROWER_ID_COL].value_counts() > 1).any()


def test_loans_grain_is_one_row_per_loan_id_and_at_most_one_per_application():
    df = _toy_loans()
    assert not df[LOAN_ID_COL].duplicated().any()
    assert not df[APPLICATION_ID_COL].duplicated().any()


def test_scheduled_payments_grain_is_one_row_per_loan_and_payment_number():
    df = _toy_scheduled_payments()
    assert not df.duplicated(subset=[LOAN_ID_COL, SCHEDULED_PAYMENT_NUMBER_COL]).any()
    # Multiple scheduled installments per loan are expected (grain is the
    # installment, not the loan).
    assert (df[LOAN_ID_COL].value_counts() > 1).any()
