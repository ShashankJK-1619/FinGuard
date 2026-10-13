"""Tests for Module C, Milestone 3, Section B/C support:
src/credit_risk/feature_loader.py.

Pure-function derivation logic (`_derive_emp_length_years`,
`_derive_credit_history_months`, `feature_matrix_columns`) is tested
against small constructed toy fixtures, mirroring the project's
toy-fixture-only unit-test discipline. `load_modeling_table` itself needs
a real DuckDB connection over the real CSV, so it is exercised only in the
real-data-gated smoke test at the bottom (skipped unless the raw file is
present, per tests/credit_risk/conftest.py).
"""
import pandas as pd
import pytest

from src.credit_risk.feature_loader import (
    EMP_LENGTH_YEARS_MAP,
    FeatureLoaderError,
    _derive_credit_history_months,
    _derive_emp_length_years,
    feature_matrix_columns,
    load_modeling_table,
)
from src.credit_risk.feature_selection import FeatureSet, load_feature_sets
from src.credit_risk.raw_loader import get_connection, register_loans_view

from tests.credit_risk.conftest import requires_credit_risk_raw_data


def test_derive_emp_length_years_maps_known_values():
    s = pd.Series(["< 1 year", "1 year", "10+ years", None])
    out = _derive_emp_length_years(s)
    assert out.tolist()[:3] == [0, 1, 10]
    assert pd.isna(out.iloc[3])


def test_derive_emp_length_years_unrecognized_value_raises():
    s = pd.Series(["< 1 year", "some garbage value"])
    with pytest.raises(FeatureLoaderError):
        _derive_emp_length_years(s)


def test_emp_length_years_map_covers_every_lendingclub_bucket():
    # Explicit, auditable map -- exactly 11 buckets, 0..10.
    assert sorted(EMP_LENGTH_YEARS_MAP.values()) == list(range(11))


def test_derive_credit_history_months_whole_months_between_dates():
    earliest_cr_line = pd.Series(["Jan-2010", "Mar-2010"])
    origination_date = pd.Series([pd.Timestamp("2015-01-01"), pd.Timestamp("2015-01-01")])
    out = _derive_credit_history_months(earliest_cr_line, origination_date)
    assert out.iloc[0] == 60  # exactly 5 years
    assert out.iloc[1] == 58  # 2 months short of 5 years


def test_derive_credit_history_months_unparseable_date_is_missing_not_fabricated():
    earliest_cr_line = pd.Series(["not-a-date"])
    origination_date = pd.Series([pd.Timestamp("2015-01-01")])
    out = _derive_credit_history_months(earliest_cr_line, origination_date)
    assert pd.isna(out.iloc[0])


def test_feature_matrix_columns_returns_numeric_and_categorical_separately():
    fs = FeatureSet(
        name="toy",
        raw_numeric_fields=("loan_amnt",),
        derived_numeric_fields=("fico_avg_at_application",),
        categorical_fields=("home_ownership", "purpose"),
    )
    numeric_cols, categorical_cols = feature_matrix_columns(fs)
    assert numeric_cols == ["loan_amnt", "fico_avg_at_application"]
    assert categorical_cols == ["home_ownership", "purpose"]


@requires_credit_risk_raw_data
def test_load_modeling_table_against_real_data_has_no_censored_identifiers_missing():
    con = get_connection()
    register_loans_view(con)
    model_a, _ = load_feature_sets()
    df = load_modeling_table(con, model_a, limit=2000)

    numeric_cols, categorical_cols = feature_matrix_columns(model_a)
    for col in numeric_cols + categorical_cols:
        assert col in df.columns, f"expected derived/raw feature column '{col}' in loaded table"

    # Derived features computed, not left as raw strings.
    assert pd.api.types.is_numeric_dtype(df["fico_avg_at_application"])
    assert pd.api.types.is_numeric_dtype(df["term_months"])

    # grade/sub_grade/int_rate (Model B only) must NOT silently leak into Model A's table.
    assert "grade" not in df.columns
    assert "sub_grade" not in df.columns
    assert "int_rate" not in df.columns
