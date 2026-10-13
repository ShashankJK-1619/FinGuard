"""Tests for Module C, Milestone 3, Section J:
src/credit_risk/subgroup_diagnostics.py -- non-demographic,
loan/application-structural governance diagnostics. Explicitly verifies
the module refuses to group by anything outside the predeclared,
non-protected field list (e.g. addr_state/zip_code), which is the whole
point of this module existing."""
import numpy as np
import pandas as pd
import pytest

from src.credit_risk.subgroup_diagnostics import (
    NON_PROTECTED_SUBGROUP_FIELDS,
    SubgroupDiagnosticsError,
    compute_subgroup_diagnostics,
    row_missingness_rate,
)


def test_predeclared_fields_are_exactly_loan_application_structural():
    assert set(NON_PROTECTED_SUBGROUP_FIELDS) == {
        "home_ownership", "verification_status", "application_type", "purpose",
    }
    # Explicitly excluded, documented geographic protected-class proxies.
    assert "addr_state" not in NON_PROTECTED_SUBGROUP_FIELDS
    assert "zip_code" not in NON_PROTECTED_SUBGROUP_FIELDS


def test_rejects_non_predeclared_group_column():
    df = pd.DataFrame({"addr_state": ["CA", "NY"], "home_ownership": ["RENT", "OWN"]})
    missingness = pd.Series([0.0, 0.0])
    with pytest.raises(SubgroupDiagnosticsError):
        compute_subgroup_diagnostics(df, "addr_state", np.array([0, 1]), np.array([0.1, 0.9]), 0.5, missingness)


def test_row_missingness_rate_computes_fraction_of_numeric_columns_null():
    df = pd.DataFrame({"a": [1.0, np.nan, np.nan], "b": [np.nan, np.nan, 3.0]})
    rates = row_missingness_rate(df, ["a", "b"])
    assert rates.tolist() == [0.5, 1.0, 0.5]


def test_row_missingness_rate_rejects_empty_column_list():
    df = pd.DataFrame({"a": [1.0]})
    with pytest.raises(SubgroupDiagnosticsError):
        row_missingness_rate(df, [])


def test_compute_subgroup_diagnostics_groups_correctly_and_fills_missing_group_value():
    df = pd.DataFrame({"home_ownership": ["RENT", "RENT", None, "OWN"]})
    y_true = np.array([0, 1, 0, 1])
    y_prob = np.array([0.1, 0.6, 0.2, 0.9])
    missingness = pd.Series([0.0, 0.1, 0.2, 0.0])
    out = compute_subgroup_diagnostics(df, "home_ownership", y_true, y_prob, 0.5, missingness)
    assert set(out["group_value"]) == {"RENT", "__missing__", "OWN"}
    rent_row = out[out["group_value"] == "RENT"].iloc[0]
    assert rent_row["n"] == 2
    assert rent_row["observed_default_rate_pct"] == pytest.approx(50.0)


def test_calibration_gap_is_mean_predicted_minus_mean_observed():
    df = pd.DataFrame({"purpose": ["car", "car", "car"]})
    y_true = np.array([0, 0, 1])  # observed rate 1/3
    y_prob = np.array([0.5, 0.5, 0.5])  # mean predicted 0.5
    missingness = pd.Series([0.0, 0.0, 0.0])
    out = compute_subgroup_diagnostics(df, "purpose", y_true, y_prob, 0.5, missingness)
    row = out.iloc[0]
    expected_gap = (0.5 - 1 / 3) * 100
    assert row["calibration_gap_pct_points"] == pytest.approx(round(expected_gap, 3))


def test_fpr_and_fnr_none_when_no_actual_negatives_or_positives():
    df = pd.DataFrame({"application_type": ["Individual", "Individual"]})
    y_true = np.array([1, 1])  # no negatives at all -> FPR undefined
    y_prob = np.array([0.6, 0.9])
    missingness = pd.Series([0.0, 0.0])
    out = compute_subgroup_diagnostics(df, "application_type", y_true, y_prob, 0.5, missingness)
    row = out.iloc[0]
    assert row["false_positive_rate"] is None
    assert row["false_negative_rate"] is not None


def test_rows_sorted_by_n_descending():
    df = pd.DataFrame({"verification_status": ["Verified"] * 5 + ["Not Verified"] * 2})
    y_true = np.array([0, 1, 0, 1, 0, 0, 1])
    y_prob = np.array([0.1, 0.9, 0.2, 0.8, 0.3, 0.4, 0.6])
    missingness = pd.Series([0.0] * 7)
    out = compute_subgroup_diagnostics(df, "verification_status", y_true, y_prob, 0.5, missingness)
    assert out["n"].tolist() == sorted(out["n"].tolist(), reverse=True)
    assert out.iloc[0]["group_value"] == "Verified"
