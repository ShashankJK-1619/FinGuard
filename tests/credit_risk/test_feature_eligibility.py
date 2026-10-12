from __future__ import annotations

import pytest

from src.credit_risk.feature_eligibility import (
    DEFAULT_FEATURE_ELIGIBILITY_FILE,
    FeatureEligibilityEntry,
    FeatureEligibilityError,
    filter_application_time_safe,
    load_feature_eligibility_table,
)


def test_load_feature_eligibility_table_reads_configs_file():
    entries = load_feature_eligibility_table()
    assert len(entries) > 0
    assert DEFAULT_FEATURE_ELIGIBILITY_FILE.exists()


def test_known_post_origination_fields_are_excluded_from_application_time_safe_set():
    entries = load_feature_eligibility_table()
    safe_fields = {e.field for e in filter_application_time_safe(entries)}
    known_leakage_fields = {
        "loan_status",
        "total_pymnt",
        "total_rec_prncp",
        "total_rec_int",
        "last_pymnt_d",
        "last_pymnt_amnt",
        "next_pymnt_d",
        "out_prncp",
        "recoveries",
        "collection_recovery_fee",
        "hardship_flag",
        "debt_settlement_flag",
        "last_fico_range_high",
        "last_credit_pull_d",
    }
    assert known_leakage_fields.isdisjoint(safe_fields)


def test_known_application_time_fields_are_included_in_safe_set():
    entries = load_feature_eligibility_table()
    safe_fields = {e.field for e in filter_application_time_safe(entries)}
    known_safe_fields = {"annual_inc", "emp_length", "home_ownership", "purpose", "dti"}
    assert known_safe_fields <= safe_fields


def test_grade_and_int_rate_are_flagged_unknown_not_blindly_safe():
    """Regression guard for the specific circularity caution documented in
    docs/credit_risk_data_model.md: grade/sub_grade/int_rate must not be
    silently treated as safe application-time features."""
    entries = load_feature_eligibility_table()
    by_field = {e.field: e for e in entries}
    for field in ("grade", "sub_grade", "int_rate"):
        assert by_field[field].available_at_application == "unknown"
    safe_fields = {e.field for e in filter_application_time_safe(entries)}
    assert {"grade", "sub_grade", "int_rate"}.isdisjoint(safe_fields)


def test_entry_rejects_invalid_availability_value():
    with pytest.raises(FeatureEligibilityError):
        FeatureEligibilityEntry(
            field="some_field",
            category="test",
            available_at_application="maybe",
            proposed_use="n/a",
            leakage_concern="",
        )


def test_entry_rejects_leakage_keyword_marked_available_yes():
    """Direct regression guard for the task's explicit requirement:
    rejection of obvious post-origination leakage fields, even if
    mis-configured as available at application time."""
    with pytest.raises(FeatureEligibilityError):
        FeatureEligibilityEntry(
            field="total_pymnt_to_date",
            category="post_origination_payment_history",
            available_at_application="yes",
            proposed_use="n/a",
            leakage_concern="",
        )


def test_entry_with_leakage_keyword_and_available_no_does_not_raise():
    entry = FeatureEligibilityEntry(
        field="recoveries",
        category="post_origination_collections",
        available_at_application="no",
        proposed_use="exclude",
        leakage_concern="post-default collections outcome",
    )
    assert not entry.is_application_time_safe
