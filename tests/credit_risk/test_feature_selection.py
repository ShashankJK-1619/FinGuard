"""Tests for Module C, Milestone 3, Section B: predeclared Model A / Model
B feature-set loading and cross-checking (src/credit_risk/feature_selection.py).

These tests exercise the *validation logic* against small, temp-written
config fixtures (never the real multi-hundred-field config) so each
contradiction case is explicit and fast, plus one test against the real
project configs to confirm they currently pass the same validation the
milestone script relies on.
"""
import json

import pytest

from src.credit_risk.feature_selection import (
    GRADE_CIRCULARITY_FIELDS,
    FeatureSelectionError,
    load_feature_sets,
)

ELIGIBILITY_FIELDS = {
    "loan_amnt": dict(category="loan_terms", available_at_application="yes", proposed_use="numeric", leakage_concern=""),
    "home_ownership": dict(category="borrower", available_at_application="yes", proposed_use="categorical", leakage_concern=""),
    "grade": dict(category="lender_assigned", available_at_application="yes", proposed_use="categorical", leakage_concern="circularity"),
    "sub_grade": dict(category="lender_assigned", available_at_application="yes", proposed_use="categorical", leakage_concern="circularity"),
    "int_rate": dict(category="lender_assigned", available_at_application="yes", proposed_use="numeric", leakage_concern="circularity"),
    "total_pymnt": dict(category="post_origination", available_at_application="no", proposed_use="numeric", leakage_concern="post-origination payment history"),
}


def _write_eligibility_file(tmp_path, fields=None):
    fields = fields if fields is not None else ELIGIBILITY_FIELDS
    path = tmp_path / "eligibility.json"
    path.write_text(json.dumps({"fields": [{"field": k, **v} for k, v in fields.items()]}))
    return path


def _write_feature_sets_file(
    tmp_path,
    model_a_raw_numeric=("loan_amnt",),
    model_a_categorical=("home_ownership",),
    model_b_extra_numeric=("int_rate",),
    model_b_extra_categorical=("grade", "sub_grade"),
):
    path = tmp_path / "model_feature_sets.json"
    path.write_text(
        json.dumps(
            {
                "model_a_raw_numeric_fields": list(model_a_raw_numeric),
                "model_a_derived_numeric_fields": {},
                "model_a_categorical_fields": list(model_a_categorical),
                "model_b_additional_raw_numeric_fields": list(model_b_extra_numeric),
                "model_b_additional_categorical_fields": list(model_b_extra_categorical),
            }
        )
    )
    return path


def test_load_feature_sets_valid_case(tmp_path):
    eligibility_file = _write_eligibility_file(tmp_path)
    feature_sets_file = _write_feature_sets_file(tmp_path)

    model_a, model_b = load_feature_sets(feature_sets_file, eligibility_file)

    assert "grade" not in model_a.all_fields
    assert "sub_grade" not in model_a.all_fields
    assert "int_rate" not in model_a.all_fields
    for field in GRADE_CIRCULARITY_FIELDS:
        assert field in model_b.all_fields
    # Model B is Model A plus exactly the circularity fields -- nothing else differs.
    assert set(model_b.all_fields) - set(model_a.all_fields) == set(GRADE_CIRCULARITY_FIELDS)


def test_model_a_field_not_application_time_safe_raises(tmp_path):
    eligibility_file = _write_eligibility_file(tmp_path)
    feature_sets_file = _write_feature_sets_file(tmp_path, model_a_raw_numeric=("total_pymnt",))
    with pytest.raises(FeatureSelectionError):
        load_feature_sets(feature_sets_file, eligibility_file)


def test_model_a_field_missing_from_eligibility_table_raises(tmp_path):
    eligibility_file = _write_eligibility_file(tmp_path)
    feature_sets_file = _write_feature_sets_file(tmp_path, model_a_raw_numeric=("never_reviewed_field",))
    with pytest.raises(FeatureSelectionError):
        load_feature_sets(feature_sets_file, eligibility_file)


def test_model_b_extra_fields_must_equal_grade_circularity_fields_exactly(tmp_path):
    eligibility_file = _write_eligibility_file(tmp_path)
    feature_sets_file = _write_feature_sets_file(tmp_path, model_b_extra_numeric=(), model_b_extra_categorical=("grade",))
    with pytest.raises(FeatureSelectionError):
        load_feature_sets(feature_sets_file, eligibility_file)


def test_model_b_extra_fields_cannot_overlap_model_a(tmp_path):
    eligibility_file = _write_eligibility_file(tmp_path)
    feature_sets_file = _write_feature_sets_file(
        tmp_path,
        model_a_categorical=("home_ownership", "grade"),
    )
    with pytest.raises(FeatureSelectionError):
        load_feature_sets(feature_sets_file, eligibility_file)


def test_real_project_feature_sets_config_is_currently_valid():
    """The real configs/credit_risk_model_feature_sets.json this milestone's
    orchestration script actually loads must pass its own validation."""
    model_a, model_b = load_feature_sets()
    assert model_b.name == "model_b_with_lender_grade_signal"
    assert model_a.name == "model_a_application_time_safe"
    for field in GRADE_CIRCULARITY_FIELDS:
        assert field not in model_a.all_fields
        assert field in model_b.all_fields
