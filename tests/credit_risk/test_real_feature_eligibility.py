"""Module C, Milestone 2, Section F regression tests: the feature-
eligibility table now covers the REAL 151-column schema, not the
Milestone 1 ~29-entry provisional guess."""
from src.credit_risk.feature_eligibility import filter_application_time_safe, load_feature_eligibility_table

REAL_SCHEMA_COLUMN_COUNT = 151


def test_every_real_column_is_covered():
    entries = load_feature_eligibility_table()
    assert len(entries) == REAL_SCHEMA_COLUMN_COUNT
    assert len({e.field for e in entries}) == REAL_SCHEMA_COLUMN_COUNT


def test_hardship_and_settlement_fields_all_excluded():
    entries = load_feature_eligibility_table()
    by_field = {e.field: e for e in entries}
    hardship_and_settlement_fields = [f for f in by_field if "hardship" in f or "settlement" in f]
    assert len(hardship_and_settlement_fields) > 0
    for field in hardship_and_settlement_fields:
        assert by_field[field].available_at_application == "no"


def test_credit_bureau_pull_fields_are_safe_despite_outcome_sounding_names():
    """chargeoff_within_12_mths describes OTHER accounts' history from the
    bureau pull at application, not this loan's own outcome -- must be
    marked safe, not excluded by name-pattern-matching alone."""
    entries = load_feature_eligibility_table()
    by_field = {e.field: e for e in entries}
    safe_fields = {e.field for e in filter_application_time_safe(entries)}
    assert "chargeoff_within_12_mths" in safe_fields
    assert by_field["chargeoff_within_12_mths"].available_at_application == "yes"


def test_member_id_flagged_unusable_despite_being_an_identifier_column():
    """Governance correction: member_id is a pure identifier with no
    defensible predictive business role and is always 100% NULL in this
    public release -- it is excluded ('no'), not left unresolved
    ('unknown'). See docs/decision_log.md 'Milestone 2 target-governance
    correction' entry."""
    entries = load_feature_eligibility_table()
    by_field = {e.field: e for e in entries}
    assert by_field["member_id"].available_at_application == "no"


def test_known_leakage_field_count_matches_section_f_audit():
    entries = load_feature_eligibility_table()
    excluded = [e for e in entries if e.available_at_application == "no"]
    assert len(excluded) == 40
