"""Tests for the Module C Milestone 2 target-governance correction: the
machine-readable declaration of which target is PRIMARY (`terminal_adverse_
outcome`) vs SECONDARY (`default_12m_last_pymnt_proxy`), and the loud
validation that rejects any config contradicting that governance decision."""
from dataclasses import asdict

import pytest

from src.credit_risk.target_governance import (
    TargetGovernance,
    TargetGovernanceError,
    load_target_governance,
)

VALID_KWARGS = dict(
    primary_target_column="terminal_adverse_outcome",
    primary_label_status_column="label_status_terminal_adverse_outcome",
    primary_target_interpretation="Estimates eventual adverse credit outcome, not a 12-month PD.",
    primary_target_is_fixed_horizon_pd=False,
    primary_split_design_file="configs/credit_risk_split_design_primary_terminal.json",
    secondary_target_column="default_12m_last_pymnt_proxy",
    secondary_label_status_column="label_status_12m_proxy",
    secondary_target_interpretation="Secondary sensitivity-analysis-only; last_pymnt_d is not a default-event timestamp.",
    secondary_event_date_is_genuine_timestamp=False,
    secondary_event_date_source_column="last_pymnt_d",
    secondary_split_design_file="configs/credit_risk_split_design_secondary_proxy.json",
)


def test_load_real_target_governance_designates_terminal_as_primary():
    governance = load_target_governance()
    assert governance.primary_target_column == "terminal_adverse_outcome"
    assert governance.secondary_target_column == "default_12m_last_pymnt_proxy"
    assert governance.primary_target_is_fixed_horizon_pd is False
    assert governance.secondary_event_date_is_genuine_timestamp is False
    assert "12-month" in governance.primary_target_interpretation or "12 month" in governance.primary_target_interpretation.lower()
    assert "last_pymnt_d" in governance.secondary_target_interpretation


def test_real_governance_config_is_internally_consistent():
    # Should not raise -- the real config must pass its own validation.
    load_target_governance()


def test_rejects_same_column_as_both_primary_and_secondary():
    with pytest.raises(TargetGovernanceError):
        TargetGovernance(**{**VALID_KWARGS, "secondary_target_column": VALID_KWARGS["primary_target_column"]})


def test_rejects_primary_marked_as_fixed_horizon_pd():
    with pytest.raises(TargetGovernanceError):
        TargetGovernance(**{**VALID_KWARGS, "primary_target_is_fixed_horizon_pd": True})


def test_rejects_secondary_event_date_marked_as_genuine_timestamp():
    with pytest.raises(TargetGovernanceError):
        TargetGovernance(**{**VALID_KWARGS, "secondary_event_date_is_genuine_timestamp": True})


def test_rejects_blank_primary_interpretation():
    with pytest.raises(TargetGovernanceError):
        TargetGovernance(**{**VALID_KWARGS, "primary_target_interpretation": "   "})


def test_rejects_blank_secondary_interpretation():
    with pytest.raises(TargetGovernanceError):
        TargetGovernance(**{**VALID_KWARGS, "secondary_target_interpretation": ""})


def test_valid_governance_constructs_without_error():
    governance = TargetGovernance(**VALID_KWARGS)
    assert asdict(governance)["primary_target_column"] == "terminal_adverse_outcome"
