from __future__ import annotations

import pytest

from src.credit_risk.config import (
    DEFAULT_TARGET_DEFINITION_FILE,
    CreditRiskConfigError,
    TargetDefinitionAssumptions,
    load_target_definition_assumptions,
)


def test_load_target_definition_assumptions_reads_configs_file():
    assumptions = load_target_definition_assumptions()
    assert assumptions.outcome_window_months >= 1
    assert len(assumptions.default_event_statuses) > 0
    assert DEFAULT_TARGET_DEFINITION_FILE.exists()


def _valid_kwargs() -> dict:
    return dict(
        outcome_window_months=12,
        default_event_statuses=("Charged Off", "Default"),
        delinquency_event_statuses=("Late (31-120 days)",),
        minimum_followup_days_tolerance=0,
    )


def test_rejects_non_positive_outcome_window():
    kwargs = _valid_kwargs()
    kwargs["outcome_window_months"] = 0
    with pytest.raises(CreditRiskConfigError):
        TargetDefinitionAssumptions(**kwargs)


def test_rejects_empty_default_event_statuses():
    kwargs = _valid_kwargs()
    kwargs["default_event_statuses"] = ()
    with pytest.raises(CreditRiskConfigError):
        TargetDefinitionAssumptions(**kwargs)


def test_rejects_overlap_between_default_and_delinquency_statuses():
    kwargs = _valid_kwargs()
    kwargs["delinquency_event_statuses"] = ("Charged Off",)
    with pytest.raises(CreditRiskConfigError):
        TargetDefinitionAssumptions(**kwargs)


def test_rejects_negative_followup_tolerance():
    kwargs = _valid_kwargs()
    kwargs["minimum_followup_days_tolerance"] = -1
    with pytest.raises(CreditRiskConfigError):
        TargetDefinitionAssumptions(**kwargs)
