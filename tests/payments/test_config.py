from __future__ import annotations

import pytest

from src.payments.config import (
    DEFAULT_GUARDRAIL_RULES_FILE,
    DEFAULT_SYNTHETIC_ASSUMPTIONS_FILE,
    GuardrailRules,
    PaymentsConfigError,
    SyntheticAssumptions,
    load_guardrail_rules,
    load_synthetic_assumptions,
)


def test_load_synthetic_assumptions_reads_configs_file():
    assumptions = load_synthetic_assumptions()
    assert 0 <= assumptions.abandonment_rate <= 1
    assert sum(assumptions.decline_before_success_probs) == pytest.approx(1.0)
    assert sum(assumptions.abandonment_attempt_count_probs) == pytest.approx(1.0)
    assert sum(assumptions.decline_reason_weights.values()) == pytest.approx(1.0)
    assert DEFAULT_SYNTHETIC_ASSUMPTIONS_FILE.exists()


def test_load_guardrail_rules_reads_configs_file():
    rules = load_guardrail_rules()
    assert rules.velocity_attempt_threshold >= 1
    assert 0 < rules.high_amount_percentile < 1
    assert DEFAULT_GUARDRAIL_RULES_FILE.exists()


def _valid_assumptions_kwargs() -> dict:
    return dict(
        random_seed=1,
        abandonment_rate=0.1,
        decline_before_success_probs=(0.5, 0.5),
        abandonment_attempt_count_probs=(1.0,),
        decline_reason_weights={"a": 1.0},
        session_lead_time_seconds_min=1.0,
        session_lead_time_seconds_max=10.0,
        retry_gap_seconds_min=1.0,
        retry_gap_seconds_max=10.0,
    )


def test_synthetic_assumptions_rejects_out_of_range_abandonment_rate():
    kwargs = _valid_assumptions_kwargs()
    kwargs["abandonment_rate"] = 1.5
    with pytest.raises(PaymentsConfigError):
        SyntheticAssumptions(**kwargs)


def test_synthetic_assumptions_rejects_probs_not_summing_to_one():
    kwargs = _valid_assumptions_kwargs()
    kwargs["decline_before_success_probs"] = (0.5, 0.4)
    with pytest.raises(PaymentsConfigError):
        SyntheticAssumptions(**kwargs)


def test_synthetic_assumptions_rejects_decline_reason_weights_not_summing_to_one():
    kwargs = _valid_assumptions_kwargs()
    kwargs["decline_reason_weights"] = {"a": 0.5, "b": 0.3}
    with pytest.raises(PaymentsConfigError):
        SyntheticAssumptions(**kwargs)


def test_synthetic_assumptions_rejects_inverted_time_windows():
    kwargs = _valid_assumptions_kwargs()
    kwargs["session_lead_time_seconds_min"] = 20.0
    kwargs["session_lead_time_seconds_max"] = 10.0
    with pytest.raises(PaymentsConfigError):
        SyntheticAssumptions(**kwargs)


def test_guardrail_rules_rejects_invalid_threshold():
    with pytest.raises(PaymentsConfigError):
        GuardrailRules(velocity_attempt_threshold=0, high_amount_percentile=0.9)


def test_guardrail_rules_rejects_invalid_percentile():
    with pytest.raises(PaymentsConfigError):
        GuardrailRules(velocity_attempt_threshold=3, high_amount_percentile=1.5)
