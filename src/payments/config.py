"""Module A, Milestone 1: config loaders for the synthetic-generation
assumptions and the guardrail-rule thresholds.

Same pattern as `src.models.cost_model.load_cost_assumptions`: every
illustrative numeric assumption lives in a JSON file under `configs/`, not
hardcoded in this module, so it is visibly swappable and never confused
with a measured value. See docs/payments_data_model.md for what each
field means and why it's illustrative.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from src.data.config import PROJECT_ROOT

DEFAULT_SYNTHETIC_ASSUMPTIONS_FILE = PROJECT_ROOT / "configs" / "payments_synthetic_assumptions.json"
DEFAULT_GUARDRAIL_RULES_FILE = PROJECT_ROOT / "configs" / "payments_guardrail_rules.json"


class PaymentsConfigError(ValueError):
    pass


@dataclass(frozen=True)
class SyntheticAssumptions:
    random_seed: int
    abandonment_rate: float
    # decline_before_success_probs[k] = P(a real-linked session has exactly
    # k declined attempts before its one real, authorized attempt).
    decline_before_success_probs: tuple[float, ...]
    # abandonment_attempt_count_probs[k] = P(a synthetic-abandonment session
    # makes exactly (k + 1) declined attempts before giving up, k = 0-indexed).
    abandonment_attempt_count_probs: tuple[float, ...]
    decline_reason_weights: dict[str, float] = field(default_factory=dict)
    session_lead_time_seconds_min: float = 1.0
    session_lead_time_seconds_max: float = 120.0
    retry_gap_seconds_min: float = 5.0
    retry_gap_seconds_max: float = 600.0

    def __post_init__(self) -> None:
        if not (0 <= self.abandonment_rate <= 1):
            raise PaymentsConfigError(f"abandonment_rate must be in [0, 1]; got {self.abandonment_rate}")
        for name, probs in (
            ("decline_before_success_probs", self.decline_before_success_probs),
            ("abandonment_attempt_count_probs", self.abandonment_attempt_count_probs),
        ):
            if abs(sum(probs) - 1.0) > 1e-9:
                raise PaymentsConfigError(f"{name} must sum to 1.0; got {sum(probs)}")
        if abs(sum(self.decline_reason_weights.values()) - 1.0) > 1e-9:
            raise PaymentsConfigError(
                f"decline_reason_weights must sum to 1.0; got {sum(self.decline_reason_weights.values())}"
            )
        if not (0 < self.session_lead_time_seconds_min <= self.session_lead_time_seconds_max):
            raise PaymentsConfigError("session_lead_time_seconds_min must be > 0 and <= the max")
        if not (0 < self.retry_gap_seconds_min <= self.retry_gap_seconds_max):
            raise PaymentsConfigError("retry_gap_seconds_min must be > 0 and <= the max")

    @property
    def decline_reason_codes(self) -> list[str]:
        return list(self.decline_reason_weights.keys())

    @property
    def decline_reason_probs(self) -> list[float]:
        return list(self.decline_reason_weights.values())


def load_synthetic_assumptions(config_file: Path = DEFAULT_SYNTHETIC_ASSUMPTIONS_FILE) -> SyntheticAssumptions:
    with open(config_file) as f:
        cfg = json.load(f)
    return SyntheticAssumptions(
        random_seed=int(cfg["random_seed"]),
        abandonment_rate=float(cfg["abandonment_rate"]),
        decline_before_success_probs=tuple(float(p) for p in cfg["decline_before_success_probs"]),
        abandonment_attempt_count_probs=tuple(float(p) for p in cfg["abandonment_attempt_count_probs"]),
        decline_reason_weights={k: float(v) for k, v in cfg["decline_reason_weights"].items()},
        session_lead_time_seconds_min=float(cfg["session_lead_time_seconds_min"]),
        session_lead_time_seconds_max=float(cfg["session_lead_time_seconds_max"]),
        retry_gap_seconds_min=float(cfg["retry_gap_seconds_min"]),
        retry_gap_seconds_max=float(cfg["retry_gap_seconds_max"]),
    )


@dataclass(frozen=True)
class GuardrailRules:
    velocity_attempt_threshold: int
    high_amount_percentile: float

    def __post_init__(self) -> None:
        if self.velocity_attempt_threshold < 1:
            raise PaymentsConfigError(f"velocity_attempt_threshold must be >= 1; got {self.velocity_attempt_threshold}")
        if not (0 < self.high_amount_percentile < 1):
            raise PaymentsConfigError(f"high_amount_percentile must be in (0, 1); got {self.high_amount_percentile}")


def load_guardrail_rules(config_file: Path = DEFAULT_GUARDRAIL_RULES_FILE) -> GuardrailRules:
    with open(config_file) as f:
        cfg = json.load(f)
    return GuardrailRules(
        velocity_attempt_threshold=int(cfg["velocity_attempt_threshold"]),
        high_amount_percentile=float(cfg["high_amount_percentile"]),
    )
