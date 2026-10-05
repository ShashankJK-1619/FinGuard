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
DEFAULT_EXPERIMENT_ASSUMPTIONS_FILE = PROJECT_ROOT / "configs" / "payments_experiment_assumptions.json"


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


@dataclass(frozen=True)
class ExperimentAssumptions:
    """Module A, Milestone 2. See docs/moduleA_milestone2_experimentation.md
    for what each field means and why it's illustrative — none of these
    numbers are a measured or claimed real-world effect size; they are
    modeling choices that make the statistical-analysis code below have
    something non-trivial to analyze (Sec E)."""

    experiment_seed: int
    # Target fraction of sessions assigned to the treatment arm.
    treatment_allocation_fraction: float
    # Minimum attempt_count for a session to be "retry eligible" — the
    # synthetic treatment-response mechanism (Sec E) only ever acts on
    # sessions that actually went through at least one retry.
    eligible_min_attempt_count: int
    # P(an eligible, treatment-arm, synthetic-abandonment session's
    # experiment-observed outcome flips from 'abandoned' to 'authorized').
    recovery_uplift_prob: float
    alpha: float
    target_power: float
    # Minimum absolute uplift in the primary metric to count as
    # "practically significant" for the ship/no-ship decision rule (Sec H).
    min_practical_uplift_abs: float
    # Per-arm sample sizes used to illustrate how MDE shrinks with sample
    # size (Sec G) — not a claim about the actual dataset's size.
    mde_reference_sample_sizes: tuple[int, ...]
    # Guardrail decision-rule tolerances (Sec H): max allowed ABSOLUTE
    # increase (treatment - control) before a guardrail is considered
    # breached. All four guardrails here are "higher = worse".
    fraud_rate_guardrail_tolerance_abs: float
    fraudulent_completed_rate_guardrail_tolerance_abs: float
    legitimate_decline_proxy_guardrail_tolerance_abs: float
    excessive_retry_rate_guardrail_tolerance_abs: float
    # A session counts toward "excessive retry rate" if attempt_count
    # exceeds this.
    excessive_retry_attempt_threshold: int

    def __post_init__(self) -> None:
        if not (0 < self.treatment_allocation_fraction < 1):
            raise PaymentsConfigError(
                f"treatment_allocation_fraction must be in (0, 1); got {self.treatment_allocation_fraction}"
            )
        if self.eligible_min_attempt_count < 2:
            raise PaymentsConfigError(
                f"eligible_min_attempt_count must be >= 2 (a retry-experience treatment cannot affect a "
                f"session with no retry); got {self.eligible_min_attempt_count}"
            )
        if not (0 <= self.recovery_uplift_prob <= 1):
            raise PaymentsConfigError(f"recovery_uplift_prob must be in [0, 1]; got {self.recovery_uplift_prob}")
        if not (0 < self.alpha < 1):
            raise PaymentsConfigError(f"alpha must be in (0, 1); got {self.alpha}")
        if not (0 < self.target_power < 1):
            raise PaymentsConfigError(f"target_power must be in (0, 1); got {self.target_power}")
        if not (0 <= self.min_practical_uplift_abs <= 1):
            raise PaymentsConfigError(
                f"min_practical_uplift_abs must be in [0, 1]; got {self.min_practical_uplift_abs}"
            )
        if not self.mde_reference_sample_sizes or any(n <= 0 for n in self.mde_reference_sample_sizes):
            raise PaymentsConfigError("mde_reference_sample_sizes must be a non-empty list of positive integers")
        for name, tolerance in (
            ("fraud_rate_guardrail_tolerance_abs", self.fraud_rate_guardrail_tolerance_abs),
            ("fraudulent_completed_rate_guardrail_tolerance_abs", self.fraudulent_completed_rate_guardrail_tolerance_abs),
            ("legitimate_decline_proxy_guardrail_tolerance_abs", self.legitimate_decline_proxy_guardrail_tolerance_abs),
            ("excessive_retry_rate_guardrail_tolerance_abs", self.excessive_retry_rate_guardrail_tolerance_abs),
        ):
            if tolerance < 0:
                raise PaymentsConfigError(f"{name} must be >= 0; got {tolerance}")
        if self.excessive_retry_attempt_threshold < 1:
            raise PaymentsConfigError(
                f"excessive_retry_attempt_threshold must be >= 1; got {self.excessive_retry_attempt_threshold}"
            )


def load_experiment_assumptions(config_file: Path = DEFAULT_EXPERIMENT_ASSUMPTIONS_FILE) -> ExperimentAssumptions:
    with open(config_file) as f:
        cfg = json.load(f)
    return ExperimentAssumptions(
        experiment_seed=int(cfg["experiment_seed"]),
        treatment_allocation_fraction=float(cfg["treatment_allocation_fraction"]),
        eligible_min_attempt_count=int(cfg["eligible_min_attempt_count"]),
        recovery_uplift_prob=float(cfg["recovery_uplift_prob"]),
        alpha=float(cfg["alpha"]),
        target_power=float(cfg["target_power"]),
        min_practical_uplift_abs=float(cfg["min_practical_uplift_abs"]),
        mde_reference_sample_sizes=tuple(int(n) for n in cfg["mde_reference_sample_sizes"]),
        fraud_rate_guardrail_tolerance_abs=float(cfg["fraud_rate_guardrail_tolerance_abs"]),
        fraudulent_completed_rate_guardrail_tolerance_abs=float(cfg["fraudulent_completed_rate_guardrail_tolerance_abs"]),
        legitimate_decline_proxy_guardrail_tolerance_abs=float(cfg["legitimate_decline_proxy_guardrail_tolerance_abs"]),
        excessive_retry_rate_guardrail_tolerance_abs=float(cfg["excessive_retry_rate_guardrail_tolerance_abs"]),
        excessive_retry_attempt_threshold=int(cfg["excessive_retry_attempt_threshold"]),
    )
