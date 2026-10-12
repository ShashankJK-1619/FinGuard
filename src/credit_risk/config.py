"""Configuration-driven assumptions for Module C's default-definition logic.

Mirrors the pattern already established in `src/payments/config.py`
(Module A): every illustrative assumption lives in a JSON config file with
its own `_disclaimer`, never hardcoded in logic, and is validated loudly
via a frozen dataclass's `__post_init__`. No dataset has been downloaded
for Module C yet -- this config only defines the *rule* for turning a
standardized loan-performance table into a `default_12m` label; it does
not itself read or depend on any real file.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TARGET_DEFINITION_FILE = PROJECT_ROOT / "configs" / "credit_risk_target_definition.json"


class CreditRiskConfigError(ValueError):
    pass


@dataclass(frozen=True)
class TargetDefinitionAssumptions:
    """See docs/credit_risk_target_definition.md for the full rationale.

    `default_event_statuses` / `delinquency_event_statuses` are the
    illustrative, documented status vocabulary expected from the
    recommended candidate dataset (see docs/credit_risk_dataset_comparison.md)
    -- they are provisional until Module C Milestone 2 actually downloads
    the file and confirms its exact status vocabulary against this list.
    """

    outcome_window_months: int
    default_event_statuses: tuple[str, ...]
    delinquency_event_statuses: tuple[str, ...]
    minimum_followup_days_tolerance: int = 0

    def __post_init__(self) -> None:
        if self.outcome_window_months < 1:
            raise CreditRiskConfigError(
                f"outcome_window_months must be >= 1; got {self.outcome_window_months}"
            )
        if not self.default_event_statuses:
            raise CreditRiskConfigError("default_event_statuses must not be empty")
        if not all(isinstance(s, str) and s for s in self.default_event_statuses):
            raise CreditRiskConfigError("default_event_statuses must contain non-empty strings")
        if not all(isinstance(s, str) and s for s in self.delinquency_event_statuses):
            raise CreditRiskConfigError("delinquency_event_statuses must contain non-empty strings")
        overlap = set(self.default_event_statuses) & set(self.delinquency_event_statuses)
        if overlap:
            raise CreditRiskConfigError(
                "default_event_statuses and delinquency_event_statuses must be disjoint -- "
                f"default and delinquency are tracked as separate severities, got overlap: {overlap}"
            )
        if self.minimum_followup_days_tolerance < 0:
            raise CreditRiskConfigError(
                f"minimum_followup_days_tolerance must be >= 0; got {self.minimum_followup_days_tolerance}"
            )


def load_target_definition_assumptions(
    config_file: Path = DEFAULT_TARGET_DEFINITION_FILE,
) -> TargetDefinitionAssumptions:
    with open(config_file) as f:
        cfg = json.load(f)
    return TargetDefinitionAssumptions(
        outcome_window_months=cfg["outcome_window_months"],
        default_event_statuses=tuple(cfg["default_event_statuses"]),
        delinquency_event_statuses=tuple(cfg["delinquency_event_statuses"]),
        minimum_followup_days_tolerance=cfg.get("minimum_followup_days_tolerance", 0),
    )
