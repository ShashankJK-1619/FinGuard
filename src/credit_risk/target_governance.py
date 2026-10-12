"""Module C, Milestone 2 target-governance correction: a loudly-validated,
machine-readable declaration of which target is PRIMARY vs SECONDARY for
Milestone 3 modeling.

This exists because an earlier draft of Milestone 2 treated
`default_12m_last_pymnt_proxy` as the presumptive primary PD target. That
was wrong: it is a fixed 12-month-window target fed a *proxy* event date
(`last_pymnt_d`, which is NOT a default-event timestamp -- see
`src/credit_risk/target_construction.py` module docstring for the
documented one-directional bias this creates). The project correction
instead designates `terminal_adverse_outcome` -- built only from explicitly
mapped resolved terminal `loan_status` categories, requiring no event date
at all -- as primary, and demotes the proxy target to secondary,
sensitivity-analysis-only status. See `docs/decision_log.md`'s "Milestone 2
target-governance correction" entry for full rationale.

This module does not re-derive that judgment -- the actual governance
decision lives in `configs/credit_risk_target_governance.json`, authored
and reviewed by a human. What this module does is (a) load and validate
that config, and (b) run a loud, fail-fast check: if the config ever
contradicts the project's governance rules (same column marked both
primary and secondary, the primary target marked as a fixed-horizon PD,
the secondary target's proxy event date marked as a genuine timestamp, or
either interpretation left blank), it raises `TargetGovernanceError`
rather than silently trusting a bad edit. This mirrors the "fail loudly"
discipline already used by `src/credit_risk/feature_eligibility.py` and
`src/credit_risk/chronological_split.py`.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TARGET_GOVERNANCE_FILE = PROJECT_ROOT / "configs" / "credit_risk_target_governance.json"


class TargetGovernanceError(ValueError):
    pass


@dataclass(frozen=True)
class TargetGovernance:
    primary_target_column: str
    primary_label_status_column: str
    primary_target_interpretation: str
    primary_target_is_fixed_horizon_pd: bool
    primary_split_design_file: str
    secondary_target_column: str
    secondary_label_status_column: str
    secondary_target_interpretation: str
    secondary_event_date_is_genuine_timestamp: bool
    secondary_event_date_source_column: str
    secondary_split_design_file: str

    def __post_init__(self) -> None:
        if self.primary_target_column == self.secondary_target_column:
            raise TargetGovernanceError(
                "primary_target_column and secondary_target_column must not be the same "
                f"column (both are {self.primary_target_column!r}) -- the whole point of "
                "this config is to keep the primary and secondary targets distinct."
            )
        if self.primary_target_is_fixed_horizon_pd:
            raise TargetGovernanceError(
                f"primary_target_column={self.primary_target_column!r} is marked "
                "primary_target_is_fixed_horizon_pd=True, but the project's governance "
                "correction explicitly requires the primary target to be the "
                "variable-horizon terminal adverse-outcome target, not a fixed-horizon PD "
                "estimate. Fix the config -- do not silently accept this contradiction."
            )
        if self.secondary_event_date_is_genuine_timestamp:
            raise TargetGovernanceError(
                f"secondary_target_column={self.secondary_target_column!r}'s event date "
                f"({self.secondary_event_date_source_column!r}) is marked "
                "secondary_event_date_is_genuine_timestamp=True, but it is a documented "
                "proxy (e.g. last_pymnt_d is a last-payment date, not a default-event "
                "timestamp). Fix the config -- do not silently accept this contradiction."
            )
        if not self.primary_target_interpretation.strip():
            raise TargetGovernanceError(
                "primary_target_interpretation must not be blank -- the primary target's "
                "meaning (eventual adverse outcome, NOT 12-month PD) must always be stated "
                "explicitly alongside the config."
            )
        if not self.secondary_target_interpretation.strip():
            raise TargetGovernanceError(
                "secondary_target_interpretation must not be blank -- the secondary "
                "target's proxy nature and documented bias must always be stated "
                "explicitly alongside the config."
            )


def load_target_governance(
    config_file: Path = DEFAULT_TARGET_GOVERNANCE_FILE,
) -> TargetGovernance:
    with open(config_file) as f:
        cfg = json.load(f)
    cfg = {k: v for k, v in cfg.items() if not k.startswith("_")}
    return TargetGovernance(**cfg)
