"""Module C, Milestone 1: point-in-time feature-eligibility rule engine.

The core rule (see docs/credit_risk_data_model.md, "Point-in-time feature
policy"): **a field may be used as an underwriting feature only if it was
genuinely available on or before the credit decision/origination date.**
This module does not decide that for you -- the actual availability
judgment for each field lives in
`configs/credit_risk_feature_eligibility.json`, authored and reviewed by a
human. What this module *does* do is (a) load and validate that table, and
(b) run a loud, conservative second check: if a field's name matches a
well-known post-origination/leakage naming pattern (payment history,
collections, recoveries, charge-off/settlement workflow, refreshed credit
pulls) but the config marked it `available_at_application = "yes"`, that is
treated as a contradiction and raises rather than silently trusting the
config. This mirrors the "fail loudly" discipline already used by
`src/payments/config.py`'s assumption validation.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_FEATURE_ELIGIBILITY_FILE = PROJECT_ROOT / "configs" / "credit_risk_feature_eligibility.json"

AVAILABLE_YES = "yes"
AVAILABLE_NO = "no"
AVAILABLE_UNKNOWN = "unknown"
_VALID_AVAILABILITY = {AVAILABLE_YES, AVAILABLE_NO, AVAILABLE_UNKNOWN}

# Conservative keyword fragments strongly associated with post-origination /
# post-outcome fields in lending datasets. Deliberately specific (not
# single generic words like "amount") to minimize false positives against
# legitimate application-time fields.
LEAKAGE_KEYWORDS = (
    "recover",
    "total_pymnt",
    "total_rec_",
    "last_pymnt",
    "next_pymnt",
    "out_prncp",
    "collection_recovery_fee",
    "charged_off",
    "charge_off",
    "settlement",
    "hardship",
    "last_fico",
    "last_credit_pull",
    "debt_settlement",
    "pymnt_plan",
)


class FeatureEligibilityError(ValueError):
    pass


@dataclass(frozen=True)
class FeatureEligibilityEntry:
    field: str
    category: str
    available_at_application: str
    proposed_use: str
    leakage_concern: str

    def __post_init__(self) -> None:
        if self.available_at_application not in _VALID_AVAILABILITY:
            raise FeatureEligibilityError(
                f"Field '{self.field}': available_at_application must be one of "
                f"{sorted(_VALID_AVAILABILITY)}; got {self.available_at_application!r}"
            )
        lowered = self.field.lower()
        matched_keywords = [k for k in LEAKAGE_KEYWORDS if k in lowered]
        if matched_keywords and self.available_at_application == AVAILABLE_YES:
            raise FeatureEligibilityError(
                f"Field '{self.field}' is marked available_at_application='yes' but its "
                f"name matches known post-origination leakage pattern(s) {matched_keywords}. "
                "This contradiction must be resolved by a human reviewing the field's real "
                "meaning, not silently trusted -- fix the config entry."
            )

    @property
    def is_application_time_safe(self) -> bool:
        no_leakage_noted = self.leakage_concern.strip().lower() in ("", "none")
        return self.available_at_application == AVAILABLE_YES and no_leakage_noted


def load_feature_eligibility_table(
    config_file: Path = DEFAULT_FEATURE_ELIGIBILITY_FILE,
) -> list[FeatureEligibilityEntry]:
    with open(config_file) as f:
        cfg = json.load(f)
    return [FeatureEligibilityEntry(**row) for row in cfg["fields"]]


def filter_application_time_safe(
    entries: list[FeatureEligibilityEntry],
) -> list[FeatureEligibilityEntry]:
    """Returns only entries judged safe to use as underwriting features:
    explicitly available at application time AND carrying no recorded
    leakage concern. `"unknown"` and flagged-but-ambiguous entries (e.g.
    LendingClub's `grade`/`int_rate` circularity caution) are deliberately
    excluded by default -- a human must promote them explicitly later,
    they are never included by this function's default behavior."""
    return [e for e in entries if e.is_application_time_safe]
