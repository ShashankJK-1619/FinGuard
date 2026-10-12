"""Module C, Milestone 2, Section C: explicit loan_status -> severity-category
mapping, built from the real, empirically-observed loan_status vocabulary
(see docs/moduleC_milestone2_real_data_validation.md Section C).

Design discipline: every real loan_status value must be mapped explicitly
in `configs/credit_risk_status_mapping.json`. `map_loan_status` raises
`StatusMappingError` on any value not present in that table rather than
silently bucketing it into "ambiguous_other" or dropping the row -- a
previously-unseen status string (e.g. from a future LendingClub vintage)
must be reviewed and added by a human, never auto-classified.

This module's `category` output is a descriptive severity bucket for
reporting only. It is intentionally decoupled from
`default_event_statuses`/`delinquency_event_statuses` in
`configs/credit_risk_target_definition.json`, which is what actually
drives default_12m_last_pymnt_proxy/terminal_adverse_outcome construction
-- see `target_construction.py` and each status entry's
`target_trigger_note`.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from src.credit_risk.schema import STATUS_CATEGORIES

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_STATUS_MAPPING_FILE = PROJECT_ROOT / "configs" / "credit_risk_status_mapping.json"


class StatusMappingError(ValueError):
    pass


@dataclass(frozen=True)
class StatusMappingEntry:
    loan_status: str
    category: str
    meets_credit_policy: bool
    rationale: str

    def __post_init__(self) -> None:
        if not self.loan_status:
            raise StatusMappingError("loan_status must not be empty")
        if self.category not in STATUS_CATEGORIES:
            raise StatusMappingError(
                f"Status '{self.loan_status}': category must be one of "
                f"{STATUS_CATEGORIES}; got {self.category!r}"
            )
        if not self.rationale.strip():
            raise StatusMappingError(
                f"Status '{self.loan_status}': rationale must not be empty -- "
                "every mapping decision must be documented, not silent."
            )


def load_status_mapping_table(
    config_file: Path = DEFAULT_STATUS_MAPPING_FILE,
) -> dict[str, StatusMappingEntry]:
    with open(config_file) as f:
        cfg = json.load(f)
    entries = {}
    for row in cfg["statuses"]:
        entry = StatusMappingEntry(
            loan_status=row["loan_status"],
            category=row["category"],
            meets_credit_policy=row["meets_credit_policy"],
            rationale=row["rationale"],
        )
        if entry.loan_status in entries:
            raise StatusMappingError(f"Duplicate mapping entry for '{entry.loan_status}'")
        entries[entry.loan_status] = entry
    return entries


def map_loan_status(
    loan_status: str, table: dict[str, StatusMappingEntry] | None = None
) -> StatusMappingEntry:
    table = table if table is not None else load_status_mapping_table()
    if loan_status not in table:
        raise StatusMappingError(
            f"Unrecognized loan_status {loan_status!r} -- not present in "
            "configs/credit_risk_status_mapping.json. A previously-unseen "
            "status must be reviewed and added explicitly, never silently "
            "collapsed into an existing category."
        )
    return table[loan_status]


def map_status_series(
    loan_status_series: pd.Series, table: dict[str, StatusMappingEntry] | None = None
) -> pd.DataFrame:
    """Vectorized form for a full loans table. Returns a DataFrame with
    columns `status_category` and `meets_credit_policy`, index-aligned to
    the input series. Raises StatusMappingError (via map_loan_status) if
    any value is unmapped -- fails loudly on the whole batch rather than
    silently NaN-ing out unmapped rows."""
    table = table if table is not None else load_status_mapping_table()
    unique_values = loan_status_series.dropna().unique()
    unmapped = [v for v in unique_values if v not in table]
    if unmapped:
        raise StatusMappingError(
            f"Unrecognized loan_status value(s) {unmapped} -- not present in "
            "configs/credit_risk_status_mapping.json. Review and add them "
            "explicitly before mapping this table."
        )
    category = loan_status_series.map(lambda s: table[s].category if pd.notna(s) else None)
    meets_policy = loan_status_series.map(
        lambda s: table[s].meets_credit_policy if pd.notna(s) else None
    )
    return pd.DataFrame({"status_category": category, "meets_credit_policy": meets_policy})
