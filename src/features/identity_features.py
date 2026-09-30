"""Milestone 2: identity-availability and missingness features.

All functions here are row-level: they depend only on the transaction's
own columns (including whether it has a matching identity row at all), not
on any other transaction. Point-in-time safety assumption: whether a
TransactionID has an associated identity row is treated as known at
scoring time (this mirrors Milestone 1's join validation finding that the
transaction<->identity relationship is a clean one-to-zero-or-one join, not
something resolved after the fact).
"""
from __future__ import annotations

import pandas as pd

# Deliberately NOT including id_01..id_38: those columns' business meaning
# is masked (see docs/data_dictionary.md), and creating 38 per-column
# missingness indicators from them would smuggle masked-feature granularity
# into a "transparent baseline" through a side door. DeviceType/DeviceInfo
# are the identity table's documented device fields; the email domains are
# the clearest documented, non-masked identity-adjacent signal. This choice
# is deliberate — see docs/milestone2_feature_design.md.
IMPORTANT_MISSINGNESS_COLUMNS = [
    "DeviceType",
    "DeviceInfo",
    "P_emaildomain",
    "R_emaildomain",
]


def add_has_identity_record(df: pd.DataFrame, identity_ids: pd.Series) -> pd.DataFrame:
    """Adds has_identity_record: 1 if TransactionID has a matching identity row."""
    out = df.copy()
    out["has_identity_record"] = out["TransactionID"].isin(set(identity_ids)).astype(int)
    return out


def add_missingness_indicators(
    df: pd.DataFrame, columns: list[str] | None = None
) -> pd.DataFrame:
    """Adds `<col>_missing` = 1 if the source column is null, else 0.

    Only applied to `columns` that are actually present in `df` (identity
    columns won't exist on rows built before the identity merge, or in a
    table that never merged identity at all).
    """
    columns = columns if columns is not None else IMPORTANT_MISSINGNESS_COLUMNS
    out = df.copy()
    for col in columns:
        if col in out.columns:
            out[f"{col}_missing"] = out[col].isna().astype(int)
    return out
