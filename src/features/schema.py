"""Milestone 2: schema normalization for the identity table.

IEEE-CIS ships train_identity with underscore-delimited id columns
(id_01 .. id_38) and test_identity with the *same* 38 columns but
hyphen-delimited (id-01 .. id-38). This was found empirically in
Milestone 1 (see docs/decision_log.md, 2026-09-30 entry) and is not
documented anywhere by the competition host.

Milestone 2 only builds features from train, but this normalizer is
written to be safe to apply to either table so the same feature code
can be reused at scoring time in a later milestone without silently
producing all-null id_* columns on test.
"""
from __future__ import annotations

import re

import pandas as pd

_HYPHEN_ID_PATTERN = re.compile(r"^id-(\d+)$")


def normalize_identity_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rename any `id-NN` column to `id_NN`. No-op if already underscored.

    Returns a new DataFrame (does not mutate the input) with a rename map
    applied. Raises if normalization would collide with an existing column.
    """
    rename_map = {}
    for col in df.columns:
        match = _HYPHEN_ID_PATTERN.match(col)
        if match:
            new_name = f"id_{match.group(1)}"
            rename_map[col] = new_name

    collisions = set(rename_map.values()) & set(df.columns)
    if collisions:
        raise ValueError(
            f"Normalizing id-NN -> id_NN would collide with existing columns: {collisions}"
        )

    return df.rename(columns=rename_map)
