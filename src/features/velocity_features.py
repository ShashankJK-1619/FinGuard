"""Milestone 2: causal (expanding-window) frequency and recency features.

This is the highest-leakage-risk part of Milestone 2, so the design is
deliberately narrow and heavily tested (see tests/features/test_velocity_features.py).

Core idea: a "how many times have I seen this card before" feature is only
point-in-time safe if "before" means strictly earlier in TransactionDT, not
"anywhere in the dataset." The standard, leaky mistake is a *global*
frequency encoding (count over the whole table, train+future included).
The safe version is an *expanding* aggregate computed after sorting by
time, where row i only ever sees rows 0..i-1.

Same-second ties: TransactionDT has 1-second resolution. Rows sharing the
same TransactionDT are ordered by TransactionID as a deterministic
tie-break. This is a resolution limitation, not a correctness guarantee —
it is documented, not hidden (see docs/feature_dictionary.md).
"""
from __future__ import annotations

import pandas as pd


def _time_sorted(df: pd.DataFrame, time_col: str, id_col: str) -> pd.DataFrame:
    return df.sort_values([time_col, id_col], kind="mergesort")


def add_expanding_frequency(
    df: pd.DataFrame,
    key_col: str,
    time_col: str = "TransactionDT",
    id_col: str = "TransactionID",
    out_col: str | None = None,
) -> pd.DataFrame:
    """Adds `<key_col>_prior_count`: number of STRICTLY EARLIER rows sharing
    the same key_col value.

    First occurrence of any key value gets 0 (no prior rows yet) — not NaN,
    since "zero prior sightings" is well-defined and exact, unlike recency
    (see add_time_since_last) where "no prior row" has no numeric distance.

    Rows with a null key_col value are all treated as members of one group
    (pandas groupby default drops NaN groups; we keep them explicit instead,
    since "key is missing" is itself a real, recurring case in this
    dataset, e.g. card4/card6 missingness) by filling a sentinel before
    grouping and restoring it after.
    """
    out_col = out_col or f"{key_col}_prior_count"
    ordered = _time_sorted(df, time_col, id_col)

    key = ordered[key_col]
    null_mask = key.isna()
    key_filled = key.astype("object").where(~null_mask, "__NULL_SENTINEL__")

    prior_count = ordered.groupby(key_filled).cumcount()

    result = pd.Series(prior_count.values, index=ordered.index, name=out_col)
    out = df.copy()
    out[out_col] = result.reindex(out.index)
    return out


def add_time_since_last(
    df: pd.DataFrame,
    key_col: str,
    time_col: str = "TransactionDT",
    id_col: str = "TransactionID",
    out_col: str | None = None,
) -> pd.DataFrame:
    """Adds `seconds_since_last_<key_col>`: TransactionDT minus the
    TransactionDT of the most recent STRICTLY EARLIER row with the same
    key_col value. NaN if this is the first occurrence of that key value
    (there is no prior row to measure a distance to — a missing value,
    not a zero).
    """
    out_col = out_col or f"seconds_since_last_{key_col}"
    ordered = _time_sorted(df, time_col, id_col)

    key = ordered[key_col]
    null_mask = key.isna()
    key_filled = key.astype("object").where(~null_mask, "__NULL_SENTINEL__")

    prev_time = ordered.groupby(key_filled)[time_col].shift(1)
    gap = ordered[time_col] - prev_time

    result = pd.Series(gap.values, index=ordered.index, name=out_col)
    out = df.copy()
    out[out_col] = result.reindex(out.index)
    return out
