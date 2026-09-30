"""Milestone 2: assemble the transparent baseline modeling table.

Grain: one row per TransactionID (train_transaction only — test is never
loaded in this milestone, so there is no way for it to leak in, and it is
never used as labeled evaluation data).

Explicitly excluded from the baseline: all C*, D*, M*, V*, and id_01..id_38
columns. C*/D*/V* are excluded because Milestone 1's leakage review left
their computation window (point-in-time vs. global) unresolved. M* and
raw id_* values were never requested by the Milestone 2 feature list and
are deliberately left out to avoid scope creep beyond what was asked (see
docs/milestone2_feature_design.md) — this is a decision, not an oversight.

Run: python -m src.features.build_baseline_dataset
Output: data/processed/baseline_features_v1.parquet
"""
from __future__ import annotations

import sys

import duckdb
import pandas as pd

from src.data.config import Paths, load_paths
from src.data.loaders import get_connection, raw_files_present, register_raw_views
from src.features.amount_time_features import add_amount_features, add_time_features
from src.features.categorical_features import RAW_CATEGORY_COLUMNS, add_email_domain_match
from src.features.identity_features import add_has_identity_record, add_missingness_indicators
from src.features.schema import normalize_identity_columns
from src.features.velocity_features import add_expanding_frequency, add_time_since_last

# Frequency-encoding keys: chosen for plausible "familiarity/velocity"
# interpretation (account/location/email/device identifiers). ProductCD is
# deliberately excluded here — it only has 5 values, so an expanding count
# of it would mostly just track elapsed time/dataset growth, not a
# meaningful behavioral signal.
FREQUENCY_KEYS = ["card1", "addr1", "P_emaildomain", "DeviceInfo"]

# Recency keys: kept to a small, clearly-justified set (account-like,
# location-like) rather than one per frequency key, to keep the baseline
# transparent rather than exhaustive.
RECENCY_KEYS = ["card1", "addr1"]

TRANSACTION_SELECT_COLUMNS = (
    ["TransactionID", "isFraud", "TransactionDT", "TransactionAmt"] + RAW_CATEGORY_COLUMNS
)


def load_raw_frames(con: duckdb.DuckDBPyConnection):
    txn_cols = ", ".join(f'"{c}"' for c in TRANSACTION_SELECT_COLUMNS if c not in ("DeviceType", "DeviceInfo"))
    txn = con.execute(f"SELECT {txn_cols} FROM train_transaction").fetchdf()

    identity = con.execute(
        "SELECT TransactionID, DeviceType, DeviceInfo FROM train_identity"
    ).fetchdf()
    identity = normalize_identity_columns(identity)
    return txn, identity


def build_baseline_table(txn: pd.DataFrame, identity: pd.DataFrame) -> pd.DataFrame:
    df = txn.merge(identity, on="TransactionID", how="left")

    df = add_has_identity_record(df, identity["TransactionID"])
    df = add_amount_features(df)
    df = add_time_features(df)
    df = add_email_domain_match(df)
    df = add_missingness_indicators(df)

    for key in FREQUENCY_KEYS:
        df = add_expanding_frequency(df, key_col=key)
    for key in RECENCY_KEYS:
        df = add_time_since_last(df, key_col=key)

    return df


def main() -> int:
    paths: Paths = load_paths()
    present = raw_files_present(paths)
    required = ["train_transaction", "train_identity"]
    if not all(present[k] for k in required):
        missing = [k for k in required if not present[k]]
        print("Missing raw files, cannot build baseline dataset:", missing, file=sys.stderr)
        return 1

    con = get_connection()
    register_raw_views(con, paths)

    txn, identity = load_raw_frames(con)
    df = build_baseline_table(txn, identity)

    paths.processed_dir.mkdir(parents=True, exist_ok=True)
    out_path = paths.processed_dir / "baseline_features_v1.parquet"

    out_con = get_connection()
    out_con.register("baseline_df", df)
    out_con.execute(f"COPY baseline_df TO '{out_path}' (FORMAT PARQUET)")

    print(f"Wrote {out_path} ({len(df):,} rows, {len(df.columns)} columns)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
