import re

import pandas as pd
import pytest

from src.data.config import load_paths
from src.data.loaders import get_connection, register_raw_views
from src.features.build_baseline_dataset import build_baseline_table, load_raw_frames
from tests.conftest import requires_raw_data

MASKED_COLUMN_PATTERN = re.compile(r"^(C\d+|D\d+|M\d+|V\d+|id_\d+)$")


def _toy_txn_identity():
    txn = pd.DataFrame(
        {
            "TransactionID": [1, 2, 3],
            "isFraud": [0, 1, 0],
            "TransactionDT": [86_400, 90_000, 172_800],
            "TransactionAmt": [10.0, 20.0, 30.0],
            "card1": [111, 111, 222],
            "card2": [None, None, None],
            "card3": [None, None, None],
            "card4": ["visa", "visa", None],
            "card5": [None, None, None],
            "card6": ["debit", "debit", None],
            "addr1": [100, 100, None],
            "addr2": [None, None, None],
            "dist1": [None, None, None],
            "dist2": [None, None, None],
            "P_emaildomain": ["gmail.com", "gmail.com", None],
            "R_emaildomain": [None, "gmail.com", None],
            "ProductCD": ["W", "W", "C"],
        }
    )
    identity = pd.DataFrame(
        {
            "TransactionID": [1, 3],
            "DeviceType": ["mobile", None],
            "DeviceInfo": ["iOS Device", None],
        }
    )
    return txn, identity


def test_build_baseline_table_grain_matches_transaction_row_count():
    txn, identity = _toy_txn_identity()
    out = build_baseline_table(txn, identity)
    assert len(out) == len(txn)
    assert out["TransactionID"].is_unique


def test_build_baseline_table_never_contains_masked_columns():
    txn, identity = _toy_txn_identity()
    out = build_baseline_table(txn, identity)
    masked_cols = [c for c in out.columns if MASKED_COLUMN_PATTERN.match(c)]
    assert masked_cols == [], f"Masked columns leaked into baseline table: {masked_cols}"


def test_build_baseline_table_has_identity_record_correct():
    txn, identity = _toy_txn_identity()
    out = build_baseline_table(txn, identity).set_index("TransactionID")
    assert out.loc[1, "has_identity_record"] == 1
    assert out.loc[2, "has_identity_record"] == 0
    assert out.loc[3, "has_identity_record"] == 1


@requires_raw_data
def test_full_pipeline_on_real_data_matches_train_transaction_row_count():
    con = get_connection()
    paths = load_paths()
    register_raw_views(con, paths)

    txn, identity = load_raw_frames(con)
    expected_rows = con.execute("SELECT COUNT(*) FROM train_transaction").fetchone()[0]
    assert len(txn) == expected_rows

    out = build_baseline_table(txn, identity)
    assert len(out) == expected_rows
    assert out["TransactionID"].is_unique

    masked_cols = [c for c in out.columns if MASKED_COLUMN_PATTERN.match(c)]
    assert masked_cols == []
