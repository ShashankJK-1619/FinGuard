"""Leakage-safety tests for causal (expanding-window) velocity features.

These are the highest-stakes tests in Milestone 2: a bug here would mean
the baseline dataset silently leaks future information, which is exactly
the failure mode this milestone exists to prevent.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.features.velocity_features import add_expanding_frequency, add_time_since_last


def _toy_df():
    # card1=A appears at t=10 (txn1), t=20 (txn2), t=30 (txn3)
    # card1=B appears at t=15 (txn4)
    return pd.DataFrame(
        {
            "TransactionID": [1, 2, 3, 4],
            "TransactionDT": [10, 20, 30, 15],
            "card1": ["A", "A", "A", "B"],
        }
    )


def test_expanding_frequency_counts_only_strictly_earlier_rows():
    df = _toy_df()
    out = add_expanding_frequency(df, key_col="card1")
    by_id = out.set_index("TransactionID")["card1_prior_count"]
    assert by_id[1] == 0  # first A, no prior A
    assert by_id[4] == 0  # only B, no prior B
    assert by_id[2] == 1  # second A (t=20), one prior A (t=10)
    assert by_id[3] == 2  # third A (t=30), two prior A's (t=10, t=20)


def test_expanding_frequency_is_invariant_to_input_row_order():
    df = _toy_df()
    shuffled = df.sample(frac=1.0, random_state=42).reset_index(drop=True)

    out_a = add_expanding_frequency(df, key_col="card1").set_index("TransactionID")["card1_prior_count"]
    out_b = add_expanding_frequency(shuffled, key_col="card1").set_index("TransactionID")[
        "card1_prior_count"
    ]
    pd.testing.assert_series_equal(out_a.sort_index(), out_b.sort_index())


def test_expanding_frequency_mutating_a_future_row_does_not_change_past_rows():
    """The adversarial leakage check: if changing a LATER transaction's key
    value changes an EARLIER transaction's frequency feature, the feature
    is not point-in-time safe."""
    df = _toy_df()
    baseline = add_expanding_frequency(df, key_col="card1").set_index("TransactionID")[
        "card1_prior_count"
    ]

    mutated = df.copy()
    # txn3 (t=30, the LATEST row) changes its card1 from A to a brand-new value
    mutated.loc[mutated["TransactionID"] == 3, "card1"] = "Z"
    mutated_out = add_expanding_frequency(mutated, key_col="card1").set_index("TransactionID")[
        "card1_prior_count"
    ]

    # Rows strictly earlier than the mutated row (txn1, txn2, txn4) must be unaffected.
    for txn_id in (1, 2, 4):
        assert baseline[txn_id] == mutated_out[txn_id], f"txn {txn_id} changed when a future row was mutated"


def test_expanding_frequency_handles_null_keys_as_their_own_group():
    df = pd.DataFrame(
        {
            "TransactionID": [1, 2, 3],
            "TransactionDT": [10, 20, 30],
            "card4": [None, None, "visa"],
        }
    )
    out = add_expanding_frequency(df, key_col="card4")
    by_id = out.set_index("TransactionID")["card4_prior_count"]
    assert by_id[1] == 0
    assert by_id[2] == 1  # second null, one prior null
    assert by_id[3] == 0  # first non-null value


def test_time_since_last_first_occurrence_is_nan_not_zero():
    df = _toy_df()
    out = add_time_since_last(df, key_col="card1")
    by_id = out.set_index("TransactionID")["seconds_since_last_card1"]
    assert np.isnan(by_id[1])
    assert np.isnan(by_id[4])


def test_time_since_last_computes_correct_gap():
    df = _toy_df()
    out = add_time_since_last(df, key_col="card1")
    by_id = out.set_index("TransactionID")["seconds_since_last_card1"]
    assert by_id[2] == 10  # t=20 - t=10
    assert by_id[3] == 10  # t=30 - t=20


def test_time_since_last_mutating_a_future_row_does_not_change_past_rows():
    df = _toy_df()
    baseline = add_time_since_last(df, key_col="card1").set_index("TransactionID")[
        "seconds_since_last_card1"
    ]

    mutated = df.copy()
    mutated.loc[mutated["TransactionID"] == 3, "TransactionDT"] = 9_999
    mutated_out = add_time_since_last(mutated, key_col="card1").set_index("TransactionID")[
        "seconds_since_last_card1"
    ]

    for txn_id in (1, 2, 4):
        left = baseline[txn_id]
        right = mutated_out[txn_id]
        if np.isnan(left):
            assert np.isnan(right)
        else:
            assert left == right


def test_same_timestamp_ties_broken_deterministically_by_transaction_id():
    df = pd.DataFrame(
        {
            "TransactionID": [5, 2, 3],
            "TransactionDT": [100, 100, 100],
            "card1": ["A", "A", "A"],
        }
    )
    out_1 = add_expanding_frequency(df, key_col="card1")
    out_2 = add_expanding_frequency(df.sample(frac=1.0, random_state=7), key_col="card1")

    by_id_1 = out_1.set_index("TransactionID")["card1_prior_count"].sort_index()
    by_id_2 = out_2.set_index("TransactionID")["card1_prior_count"].sort_index()
    pd.testing.assert_series_equal(by_id_1, by_id_2)
    # Tie-break is by TransactionID ascending: id 2 first (count 0), then 3 (1), then 5 (2)
    assert by_id_1[2] == 0
    assert by_id_1[3] == 1
    assert by_id_1[5] == 2
