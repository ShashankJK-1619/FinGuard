from __future__ import annotations

import numpy as np
import pandas as pd

from src.payments.generate_sessions import generate_payment_sessions_and_attempts
from src.payments.schema import (
    ATTEMPT_OUTCOME_AUTHORIZED,
    ATTEMPT_OUTCOME_DECLINED,
    FINAL_OUTCOME_ABANDONED,
    FINAL_OUTCOME_AUTHORIZED,
)


def test_determinism_same_seed_gives_identical_output(toy_real_transactions, toy_assumptions):
    sessions1, attempts1 = generate_payment_sessions_and_attempts(toy_real_transactions, toy_assumptions, seed=7)
    sessions2, attempts2 = generate_payment_sessions_and_attempts(toy_real_transactions, toy_assumptions, seed=7)
    pd.testing.assert_frame_equal(sessions1, sessions2)
    pd.testing.assert_frame_equal(attempts1, attempts2)


def test_different_seed_gives_different_output(toy_real_transactions, toy_assumptions):
    sessions1, _ = generate_payment_sessions_and_attempts(toy_real_transactions, toy_assumptions, seed=1)
    sessions2, _ = generate_payment_sessions_and_attempts(toy_real_transactions, toy_assumptions, seed=2)
    assert not sessions1["attempt_count"].equals(sessions2["attempt_count"])


def test_output_is_invariant_to_input_row_order(toy_real_transactions, toy_assumptions):
    shuffled = toy_real_transactions.sample(frac=1.0, random_state=123).reset_index(drop=True)
    sessions_a, attempts_a = generate_payment_sessions_and_attempts(toy_real_transactions, toy_assumptions, seed=7)
    sessions_b, attempts_b = generate_payment_sessions_and_attempts(shuffled, toy_assumptions, seed=7)
    pd.testing.assert_frame_equal(sessions_a, sessions_b)
    pd.testing.assert_frame_equal(attempts_a, attempts_b)


def test_session_count_matches_real_count_plus_abandonment_rate(toy_real_transactions, toy_assumptions):
    sessions_df, _ = generate_payment_sessions_and_attempts(toy_real_transactions, toy_assumptions, seed=7)
    n_real = len(toy_real_transactions)
    expected_abandon = round(toy_assumptions.abandonment_rate * n_real)
    assert len(sessions_df) == n_real + expected_abandon
    assert (~sessions_df["is_synthetic_abandonment"]).sum() == n_real
    assert sessions_df["is_synthetic_abandonment"].sum() == expected_abandon


def test_every_real_transaction_id_maps_to_exactly_one_session(toy_real_transactions, toy_assumptions):
    sessions_df, _ = generate_payment_sessions_and_attempts(toy_real_transactions, toy_assumptions, seed=7)
    real_sessions = sessions_df[~sessions_df["is_synthetic_abandonment"]]
    assert sorted(real_sessions["transaction_id"].tolist()) == sorted(toy_real_transactions["TransactionID"].tolist())
    assert real_sessions["transaction_id"].is_unique


def test_abandonment_sessions_have_no_transaction_id_or_fraud_label(toy_real_transactions, toy_assumptions):
    sessions_df, _ = generate_payment_sessions_and_attempts(toy_real_transactions, toy_assumptions, seed=7)
    abandon = sessions_df[sessions_df["is_synthetic_abandonment"]]
    assert abandon["transaction_id"].isna().all()
    assert abandon["isFraud"].isna().all()
    assert (abandon["final_outcome"] == FINAL_OUTCOME_ABANDONED).all()


def test_real_linked_sessions_have_fraud_label_matching_source(toy_real_transactions, toy_assumptions):
    sessions_df, _ = generate_payment_sessions_and_attempts(toy_real_transactions, toy_assumptions, seed=7)
    real_sessions = sessions_df[~sessions_df["is_synthetic_abandonment"]].set_index("transaction_id")
    source = toy_real_transactions.set_index("TransactionID")
    merged = real_sessions.join(source[["isFraud"]], rsuffix="_source")
    assert (merged["isFraud"] == merged["isFraud_source"]).all()
    assert (real_sessions["final_outcome"] == FINAL_OUTCOME_AUTHORIZED).all()


def test_real_linked_session_devicetype_matches_source_exactly(toy_real_transactions, toy_assumptions):
    """Direct regression guard for the DeviceType-all-NaN bug: for every
    real-linked session, the session-level DeviceType must equal the
    exact DeviceType of its source transaction row (including rows where
    the source legitimately has no identity record, i.e. NaN-to-NaN), not
    just fall within the right set of values or be non-null somewhere."""
    sessions_df, _ = generate_payment_sessions_and_attempts(toy_real_transactions, toy_assumptions, seed=7)
    real_sessions = sessions_df[~sessions_df["is_synthetic_abandonment"]].set_index("transaction_id")
    source = toy_real_transactions.set_index("TransactionID")
    merged = real_sessions.join(source[["DeviceType"]], rsuffix="_source")
    # Compare with NaNs treated as equal to each other (fillna sentinel),
    # since DeviceType is legitimately NaN for transactions with no
    # train_identity record and plain `==` would mark those as mismatches.
    sentinel = "__NA__"
    lhs = merged["DeviceType"].fillna(sentinel)
    rhs = merged["DeviceType_source"].fillna(sentinel)
    assert (lhs == rhs).all()
    # Guard against the comparison being vacuously true because every
    # value collapsed to the sentinel (i.e. the bug is still present).
    assert (lhs != sentinel).any()


def test_attempt_numbers_are_contiguous_from_one_per_session(toy_real_transactions, toy_assumptions):
    sessions_df, attempts_df = generate_payment_sessions_and_attempts(toy_real_transactions, toy_assumptions, seed=7)
    for session_id, group in attempts_df.groupby("session_id"):
        numbers = sorted(group["attempt_number"].tolist())
        assert numbers == list(range(1, len(numbers) + 1))
        expected_count = sessions_df.loc[sessions_df["session_id"] == session_id, "attempt_count"].iloc[0]
        assert len(numbers) == expected_count


def test_authorized_outcome_appears_exactly_once_and_only_as_last_attempt_of_authorized_sessions(
    toy_real_transactions, toy_assumptions
):
    sessions_df, attempts_df = generate_payment_sessions_and_attempts(toy_real_transactions, toy_assumptions, seed=7)
    authorized_attempts = attempts_df[attempts_df["outcome"] == ATTEMPT_OUTCOME_AUTHORIZED]
    authorized_sessions = sessions_df[sessions_df["final_outcome"] == FINAL_OUTCOME_AUTHORIZED]
    assert set(authorized_attempts["session_id"]) == set(authorized_sessions["session_id"])
    # exactly one authorized attempt per authorized session, and it's the last attempt_number
    merged = authorized_attempts.merge(sessions_df[["session_id", "attempt_count"]], on="session_id")
    assert (merged["attempt_number"] == merged["attempt_count"]).all()
    assert authorized_attempts["session_id"].is_unique

    abandon_sessions = sessions_df[sessions_df["final_outcome"] == FINAL_OUTCOME_ABANDONED]
    abandon_attempts = attempts_df[attempts_df["session_id"].isin(abandon_sessions["session_id"])]
    assert (abandon_attempts["outcome"] == ATTEMPT_OUTCOME_DECLINED).all()


def test_decline_reason_code_populated_iff_declined(toy_real_transactions, toy_assumptions):
    _, attempts_df = generate_payment_sessions_and_attempts(toy_real_transactions, toy_assumptions, seed=7)
    declined = attempts_df[attempts_df["outcome"] == ATTEMPT_OUTCOME_DECLINED]
    authorized = attempts_df[attempts_df["outcome"] == ATTEMPT_OUTCOME_AUTHORIZED]
    assert declined["decline_reason_code"].notna().all()
    assert authorized["decline_reason_code"].isna().all()
    assert set(declined["decline_reason_code"].unique()) <= set(toy_assumptions.decline_reason_codes)


def test_attempt_offsets_are_non_decreasing_within_a_session_and_last_matches_anchor(
    toy_real_transactions, toy_assumptions
):
    sessions_df, attempts_df = generate_payment_sessions_and_attempts(toy_real_transactions, toy_assumptions, seed=7)
    source = toy_real_transactions.set_index("TransactionID")["TransactionDT"]

    for session_id, group in attempts_df.groupby("session_id"):
        ordered = group.sort_values("attempt_number")["attempt_offset_seconds"].tolist()
        assert ordered == sorted(ordered)
        assert all(x >= 0 for x in ordered)

    # Last attempt's offset equals the real TransactionDT exactly, per real-linked session.
    sess_to_txn = sessions_df.set_index("session_id")["transaction_id"].dropna().to_dict()
    for session_id, group in attempts_df.groupby("session_id"):
        if session_id not in sess_to_txn:
            continue
        txn_id = sess_to_txn[session_id]
        last_attempt = group.loc[group["attempt_number"].idxmax()]
        assert last_attempt["attempt_offset_seconds"] == source.loc[txn_id]


def test_near_zero_anchor_time_clamps_offsets_to_nonnegative_without_erroring(toy_assumptions):
    """Boundary case called out in generate_sessions.py's
    _sample_attempt_offsets docstring: a session anchored very close to
    TransactionDT == 0 combined with large sampled retry gaps could
    otherwise push an early attempt's offset negative."""
    small_df = pd.DataFrame(
        {
            "TransactionID": [1, 2],
            "TransactionDT": [0, 1],
            "TransactionAmt": [10.0, 20.0],
            "isFraud": [0, 0],
            "ProductCD": ["W", "W"],
            "card4": ["ct_a", "ct_a"],
            "card6": ["cc_a", "cc_a"],
            "P_emaildomain": ["gmail.com", "gmail.com"],
            "R_emaildomain": ["gmail.com", "gmail.com"],
            "DeviceType": ["mobile", "mobile"],
            "email_domain_match": ["MATCH", "MATCH"],
        }
    )
    # Force a high attempt count so backward gap subtraction is maximally
    # likely to need clamping.
    forced_assumptions = toy_assumptions.__class__(
        **{**toy_assumptions.__dict__, "decline_before_success_probs": (0.0, 0.0, 0.0, 0.0, 1.0)}
    )
    sessions_df, attempts_df = generate_payment_sessions_and_attempts(small_df, forced_assumptions, seed=0)
    assert (attempts_df["attempt_offset_seconds"] >= 0).all()
    assert (sessions_df["session_start_offset_seconds"] >= 0).all()


def test_abandonment_session_profile_is_cloned_from_a_real_donor_row(toy_real_transactions, toy_assumptions):
    sessions_df, _ = generate_payment_sessions_and_attempts(toy_real_transactions, toy_assumptions, seed=7)
    abandon = sessions_df[sessions_df["is_synthetic_abandonment"]]
    valid_values = {
        "ProductCD": set(toy_real_transactions["ProductCD"]),
        "card4": set(toy_real_transactions["card4"]),
        "card6": set(toy_real_transactions["card6"]),
    }
    for col, allowed in valid_values.items():
        assert set(abandon[col].unique()) <= allowed
    # DeviceType can be legitimately NaN (no identity record) for some
    # donors, but must not be NaN for EVERY abandonment session — this is
    # a direct regression guard for a bug where DeviceType was listed in
    # SESSION_COLUMNS but never actually written into the session dict,
    # silently becoming all-NaN for every session (real and abandonment).
    assert abandon["DeviceType"].notna().any()
    non_null_device_types = set(abandon["DeviceType"].dropna().unique())
    assert non_null_device_types <= set(toy_real_transactions["DeviceType"].dropna().unique())


def test_no_populated_session_column_is_silently_all_null(toy_real_transactions, toy_assumptions):
    """Regression guard: every session column that has a non-null value
    somewhere in the real source data must also have at least one
    non-null value in the generated sessions table. Catches the class of
    bug where a column is declared in SESSION_COLUMNS but never actually
    assigned when building a session row (pandas then silently fills it
    with NaN for every row)."""
    sessions_df, _ = generate_payment_sessions_and_attempts(toy_real_transactions, toy_assumptions, seed=7)
    always_nullable = {"transaction_id", "isFraud"}  # legitimately all-NaN for abandonment-only sessions
    for col in ["ProductCD", "card4", "card6", "DeviceType", "TransactionAmt", "email_domain_match"]:
        assert sessions_df[col].notna().any(), f"{col} is unexpectedly all-null in the generated sessions table"
    for col in always_nullable:
        assert sessions_df[col].notna().any()
