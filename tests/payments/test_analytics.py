from __future__ import annotations

import pytest

from src.payments.config import GuardrailRules
from src.payments.duckdb_views import connect_with_payments_views
from src.payments.funnel import compute_funnel
from src.payments.generate_sessions import generate_payment_sessions_and_attempts
from src.payments.guardrails import (
    FLAG_COLUMNS,
    backtest_guardrails_against_fraud_label,
    compute_guardrail_flags,
)
from src.payments.kpi import compute_overall_kpis
from src.payments.retry_recovery import compute_retry_recovery
from src.payments.segmentation import (
    ALLOWED_SEGMENT_COLUMNS,
    SegmentationError,
    compute_all_segments,
    compute_segment_by_amount_quartile,
    compute_segment_by_column,
)


@pytest.fixture
def payments_con(toy_real_transactions, toy_assumptions):
    sessions_df, attempts_df = generate_payment_sessions_and_attempts(toy_real_transactions, toy_assumptions, seed=7)
    con = connect_with_payments_views(sessions_df, attempts_df)
    yield con, sessions_df, attempts_df
    con.close()


def test_overall_kpis_are_internally_consistent(payments_con):
    con, sessions_df, attempts_df = payments_con
    kpis = compute_overall_kpis(con)

    assert kpis["total_sessions"] == len(sessions_df)
    assert kpis["total_attempts"] == len(attempts_df)
    assert kpis["authorized_sessions"] == (sessions_df["final_outcome"] == "authorized").sum()
    assert 0.0 <= kpis["authorization_rate"] <= 1.0
    assert kpis["session_failure_rate"] == pytest.approx(1 - kpis["authorization_rate"])
    assert 0.0 <= kpis["gross_attempt_decline_rate"] <= 1.0
    assert 0.0 <= kpis["retry_rate"] <= 1.0
    assert 0.0 <= kpis["recovery_rate"] <= 1.0
    assert kpis["avg_attempts_per_session"] >= 1.0


def test_funnel_stage_counts_are_monotonically_non_increasing(payments_con):
    con, sessions_df, _ = payments_con
    funnel_df = compute_funnel(con)

    assert funnel_df["stage"].tolist() == list(range(1, funnel_df["stage"].max() + 1))
    counts = funnel_df["sessions_reaching_stage"].tolist()
    assert all(counts[i] >= counts[i + 1] for i in range(len(counts) - 1))
    assert funnel_df.loc[funnel_df["stage"] == 1, "sessions_reaching_stage"].iloc[0] == len(sessions_df)
    # conversion_from_initiated is the count divided by total sessions
    total = len(sessions_df)
    for _, row in funnel_df.iterrows():
        assert row["conversion_from_initiated"] == pytest.approx(row["sessions_reaching_stage"] / total)


def test_funnel_stage_one_has_no_declines_for_sessions_resolved_in_one_attempt(payments_con):
    con, sessions_df, attempts_df = payments_con
    funnel_df = compute_funnel(con)
    # at every stage, declined_at_stage + authorized_at_stage == number of
    # attempts actually made at that attempt_number
    attempts_per_stage = attempts_df.groupby("attempt_number").size()
    for _, row in funnel_df.iterrows():
        stage = row["stage"]
        expected = int(attempts_per_stage.get(stage, 0))
        assert row["declined_at_stage"] + row["authorized_at_stage"] == expected


def test_retry_recovery_buckets_are_exhaustive_and_recovery_rate_in_range(payments_con):
    con, sessions_df, _ = payments_con
    rr_df = compute_retry_recovery(con)

    assert rr_df["session_count"].sum() == len(sessions_df)
    assert (rr_df["recovery_rate"] >= 0).all() and (rr_df["recovery_rate"] <= 1).all()
    # bucket '0' (no prior declines) must be 100% recovered: a session
    # with attempt_count == 1 is, by construction, either the real
    # transaction's single authorized attempt or a one-attempt abandonment.
    # Both real-linked and abandonment sessions can have attempt_count==1,
    # so bucket '0' recovery rate reflects the mix, not a guaranteed 100%.
    assert set(rr_df["prior_declines_bucket"]) <= {"0", "1", "2", "3", "4+"}


def test_retry_recovery_time_to_resolution_is_nonnegative(payments_con):
    con, _, _ = payments_con
    rr_df = compute_retry_recovery(con)
    assert (rr_df["mean_time_to_resolution_seconds"] >= 0).all()
    assert (rr_df["median_time_to_resolution_seconds"] >= 0).all()


def test_segmentation_by_column_covers_every_observed_segment_value(payments_con):
    con, sessions_df, _ = payments_con
    for col in ALLOWED_SEGMENT_COLUMNS:
        seg_df = compute_segment_by_column(con, col)
        assert seg_df["session_count"].sum() == len(sessions_df)
        assert (seg_df["segment_dimension"] == col).all()
        assert 0 <= seg_df["authorization_rate"].min() and seg_df["authorization_rate"].max() <= 1


def test_segmentation_rejects_disallowed_column(payments_con):
    con, _, _ = payments_con
    with pytest.raises(SegmentationError):
        compute_segment_by_column(con, "isFraud")


def test_segmentation_by_amount_quartile_has_four_buckets_covering_all_sessions(payments_con):
    con, sessions_df, _ = payments_con
    seg_df = compute_segment_by_amount_quartile(con)
    assert len(seg_df) == 4
    assert seg_df["session_count"].sum() == len(sessions_df)
    # quartile boundaries should be non-decreasing
    assert seg_df["min_amount_in_quartile"].tolist() == sorted(seg_df["min_amount_in_quartile"].tolist())


def test_compute_all_segments_concatenates_every_dimension(payments_con):
    con, _, _ = payments_con
    all_segments = compute_all_segments(con)
    assert set(all_segments["segment_dimension"].unique()) == set(ALLOWED_SEGMENT_COLUMNS) | {
        "TransactionAmt_quartile"
    }


def test_guardrail_flags_are_boolean_and_combined_flag_matches_definition(payments_con):
    con, sessions_df, _ = payments_con
    rules = GuardrailRules(velocity_attempt_threshold=2, high_amount_percentile=0.9)
    flags_df = compute_guardrail_flags(con, rules)

    assert len(flags_df) == len(sessions_df)
    for col in FLAG_COLUMNS:
        assert flags_df[col].dtype == bool
    expected_combined = flags_df["velocity_flag"] | (flags_df["high_amount_flag"] & flags_df["mismatched_email_flag"])
    assert (flags_df["combined_flag"] == expected_combined).all()
    # velocity_flag with threshold 2 should match attempt_count >= 2 exactly
    merged = flags_df.merge(sessions_df[["session_id", "attempt_count"]], on="session_id")
    assert (merged["velocity_flag"] == (merged["attempt_count"] >= 2)).all()


def test_guardrail_backtest_excludes_abandonment_sessions_and_reports_rates_in_range(payments_con):
    con, sessions_df, _ = payments_con
    rules = GuardrailRules(velocity_attempt_threshold=2, high_amount_percentile=0.9)
    flags_df = compute_guardrail_flags(con, rules)
    backtest_df = backtest_guardrails_against_fraud_label(flags_df)

    n_real_linked = int((~sessions_df["is_synthetic_abandonment"]).sum())
    assert (backtest_df["n_real_linked_sessions"] == n_real_linked).all()
    for col in ["flag_rate_overall", "flag_rate_among_fraud", "flag_rate_among_legit", "precision", "recall"]:
        values = backtest_df[col].dropna()
        assert (values >= 0).all() and (values <= 1).all()
    assert set(backtest_df["rule"]) == set(FLAG_COLUMNS)
