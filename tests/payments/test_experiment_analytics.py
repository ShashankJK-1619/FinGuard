from __future__ import annotations

import pytest

from src.payments.duckdb_views import connect_with_payments_views, register_experiment_sessions_view
from src.payments.experiment_assignment import assign_experiment_arms
from src.payments.experiment_effects import apply_synthetic_treatment_effect
from src.payments.experiment_guardrails import GUARDRAIL_METRICS, compute_guardrail_metrics, evaluate_decision_rule
from src.payments.experiment_metrics import compute_primary_and_secondary_metrics
from src.payments.experiment_power import compute_achieved_power, compute_mde, compute_mde_table
from src.payments.experiment_segments import SEGMENT_DIMENSIONS, ExperimentSegmentError, compute_segment_heterogeneity
from src.payments.experiment_stats import compute_two_proportion_test
from src.payments.generate_sessions import generate_payment_sessions_and_attempts
from src.payments.schema import ASSIGNMENT_COL, ASSIGNMENT_CONTROL, ASSIGNMENT_TREATMENT


@pytest.fixture
def experiment_con(toy_real_transactions, toy_assumptions, toy_experiment_assumptions):
    sessions_df, attempts_df = generate_payment_sessions_and_attempts(toy_real_transactions, toy_assumptions, seed=7)
    assigned = assign_experiment_arms(sessions_df, toy_experiment_assumptions)
    experiment_df = apply_synthetic_treatment_effect(assigned, toy_experiment_assumptions)
    con = connect_with_payments_views(sessions_df, attempts_df)
    register_experiment_sessions_view(con, experiment_df)
    yield con, sessions_df, attempts_df, experiment_df
    con.close()


# --- D: metric denominators ---


def test_primary_and_secondary_metrics_denominators(experiment_con):
    con, sessions_df, attempts_df, experiment_df = experiment_con
    metrics_df = compute_primary_and_secondary_metrics(con)

    assert set(metrics_df[ASSIGNMENT_COL]) == {ASSIGNMENT_CONTROL, ASSIGNMENT_TREATMENT}
    assert metrics_df["n_sessions"].sum() == len(experiment_df)

    for _, row in metrics_df.iterrows():
        arm_df = experiment_df[experiment_df[ASSIGNMENT_COL] == row[ASSIGNMENT_COL]]
        assert row["n_sessions"] == len(arm_df)
        assert row["n_completed"] == (arm_df["experiment_final_outcome"] == "authorized").sum()
        assert row["payment_completion_rate"] == pytest.approx(row["n_completed"] / row["n_sessions"])
        assert 0.0 <= row["retry_rate"] <= 1.0
        assert 0.0 <= row["retry_recovery_rate"] <= 1.0
        assert 0.0 <= row["first_attempt_authorization_rate"] <= 1.0
        assert row["attempts_per_completed_session"] >= 1.0


# --- F: two-proportion test / CI on a KNOWN toy example (hand-computable) ---


def test_two_proportion_test_matches_known_toy_example():
    # control: 100/1000 = 0.10, treatment: 150/1000 = 0.15 -- a clear,
    # large, easily-significant uplift with round numbers to sanity-check
    # against by hand.
    result = compute_two_proportion_test(
        n_control=1000, n_completed_control=100, n_treatment=1000, n_completed_treatment=150, alpha=0.05
    )
    assert result["control_rate"] == pytest.approx(0.10)
    assert result["treatment_rate"] == pytest.approx(0.15)
    assert result["absolute_uplift"] == pytest.approx(0.05)
    assert result["relative_uplift"] == pytest.approx(0.5)
    assert result["is_statistically_significant"]
    assert result["p_value"] < 0.05
    assert result["ci_low"] < result["absolute_uplift"] < result["ci_high"]
    assert result["ci_low"] > 0  # CI should exclude 0 given how large/clear this effect is


def test_two_proportion_test_is_not_significant_when_rates_are_identical():
    result = compute_two_proportion_test(
        n_control=500, n_completed_control=250, n_treatment=500, n_completed_treatment=250, alpha=0.05
    )
    assert result["absolute_uplift"] == pytest.approx(0.0)
    assert result["p_value"] == pytest.approx(1.0, abs=1e-6)
    assert not result["is_statistically_significant"]
    assert result["ci_low"] < 0 < result["ci_high"]


def test_two_proportion_test_rejects_empty_arm():
    with pytest.raises(ValueError):
        compute_two_proportion_test(0, 0, 100, 50, 0.05)


# --- G: power / MDE ---


def test_mde_shrinks_as_sample_size_grows():
    df = compute_mde_table(baseline_rate=0.85, alpha=0.05, power=0.8, sample_sizes_per_arm=(1000, 10000, 100000))
    assert df["mde_absolute"].is_monotonic_decreasing
    assert (df["mde_absolute"] > 0).all()


def test_mde_single_value_matches_table():
    mde = compute_mde(baseline_rate=0.85, n_per_arm=10000, alpha=0.05, power=0.8)
    df = compute_mde_table(baseline_rate=0.85, alpha=0.05, power=0.8, sample_sizes_per_arm=(10000,))
    assert mde == pytest.approx(df["mde_absolute"].iloc[0])


def test_achieved_power_is_high_for_large_clear_effect():
    power = compute_achieved_power(n_per_arm=10000, baseline_rate=0.85, observed_treatment_rate=0.90, alpha=0.05)
    assert 0.0 <= power <= 1.0
    assert power > 0.9


# --- H: guardrail decision rule — both branches ---


def test_decision_rule_ships_when_primary_wins_and_guardrails_clean(toy_experiment_assumptions):
    primary = compute_two_proportion_test(
        n_control=10000, n_completed_control=8500, n_treatment=10000, n_completed_treatment=8700, alpha=0.05
    )
    guardrail_df = _clean_guardrail_df()
    decision = evaluate_decision_rule(primary, guardrail_df, toy_experiment_assumptions)
    assert decision["decision"] == "ship"
    assert decision["primary_significant_and_positive"]
    assert decision["practical_uplift_clears_threshold"]
    assert not decision["any_guardrail_breach"]
    assert decision["reasons"] == []


def test_decision_rule_blocks_ship_on_guardrail_breach_even_with_winning_primary(toy_experiment_assumptions):
    """Demonstrates the core requirement of Sec H: a treatment must not be
    declared a winner on completion-rate improvement alone if it breaches
    a fraud/risk guardrail."""
    primary = compute_two_proportion_test(
        n_control=10000, n_completed_control=8500, n_treatment=10000, n_completed_treatment=8700, alpha=0.05
    )
    guardrail_df = _clean_guardrail_df()
    # Breach fraud_rate: control 0.02 -> treatment 0.05, diff 0.03 > tolerance.
    guardrail_df.loc[guardrail_df[ASSIGNMENT_COL] == ASSIGNMENT_TREATMENT, "fraud_rate"] = 0.05
    decision = evaluate_decision_rule(primary, guardrail_df, toy_experiment_assumptions)
    assert decision["decision"] == "no-ship"
    assert decision["any_guardrail_breach"]
    breach = next(g for g in decision["guardrail_breaches"] if g["metric"] == "fraud_rate")
    assert breach["breached"]
    assert len(decision["reasons"]) >= 1


def test_decision_rule_blocks_ship_when_uplift_is_below_practical_threshold(toy_experiment_assumptions):
    # Statistically significant (huge n) but a tiny uplift below
    # min_practical_uplift_abs (0.01 in the fixture).
    primary = compute_two_proportion_test(
        n_control=200000, n_completed_control=170000, n_treatment=200000, n_completed_treatment=170100, alpha=0.05
    )
    guardrail_df = _clean_guardrail_df()
    decision = evaluate_decision_rule(primary, guardrail_df, toy_experiment_assumptions)
    assert decision["practical_uplift_clears_threshold"] is False
    assert decision["decision"] == "no-ship"


def _clean_guardrail_df():
    import pandas as pd

    return pd.DataFrame(
        [
            {
                "assignment": ASSIGNMENT_CONTROL,
                "fraud_rate": 0.02,
                "fraudulent_completed_session_rate": 0.02,
                "legitimate_decline_proxy_rate": 0.04,
                "excessive_retry_rate": 0.05,
            },
            {
                "assignment": ASSIGNMENT_TREATMENT,
                "fraud_rate": 0.02,
                "fraudulent_completed_session_rate": 0.02,
                "legitimate_decline_proxy_rate": 0.04,
                "excessive_retry_rate": 0.05,
            },
        ]
    )


def test_guardrail_metrics_cover_all_declared_metrics_and_arms(experiment_con, toy_experiment_assumptions):
    con, _, _, _ = experiment_con
    guardrail_df = compute_guardrail_metrics(con, toy_experiment_assumptions)
    assert set(guardrail_df[ASSIGNMENT_COL]) == {ASSIGNMENT_CONTROL, ASSIGNMENT_TREATMENT}
    for metric in GUARDRAIL_METRICS:
        assert metric in guardrail_df.columns
        assert guardrail_df[metric].between(0, 1).all()


# --- I: segment heterogeneity / subgroup aggregation ---


def test_segment_heterogeneity_rejects_disallowed_dimension(experiment_con, toy_experiment_assumptions):
    con, _, _, _ = experiment_con
    with pytest.raises(ExperimentSegmentError):
        compute_segment_heterogeneity(con, "not_a_real_column", toy_experiment_assumptions)


def test_segment_heterogeneity_covers_each_pre_specified_dimension(experiment_con, toy_experiment_assumptions):
    con, _, _, experiment_df = experiment_con
    for dimension in SEGMENT_DIMENSIONS:
        segment_df = compute_segment_heterogeneity(con, dimension, toy_experiment_assumptions)
        assert (segment_df["segment_dimension"] == dimension).all()
        # Every row must have both arms represented with non-zero counts
        # (rows with an empty arm are dropped, not divided-by-zero).
        assert (segment_df["n_control"] > 0).all()
        assert (segment_df["n_treatment"] > 0).all()
        # Sanity: total segment-level n across all values should not
        # exceed the full dataset's size.
        assert segment_df["n_control"].sum() + segment_df["n_treatment"].sum() <= len(experiment_df)
