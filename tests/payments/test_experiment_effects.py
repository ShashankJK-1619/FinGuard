from __future__ import annotations

import pandas as pd

from src.payments.config import ExperimentAssumptions
from src.payments.experiment_assignment import assign_experiment_arms
from src.payments.experiment_effects import apply_synthetic_treatment_effect
from src.payments.generate_sessions import generate_payment_sessions_and_attempts
from src.payments.schema import (
    ASSIGNMENT_COL,
    ASSIGNMENT_CONTROL,
    ASSIGNMENT_TREATMENT,
    EXPERIMENT_OUTCOME_COL,
    FINAL_OUTCOME_ABANDONED,
    FINAL_OUTCOME_AUTHORIZED,
    RETRY_ELIGIBLE_COL,
    TREATMENT_EFFECT_APPLIED_COL,
)


def _experiment_df(toy_real_transactions, toy_assumptions, experiment_assumptions):
    sessions_df, _ = generate_payment_sessions_and_attempts(toy_real_transactions, toy_assumptions, seed=7)
    assigned = assign_experiment_arms(sessions_df, experiment_assumptions)
    return apply_synthetic_treatment_effect(assigned, experiment_assumptions)


def test_control_sessions_are_never_altered(toy_real_transactions, toy_assumptions, toy_experiment_assumptions):
    experiment_df = _experiment_df(toy_real_transactions, toy_assumptions, toy_experiment_assumptions)
    control = experiment_df[experiment_df[ASSIGNMENT_COL] == ASSIGNMENT_CONTROL]
    assert (~control[TREATMENT_EFFECT_APPLIED_COL]).all()
    assert (control[EXPERIMENT_OUTCOME_COL] == control["final_outcome"]).all()


def test_real_linked_sessions_are_never_altered_even_in_treatment(
    toy_real_transactions, toy_assumptions, toy_experiment_assumptions
):
    """Real transaction history is immutable: even treatment-arm,
    real-linked sessions must never have the effect applied."""
    experiment_df = _experiment_df(toy_real_transactions, toy_assumptions, toy_experiment_assumptions)
    real_linked = experiment_df[~experiment_df["is_synthetic_abandonment"]]
    assert (~real_linked[TREATMENT_EFFECT_APPLIED_COL]).all()
    assert (real_linked[EXPERIMENT_OUTCOME_COL] == FINAL_OUTCOME_AUTHORIZED).all()
    assert (real_linked[EXPERIMENT_OUTCOME_COL] == real_linked["final_outcome"]).all()


def test_treatment_effect_only_applies_to_retry_eligible_abandonment_sessions(
    toy_real_transactions, toy_assumptions, toy_experiment_assumptions
):
    experiment_df = _experiment_df(toy_real_transactions, toy_assumptions, toy_experiment_assumptions)
    applied = experiment_df[experiment_df[TREATMENT_EFFECT_APPLIED_COL]]
    assert (applied["is_synthetic_abandonment"]).all()
    assert (applied[RETRY_ELIGIBLE_COL]).all()
    assert (applied[ASSIGNMENT_COL] == ASSIGNMENT_TREATMENT).all()
    assert (applied["attempt_count"] >= toy_experiment_assumptions.eligible_min_attempt_count).all()
    # Every row where the effect was applied must show the flipped outcome.
    assert (applied[EXPERIMENT_OUTCOME_COL] == FINAL_OUTCOME_AUTHORIZED).all()
    # And its baseline outcome must have genuinely been 'abandoned' (not a
    # no-op flip).
    assert (applied["final_outcome"] == FINAL_OUTCOME_ABANDONED).all()


def test_retry_eligible_flag_matches_attempt_count_threshold(
    toy_real_transactions, toy_assumptions, toy_experiment_assumptions
):
    experiment_df = _experiment_df(toy_real_transactions, toy_assumptions, toy_experiment_assumptions)
    expected = experiment_df["attempt_count"] >= toy_experiment_assumptions.eligible_min_attempt_count
    pd.testing.assert_series_equal(experiment_df[RETRY_ELIGIBLE_COL], expected, check_names=False)


def test_effect_application_is_deterministic(toy_real_transactions, toy_assumptions, toy_experiment_assumptions):
    sessions_df, _ = generate_payment_sessions_and_attempts(toy_real_transactions, toy_assumptions, seed=7)
    assigned = assign_experiment_arms(sessions_df, toy_experiment_assumptions)
    e1 = apply_synthetic_treatment_effect(assigned, toy_experiment_assumptions)
    e2 = apply_synthetic_treatment_effect(assigned, toy_experiment_assumptions)
    pd.testing.assert_series_equal(e1[TREATMENT_EFFECT_APPLIED_COL], e2[TREATMENT_EFFECT_APPLIED_COL])
    pd.testing.assert_series_equal(e1[EXPERIMENT_OUTCOME_COL], e2[EXPERIMENT_OUTCOME_COL])


def test_zero_recovery_uplift_prob_applies_no_effect(toy_real_transactions, toy_assumptions, toy_experiment_assumptions):
    zero_effect_assumptions = ExperimentAssumptions(
        **{**toy_experiment_assumptions.__dict__, "recovery_uplift_prob": 0.0}
    )
    experiment_df = _experiment_df(toy_real_transactions, toy_assumptions, zero_effect_assumptions)
    assert (~experiment_df[TREATMENT_EFFECT_APPLIED_COL]).all()
    assert (experiment_df[EXPERIMENT_OUTCOME_COL] == experiment_df["final_outcome"]).all()
