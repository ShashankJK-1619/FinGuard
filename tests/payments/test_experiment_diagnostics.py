from __future__ import annotations

import pandas as pd
import pytest

from src.payments.config import ExperimentAssumptions
from src.payments.experiment_assignment import assign_experiment_arms
from src.payments.experiment_diagnostics import (
    ExperimentDiagnosticError,
    check_no_crossover,
    compute_pretreatment_covariate_balance,
    compute_srm_check,
)
from src.payments.experiment_effects import apply_synthetic_treatment_effect
from src.payments.duckdb_views import connect_with_payments_views, register_experiment_sessions_view
from src.payments.generate_sessions import generate_payment_sessions_and_attempts
from src.payments.schema import ASSIGNMENT_COL, SESSION_ID_COL


def _experiment_df(toy_real_transactions, toy_assumptions, experiment_assumptions):
    sessions_df, attempts_df = generate_payment_sessions_and_attempts(toy_real_transactions, toy_assumptions, seed=7)
    assigned = assign_experiment_arms(sessions_df, experiment_assumptions)
    return apply_synthetic_treatment_effect(assigned, experiment_assumptions), sessions_df, attempts_df


def test_check_no_crossover_passes_on_clean_data(toy_real_transactions, toy_assumptions, toy_experiment_assumptions):
    experiment_df, _, _ = _experiment_df(toy_real_transactions, toy_assumptions, toy_experiment_assumptions)
    result = check_no_crossover(experiment_df)
    assert result["n_sessions"] == len(experiment_df)
    assert result["n_unique_session_ids"] == len(experiment_df)


def test_check_no_crossover_fails_loudly_on_duplicate_session_id(
    toy_real_transactions, toy_assumptions, toy_experiment_assumptions
):
    experiment_df, _, _ = _experiment_df(toy_real_transactions, toy_assumptions, toy_experiment_assumptions)
    duplicated = pd.concat([experiment_df, experiment_df.iloc[[0]]], ignore_index=True)
    with pytest.raises(ExperimentDiagnosticError):
        check_no_crossover(duplicated)


def test_check_no_crossover_fails_on_same_unit_in_both_arms(
    toy_real_transactions, toy_assumptions, toy_experiment_assumptions
):
    """Direct regression guard for the specific failure mode called out in
    the task: the same randomized unit appearing in both treatment and
    control."""
    experiment_df, _, _ = _experiment_df(toy_real_transactions, toy_assumptions, toy_experiment_assumptions)
    crossover_row = experiment_df.iloc[[0]].copy()
    crossover_row[ASSIGNMENT_COL] = "control" if crossover_row[ASSIGNMENT_COL].iloc[0] == "treatment" else "treatment"
    contaminated = pd.concat([experiment_df, crossover_row], ignore_index=True)
    with pytest.raises(ExperimentDiagnosticError):
        check_no_crossover(contaminated)


def test_srm_check_does_not_flag_balanced_assignment(toy_real_transactions, toy_assumptions, toy_experiment_assumptions):
    experiment_df, _, _ = _experiment_df(toy_real_transactions, toy_assumptions, toy_experiment_assumptions)
    result = compute_srm_check(experiment_df, toy_experiment_assumptions)
    assert result["n_total"] == len(experiment_df)
    assert result["n_treatment"] + result["n_control"] == result["n_total"]
    assert not result["srm_flag"]


def test_srm_check_flags_badly_skewed_assignment(toy_real_transactions, toy_assumptions, toy_experiment_assumptions):
    experiment_df, _, _ = _experiment_df(toy_real_transactions, toy_assumptions, toy_experiment_assumptions)
    skewed = experiment_df.copy()
    # Force a severe, obviously-broken allocation (90% treatment against a
    # target of 50%) to exercise the "fail loudly" branch.
    n = len(skewed)
    skewed[ASSIGNMENT_COL] = ["treatment"] * int(n * 0.9) + ["control"] * (n - int(n * 0.9))
    result = compute_srm_check(skewed, toy_experiment_assumptions)
    assert result["srm_flag"]
    assert result["srm_p_value"] < toy_experiment_assumptions.alpha


def test_covariate_balance_query_returns_both_arms(toy_real_transactions, toy_assumptions, toy_experiment_assumptions):
    experiment_df, sessions_df, attempts_df = _experiment_df(
        toy_real_transactions, toy_assumptions, toy_experiment_assumptions
    )
    con = connect_with_payments_views(sessions_df, attempts_df)
    register_experiment_sessions_view(con, experiment_df)
    balance_df = compute_pretreatment_covariate_balance(con)
    con.close()
    assert set(balance_df[ASSIGNMENT_COL]) == {"control", "treatment"}
    assert (balance_df["n"] > 0).all()
