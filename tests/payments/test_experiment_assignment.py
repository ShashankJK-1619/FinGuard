from __future__ import annotations

import pandas as pd

from src.payments.experiment_assignment import assign_experiment_arms
from src.payments.generate_sessions import generate_payment_sessions_and_attempts
from src.payments.schema import ASSIGNMENT_COL, ASSIGNMENT_CONTROL, ASSIGNMENT_TREATMENT, SESSION_ID_COL


def _sessions(toy_real_transactions, toy_assumptions):
    sessions_df, _ = generate_payment_sessions_and_attempts(toy_real_transactions, toy_assumptions, seed=7)
    return sessions_df


def test_assignment_is_deterministic_across_reruns(toy_real_transactions, toy_assumptions, toy_experiment_assumptions):
    sessions_df = _sessions(toy_real_transactions, toy_assumptions)
    a1 = assign_experiment_arms(sessions_df, toy_experiment_assumptions)
    a2 = assign_experiment_arms(sessions_df, toy_experiment_assumptions)
    pd.testing.assert_series_equal(a1[ASSIGNMENT_COL], a2[ASSIGNMENT_COL])


def test_assignment_is_invariant_to_row_order(toy_real_transactions, toy_assumptions, toy_experiment_assumptions):
    sessions_df = _sessions(toy_real_transactions, toy_assumptions)
    shuffled = sessions_df.sample(frac=1.0, random_state=123).reset_index(drop=True)

    assigned = assign_experiment_arms(sessions_df, toy_experiment_assumptions).set_index(SESSION_ID_COL)[
        ASSIGNMENT_COL
    ]
    assigned_shuffled = assign_experiment_arms(shuffled, toy_experiment_assumptions).set_index(SESSION_ID_COL)[
        ASSIGNMENT_COL
    ]
    pd.testing.assert_series_equal(assigned.sort_index(), assigned_shuffled.sort_index())


def test_assignment_only_depends_on_session_id_not_outcome_fields(
    toy_real_transactions, toy_assumptions, toy_experiment_assumptions
):
    """No-post-treatment-information guard: mutating outcome fields
    (final_outcome, isFraud, attempt_count) for the same session_ids must
    not change assignment."""
    sessions_df = _sessions(toy_real_transactions, toy_assumptions)
    mutated = sessions_df.copy()
    mutated["final_outcome"] = "abandoned"
    mutated["isFraud"] = 1
    mutated["attempt_count"] = mutated["attempt_count"] + 10

    original_assignment = assign_experiment_arms(sessions_df, toy_experiment_assumptions)[ASSIGNMENT_COL]
    mutated_assignment = assign_experiment_arms(mutated, toy_experiment_assumptions)[ASSIGNMENT_COL]
    pd.testing.assert_series_equal(original_assignment, mutated_assignment)


def test_assignment_is_approximately_balanced(toy_real_transactions, toy_assumptions, toy_experiment_assumptions):
    sessions_df = _sessions(toy_real_transactions, toy_assumptions)
    assigned = assign_experiment_arms(sessions_df, toy_experiment_assumptions)
    n = len(assigned)
    treatment_fraction = (assigned[ASSIGNMENT_COL] == ASSIGNMENT_TREATMENT).mean()
    # Loose tolerance appropriate for a few hundred toy rows; the dedicated
    # SRM test exercises statistical rigor on the balance question.
    assert abs(treatment_fraction - toy_experiment_assumptions.treatment_allocation_fraction) < 0.15
    assert set(assigned[ASSIGNMENT_COL].unique()) <= {ASSIGNMENT_CONTROL, ASSIGNMENT_TREATMENT}
    assert n == len(sessions_df)


def test_different_seed_changes_some_assignments(toy_real_transactions, toy_assumptions, toy_experiment_assumptions):
    sessions_df = _sessions(toy_real_transactions, toy_assumptions)
    a1 = assign_experiment_arms(sessions_df, toy_experiment_assumptions)[ASSIGNMENT_COL]
    other_assumptions = toy_experiment_assumptions.__class__(
        **{**toy_experiment_assumptions.__dict__, "experiment_seed": toy_experiment_assumptions.experiment_seed + 1}
    )
    a2 = assign_experiment_arms(sessions_df, other_assumptions)[ASSIGNMENT_COL]
    assert not a1.equals(a2)
