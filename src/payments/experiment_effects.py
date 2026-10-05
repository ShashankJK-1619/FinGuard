"""Module A, Milestone 2: synthetic treatment-response mechanism.

THIS ENTIRE MODULE IS SYNTHETIC AND NO REAL EXPERIMENT OCCURRED. IEEE-CIS
transactions are historical, immutable fact — every real-linked session's
real transaction already happened and was authorized (see
docs/payments_data_model.md Sec 3) — and nothing in this module changes
that history. What it models, on top of the already-assigned synthetic
layer, is a plausible, modest "improved retry UX" treatment effect, so the
downstream statistical-analysis code in this milestone has a concrete,
end-to-end example to analyze. See
docs/moduleA_milestone2_experimentation.md Section E for the full
rationale and the explicit three-way boundary between:
  1. baseline Milestone 1 data (`src.payments.generate_sessions` — real
     transactions + the Milestone 1 synthetic session/attempt layer),
  2. synthetic randomization (`experiment_assignment.py` — WHO is in which
     arm), and
  3. this synthetic treatment-RESPONSE mechanism (WHAT changes for
     treatment-arm units, and only probabilistically).

Mechanism (deliberately single-channel and transparent):
  - Only synthetic-ABANDONMENT sessions (`is_synthetic_abandonment`) are
    eligible to be affected. Real-linked sessions already have a real,
    immutable 'authorized' `final_outcome` and are NEVER touched —
    `experiment_final_outcome` always equals `final_outcome` for them.
  - Only sessions with `attempt_count >= assumptions.eligible_min_attempt_count`
    are eligible (`retry_eligible`) — a retry-UX treatment cannot
    plausibly help someone who walked away after a single decline without
    ever retrying.
  - Only sessions assigned to the TREATMENT arm can be affected. Every
    control session's `experiment_final_outcome` equals its baseline
    `final_outcome`, unconditionally.
  - For each eligible treatment session, an independent deterministic
    Bernoulli(`assumptions.recovery_uplift_prob`) draw (using a salt
    distinct from the assignment draw, so the two random streams never
    correlate) decides whether `experiment_final_outcome` flips from
    'abandoned' to 'authorized'.

`payment_attempts` is intentionally NOT modified by this mechanism (see
Sec E for why) — it operates at the session-outcome grain only. This is a
documented simplification, not an oversight.
"""
from __future__ import annotations

import pandas as pd

from src.payments.config import ExperimentAssumptions
from src.payments.experiment_assignment import unit_hash_uniform
from src.payments.schema import (
    ASSIGNMENT_COL,
    ASSIGNMENT_TREATMENT,
    EXPERIMENT_OUTCOME_COL,
    FINAL_OUTCOME_AUTHORIZED,
    RETRY_ELIGIBLE_COL,
    SESSION_ID_COL,
    TREATMENT_EFFECT_APPLIED_COL,
)

EFFECT_SALT = "moduleA-milestone2-treatment-effect"


def apply_synthetic_treatment_effect(
    assigned_sessions_df: pd.DataFrame, assumptions: ExperimentAssumptions
) -> pd.DataFrame:
    """`assigned_sessions_df` must already have the `assignment` column
    (output of `assign_experiment_arms`). Returns a copy with three new
    columns: `retry_eligible`, `treatment_effect_applied`,
    `experiment_final_outcome`."""
    out = assigned_sessions_df.copy()
    out[RETRY_ELIGIBLE_COL] = out["attempt_count"] >= assumptions.eligible_min_attempt_count

    draws = out[SESSION_ID_COL].astype(str).map(
        lambda sid: unit_hash_uniform(sid, assumptions.experiment_seed, EFFECT_SALT)
    )
    eligible_for_effect = (
        out["is_synthetic_abandonment"] & out[RETRY_ELIGIBLE_COL] & (out[ASSIGNMENT_COL] == ASSIGNMENT_TREATMENT)
    )
    out[TREATMENT_EFFECT_APPLIED_COL] = eligible_for_effect & (draws.to_numpy() < assumptions.recovery_uplift_prob)

    out[EXPERIMENT_OUTCOME_COL] = out["final_outcome"]
    out.loc[out[TREATMENT_EFFECT_APPLIED_COL], EXPERIMENT_OUTCOME_COL] = FINAL_OUTCOME_AUTHORIZED
    return out
