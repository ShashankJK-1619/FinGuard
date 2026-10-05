"""Module A, Milestone 2: deterministic synthetic experiment assignment.

See docs/moduleA_milestone2_experimentation.md Section B for the full
randomization-unit rationale. Summary: the unit of randomization is the
`payment_session`, not the individual `payment_attempt` and not the
customer.

- Randomizing individual ATTEMPTS would be inappropriate: attempts within
  a session are not independent trials, they are a sequence of retries of
  the SAME checkout journey. The treatment (a revised retry/checkout
  experience) is a property of that one journey's UI/flow, so splitting
  attempts 1..N of a single session across both arms would mean a single
  customer session experiences a mix of treatment and control mid-journey
  — an incoherent, uninterpretable manipulation, and it would contaminate
  any comparison of session-level outcomes.
- CUSTOMER-level assignment would be preferable in a real system (it
  prevents the same person from ever being exposed to both arms, and
  avoids a customer "learning" the treatment experience in one session and
  carrying that learning into a later control session — a classic
  interference/contamination channel). It is NOT implemented here because
  Module A's real-field scope (docs/payments_data_model.md Sec 2a)
  deliberately excludes the masked fields (card1/2/3/5, addr*, dist*,
  id_*) that would be needed to construct even a weak customer proxy, and
  IEEE-CIS has no real, stable customer identifier among the documented
  fields Module A uses. Session-level assignment is the pragmatic,
  defensible choice given that constraint — stated as a limitation, not
  elided.
- Session-level contamination risk still exists in this design (the same
  real-world customer could appear in two different sessions assigned to
  different arms), but is bounded by the dataset's structure: IEEE-CIS
  rows are independent realized transactions with no session/customer
  linkage key available to Module A, so this risk cannot be measured
  directly here either — see the design doc's Limitations section.

Assignment itself must be a pure, deterministic function of
(session_id, seed) only — NOT of row position, and NOT of any
post-treatment/outcome field (attempt_count, final_outcome, isFraud,
etc. are never read here) — so it is reproducible, stable under
reordering, and free of post-treatment information by construction.
"""
from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd

from src.payments.config import ExperimentAssumptions
from src.payments.schema import ASSIGNMENT_COL, ASSIGNMENT_CONTROL, ASSIGNMENT_TREATMENT, SESSION_ID_COL

ASSIGNMENT_SALT = "moduleA-milestone2-assignment"


def unit_hash_uniform(unit_id: str, seed: int, salt: str) -> float:
    """Deterministic pseudo-uniform draw in [0, 1) for `unit_id`, a pure
    function of (unit_id, seed, salt) only. Using a cryptographic hash
    (rather than, say, a numpy Generator seeded once and advanced per row)
    means each unit's draw depends ONLY on its own id — never on row
    position, dataset size, or any other unit's draw — so assignment is
    stable even if sessions are added, removed, or reordered between runs.
    `salt` namespaces independent uses of this same draw mechanism (e.g.
    experiment_assignment.py vs. experiment_effects.py) so their random
    streams never collide or correlate.
    """
    digest = hashlib.sha256(f"{salt}:{seed}:{unit_id}".encode("utf-8")).hexdigest()
    # First 15 hex digits (60 bits) as a uniform integer is far more
    # precision than any probability threshold used in this module needs.
    as_int = int(digest[:15], 16)
    return as_int / float(16**15)


def assign_experiment_arms(sessions_df: pd.DataFrame, assumptions: ExperimentAssumptions) -> pd.DataFrame:
    """Returns a copy of `sessions_df` with one new column, `assignment`
    ('control' | 'treatment'). Depends only on `session_id` and
    `assumptions.experiment_seed` / `treatment_allocation_fraction` — no
    other column of `sessions_df` is read, so this cannot leak
    post-treatment information into assignment by construction."""
    out = sessions_df.copy()
    draws = out[SESSION_ID_COL].astype(str).map(
        lambda sid: unit_hash_uniform(sid, assumptions.experiment_seed, ASSIGNMENT_SALT)
    )
    out[ASSIGNMENT_COL] = np.where(
        draws.to_numpy() < assumptions.treatment_allocation_fraction, ASSIGNMENT_TREATMENT, ASSIGNMENT_CONTROL
    )
    return out
