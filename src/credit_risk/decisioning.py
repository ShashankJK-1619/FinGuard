"""Module C, Milestone 4, Section E: illustrative three-way decisioning
framework.

Neutral action labels (`approve_like` / `manual_review` / `decline_like`)
are used deliberately instead of "approve"/"decline" to avoid overstating
this project's underwriting authority: this is NOT LendingClub's (or any
real lender's) actual approve/decline policy, and originating a loan in
this historical dataset was already a fact by the time any row reaches
this framework -- nothing here changes a real credit decision. Thresholds
are configurable and, per this module's Section H/selection-bias
constraint, FIT (if fit at all) on development (train) data only -- this
module itself only applies already-chosen thresholds, it does not fit
them (fitting on train is the caller's responsibility, mirroring
risk_bands.py's split between `fit_risk_band_boundaries` and
`assign_risk_band`).

Two ordered thresholds split the probability of `terminal_adverse_outcome`
(see schema.py -- NOT a 12-month PD) into three bands:
  - score <  review_threshold                      -> approve_like
  - review_threshold <= score < decline_threshold   -> manual_review
  - score >= decline_threshold                      -> decline_like
"""
from __future__ import annotations

import numpy as np

DECISIONING_DISCLAIMER = (
    "ILLUSTRATIVE three-way decisioning policy only -- NOT a real "
    "LendingClub (or any lender's) approve/decline policy. Thresholds are "
    "configurable, development-data-only, and labeled with neutral "
    "approve_like/manual_review/decline_like actions precisely because "
    "this project has no actual underwriting authority."
)

DECISION_LABELS = ("approve_like", "manual_review", "decline_like")


class DecisionThresholdError(ValueError):
    pass


def classify_decisions(
    y_prob: np.ndarray, review_threshold: float, decline_threshold: float
) -> np.ndarray:
    """Three-action decisioning, mirroring `src/models/cost_model.py`'s
    `classify_actions` pattern. Requires
    `0 <= review_threshold < decline_threshold <= 1` -- a review band that
    has collapsed to zero width, or an inverted threshold pair, is not a
    valid three-action policy and is rejected rather than silently
    degenerating to two actions."""
    if not (0 <= review_threshold < decline_threshold <= 1):
        raise DecisionThresholdError(
            "Require 0 <= review_threshold < decline_threshold <= 1; got "
            f"review_threshold={review_threshold}, decline_threshold={decline_threshold}"
        )
    y_prob = np.asarray(y_prob)
    decisions = np.full(y_prob.shape, "approve_like", dtype=object)
    decisions[(y_prob >= review_threshold) & (y_prob < decline_threshold)] = "manual_review"
    decisions[y_prob >= decline_threshold] = "decline_like"
    return decisions
