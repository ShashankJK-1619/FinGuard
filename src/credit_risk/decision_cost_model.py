"""Module C, Milestone 4, Sections F-G: illustrative decision-cost framework
and threshold/assumption sensitivity analysis.

Explicitly distinguishes three numbers this milestone is careful never to
conflate:
  - MODEL PROBABILITY: Model A/B's predicted probability of
    `terminal_adverse_outcome` (schema.py) -- a model output, not a dollar
    figure, and NOT a 12-month PD.
  - EXPECTED LOSS: `risk_probability * LGD * EAD` (expected_loss.py) -- a
    forward-looking, probability-weighted economic-risk measure, computed
    independently of any decision.
  - BUSINESS DECISION COST (this module): the illustrative dollar
    consequence of actually APPLYING a three-way decisioning policy
    (decisioning.py) to historical rows whose TRUE outcome (`y_true`) is
    already known. This is a backward-looking, realized-outcome
    calculation, not a probability-weighted one -- a loan that actually
    had an adverse outcome and was not declined contributes its realized
    `lgd * ead` dollar loss to total cost, not `risk_probability * lgd *
    ead`.

Three illustrative cost components make up `estimated_illustrative_total_cost`:
  1. Realized adverse-outcome loss on loans that were NOT declined
     (`approve_like` or `manual_review`) but DID have the adverse outcome:
     `lgd * ead`, summed per loan -- deliberately NOT a flat per-event
     cost, so a bigger loan that goes bad costs more than a smaller one.
     `lgd`/`ead` are passed in by the caller, sourced from
     `expected_loss.py`'s single source of truth -- this module does not
     define its own LGD/EAD.
  2. `cost_per_manual_review`: a flat, configurable cost per loan routed
     to manual review.
  3. `opportunity_cost_per_good_loan_declined`: a flat, configurable cost
     per loan that did NOT have the adverse outcome but was declined
     (foregone interest/profit on a loan that would have performed fine).

None of this is a real LendingClub operating cost, and no output is a
production decisioning-policy recommendation -- see
`DECISION_COST_DISCLAIMER` and `decisioning.py`'s neutral-label
discipline.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd

from src.credit_risk.config import PROJECT_ROOT
from src.credit_risk.decisioning import classify_decisions

DEFAULT_DECISION_COST_ASSUMPTIONS_FILE = (
    PROJECT_ROOT / "configs" / "credit_risk_decisioning_cost_assumptions.json"
)

DECISION_COST_DISCLAIMER = (
    "ILLUSTRATIVE, synthetic decisioning-cost assumptions only -- not real "
    "LendingClub review-operations costs or foregone-profit figures, and "
    "not a production decisioning-policy recommendation. See "
    "configs/credit_risk_decisioning_cost_assumptions.json."
)


class DecisionCostAssumptionError(ValueError):
    pass


@dataclass(frozen=True)
class CreditDecisionCostAssumptions:
    cost_per_manual_review: float
    opportunity_cost_per_good_loan_declined: float

    def __post_init__(self) -> None:
        if self.cost_per_manual_review < 0:
            raise DecisionCostAssumptionError(
                f"cost_per_manual_review must be >= 0; got {self.cost_per_manual_review}"
            )
        if self.opportunity_cost_per_good_loan_declined < 0:
            raise DecisionCostAssumptionError(
                "opportunity_cost_per_good_loan_declined must be >= 0; got "
                f"{self.opportunity_cost_per_good_loan_declined}"
            )


def load_credit_decision_cost_assumptions(
    config_file: Path = DEFAULT_DECISION_COST_ASSUMPTIONS_FILE,
) -> CreditDecisionCostAssumptions:
    with open(config_file) as f:
        cfg = json.load(f)
    return CreditDecisionCostAssumptions(
        cost_per_manual_review=float(cfg["cost_per_manual_review"]),
        opportunity_cost_per_good_loan_declined=float(cfg["opportunity_cost_per_good_loan_declined"]),
    )


def evaluate_decision_policy(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    ead: np.ndarray,
    lgd: float,
    review_threshold: float,
    decline_threshold: float,
    costs: CreditDecisionCostAssumptions,
) -> dict:
    """Evaluates one (review_threshold, decline_threshold) policy against
    historical rows with known `y_true` (`terminal_adverse_outcome`).

    Definitions:
      - `adverse_outcome_capture`: fraction of actual adverse-outcome
        loans that this policy would have declined (`decline_like`).
        `manual_review` is NOT counted as captured -- this module does
        not assume review perfectly (or even partially) catches an
        adverse outcome, only that `decline_like` is an unconditional
        block (mirroring `src/models/cost_model.py`'s treatment of its
        decline action).
      - `good_loan_rejection_rate`: fraction of loans WITHOUT the adverse
        outcome that were declined -- the credit-risk analogue of a
        false-positive rate.
      - `estimated_illustrative_total_cost`: see module docstring's three
        components.
    """
    y_true = np.asarray(y_true).astype(bool)
    y_prob = np.asarray(y_prob)
    ead = np.asarray(ead, dtype=float)
    if y_true.shape != y_prob.shape or y_true.shape != ead.shape:
        raise DecisionCostAssumptionError(
            "y_true, y_prob, and ead must all have the same shape; got "
            f"{y_true.shape}, {y_prob.shape}, {ead.shape}"
        )
    n = len(y_true)
    decisions = classify_decisions(y_prob, review_threshold, decline_threshold)

    is_declined = decisions == "decline_like"
    is_reviewed = decisions == "manual_review"
    is_approved = decisions == "approve_like"

    n_adverse = int(y_true.sum())
    adverse_declined = int((y_true & is_declined).sum())
    adverse_not_declined_mask = y_true & ~is_declined  # approve_like or manual_review
    adverse_outcome_capture = float(adverse_declined / n_adverse) if n_adverse else float("nan")

    n_good = n - n_adverse
    good_declined_mask = (~y_true) & is_declined
    good_declined = int(good_declined_mask.sum())
    good_loan_rejection_rate = float(good_declined / n_good) if n_good else float("nan")

    review_volume = int(is_reviewed.sum())

    realized_adverse_outcome_loss = float(np.sum(lgd * ead[adverse_not_declined_mask]))
    review_cost = review_volume * costs.cost_per_manual_review
    opportunity_cost = good_declined * costs.opportunity_cost_per_good_loan_declined
    total_cost = realized_adverse_outcome_loss + review_cost + opportunity_cost

    return {
        "review_threshold": review_threshold,
        "decline_threshold": decline_threshold,
        "lgd": lgd,
        "n_rows": n,
        "adverse_outcome_capture": adverse_outcome_capture,
        "good_loan_rejection_rate": good_loan_rejection_rate,
        "review_volume": review_volume,
        "decline_volume": int(is_declined.sum()),
        "approve_volume": int(is_approved.sum()),
        "pct_reviewed": float(review_volume / n) if n else float("nan"),
        "pct_declined": float(is_declined.sum() / n) if n else float("nan"),
        "realized_adverse_outcome_loss": realized_adverse_outcome_loss,
        "review_cost": float(review_cost),
        "opportunity_cost": float(opportunity_cost),
        "estimated_illustrative_total_cost": float(total_cost),
    }


def sensitivity_analysis(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    ead: np.ndarray,
    base_lgd: float,
    base_review_threshold: float,
    base_decline_threshold: float,
    base_costs: CreditDecisionCostAssumptions,
    variations: list[dict],
) -> pd.DataFrame:
    """For each variation in `variations` (a dict like
    `{"label": "lgd_high", "lgd": 0.95}` or
    `{"label": "stricter_decline", "decline_threshold": 0.5}`), evaluates
    the resulting policy and reports it alongside the varied inputs.
    Any of `lgd`, `review_threshold`, `decline_threshold`,
    `cost_per_manual_review`, `opportunity_cost_per_good_loan_declined`
    may be overridden per variation; anything not present falls back to
    the corresponding `base_*` argument.

    Purpose (Section G): show that adverse-outcome capture, good-loan
    rejection rate, review volume, and estimated illustrative cost all
    move as the underlying LGD/cost/threshold assumptions move -- no
    output row is called "optimal" by this function; it only reports what
    was asked for.
    """
    rows = []
    for variation in variations:
        label = variation.get("label", "unlabeled")
        lgd = variation.get("lgd", base_lgd)
        review_threshold = variation.get("review_threshold", base_review_threshold)
        decline_threshold = variation.get("decline_threshold", base_decline_threshold)
        cost_overrides = {
            k: v
            for k, v in variation.items()
            if k in ("cost_per_manual_review", "opportunity_cost_per_good_loan_declined")
        }
        varied_costs = replace(base_costs, **cost_overrides)
        result = evaluate_decision_policy(
            y_true, y_prob, ead, lgd, review_threshold, decline_threshold, varied_costs
        )
        result["variation_label"] = label
        result["cost_per_manual_review"] = varied_costs.cost_per_manual_review
        result["opportunity_cost_per_good_loan_declined"] = varied_costs.opportunity_cost_per_good_loan_declined
        rows.append(result)
    return pd.DataFrame(rows)
