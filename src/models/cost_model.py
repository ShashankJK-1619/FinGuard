"""Milestone 4: illustrative fraud-decisioning cost framework.

Everything costed here is a HYPOTHETICAL, illustrative portfolio assumption
chosen by the author for this learning project (see
configs/cost_assumptions.json). None of it represents Vesta Corporation,
the IEEE-CIS competition, a real bank, a real payments company, or any
employer, and no total-cost figure produced by this module should be read
as a real financial saving or loss estimate. The point of this module is
to make the mechanics of cost-driven threshold selection concrete and
auditable — "the best threshold depends on assumed unit costs, here is
exactly how" — not to produce a number anyone should act on.

Three-action decisioning (approve / manual review / decline) requires two
ordered thresholds on the fraud probability, not one:
  - score <  review_threshold                      -> approve
  - review_threshold <= score < decline_threshold   -> review
  - score >= decline_threshold                      -> decline

Methodology correction (post-initial-draft): the first version of this
module modeled manual review as an unlimited-capacity, perfect backstop —
every reviewed transaction was treated as "caught" if it was fraud, with
no limit on how many transactions could be reviewed. Both assumptions were
unrealistic and are now explicit, configurable parameters instead of
hardcoded behavior:
  - `review_capacity_frac`: the maximum FRACTION of an evaluated batch
    that can be routed to manual review (`None` = unlimited, kept as the
    default so existing callers/tests that don't set it reproduce the old
    behavior exactly). When the number of transactions the threshold pair
    would route to review exceeds capacity, the highest-risk transactions
    (by predicted probability, descending) are prioritized for the
    capacity that exists; the rest fall back to **approve**, not decline —
    an explicit choice, not an oversight: the review band was never above
    the decline threshold, so substituting an automatic block for a
    capacity-starved human review would silently convert "needs a human
    look" into "blocked outright," a stricter action than the threshold
    pair itself specifies.
  - `review_fraud_catch_rate`: probability that a FRAUDULENT transaction
    actually routed to review is correctly caught (`1.0` = perfect catch,
    the old assumption, kept as default). The complement is folded into
    `false_negatives`/missed-fraud cost, in expectation.
  - `review_false_positive_rate`: probability that a LEGITIMATE
    transaction actually routed to review is incorrectly declined by the
    review process itself (`0.0` = no review error, the old assumption,
    kept as default). This is folded into `false_positives`/false-positive
    cost, in expectation, as a channel distinct from the unconditional
    decline-threshold false positives.
All three default to the original (unlimited/perfect/error-free) values so
any caller that doesn't set them reproduces the pre-correction numbers
exactly; `configs/cost_assumptions.json` sets all three to more realistic,
still explicitly hypothetical, illustrative values for Milestone 4's own
run — see docs/milestone4_calibration_decisioning.md.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.config import PROJECT_ROOT

DEFAULT_COST_ASSUMPTIONS_FILE = PROJECT_ROOT / "configs" / "cost_assumptions.json"

COST_ASSUMPTIONS_DISCLAIMER = (
    "HYPOTHETICAL, illustrative portfolio assumptions only — not real "
    "figures from Vesta Corporation, the IEEE-CIS competition, or any "
    "real bank/payments company. See configs/cost_assumptions.json."
)


class CostAssumptionError(ValueError):
    pass


@dataclass(frozen=True)
class CostAssumptions:
    cost_per_missed_fraud: float
    cost_per_false_positive_decline: float
    cost_per_manual_review: float
    # Realistic-review-model parameters (methodology correction). Defaults
    # reproduce the original unlimited-capacity/perfect-review behavior so
    # existing callers are unaffected unless they opt in.
    review_capacity_frac: float | None = None
    review_fraud_catch_rate: float = 1.0
    review_false_positive_rate: float = 0.0

    def __post_init__(self) -> None:
        if self.review_capacity_frac is not None and not (0 <= self.review_capacity_frac <= 1):
            raise CostAssumptionError(f"review_capacity_frac must be in [0, 1] or None; got {self.review_capacity_frac}")
        if not (0 <= self.review_fraud_catch_rate <= 1):
            raise CostAssumptionError(f"review_fraud_catch_rate must be in [0, 1]; got {self.review_fraud_catch_rate}")
        if not (0 <= self.review_false_positive_rate <= 1):
            raise CostAssumptionError(
                f"review_false_positive_rate must be in [0, 1]; got {self.review_false_positive_rate}"
            )


def load_cost_assumptions(config_file: Path = DEFAULT_COST_ASSUMPTIONS_FILE) -> CostAssumptions:
    with open(config_file) as f:
        cfg = json.load(f)
    return CostAssumptions(
        cost_per_missed_fraud=float(cfg["cost_per_missed_fraud"]),
        cost_per_false_positive_decline=float(cfg["cost_per_false_positive_decline"]),
        cost_per_manual_review=float(cfg["cost_per_manual_review"]),
        review_capacity_frac=(
            float(cfg["review_capacity_frac"]) if cfg.get("review_capacity_frac") is not None else None
        ),
        review_fraud_catch_rate=float(cfg.get("review_fraud_catch_rate", 1.0)),
        review_false_positive_rate=float(cfg.get("review_false_positive_rate", 0.0)),
    )


class ThresholdError(ValueError):
    pass


def classify_actions(y_prob: np.ndarray, review_threshold: float, decline_threshold: float) -> np.ndarray:
    """Three-action decisioning. Requires
    0 <= review_threshold < decline_threshold <= 1 — a review band that has
    collapsed to zero width, or is inverted, is not a valid three-action
    policy and is rejected rather than silently degenerating to two
    actions.
    """
    if not (0 <= review_threshold < decline_threshold <= 1):
        raise ThresholdError(
            "Require 0 <= review_threshold < decline_threshold <= 1; got "
            f"review_threshold={review_threshold}, decline_threshold={decline_threshold}"
        )
    y_prob = np.asarray(y_prob)
    actions = np.full(y_prob.shape, "approve", dtype=object)
    actions[(y_prob >= review_threshold) & (y_prob < decline_threshold)] = "review"
    actions[y_prob >= decline_threshold] = "decline"
    return actions


def evaluate_decision_policy(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    review_threshold: float,
    decline_threshold: float,
    costs: CostAssumptions,
) -> dict:
    """Evaluates one (review_threshold, decline_threshold) policy.

    Definitions (fixed here so every downstream table uses the same ones):
      - fraud_recall: EXPECTED fraction of actual fraud caught. Fraud that
        is declined outright is always counted as caught (decline is an
        automatic block, not a human judgment call, so — unlike review —
        it is still modeled as perfectly effective). Fraud that is
        actually routed to review (i.e. within `review_capacity_frac`,
        see below) is caught with probability `costs.review_fraud_catch_rate`
        — `1.0` reproduces the original "review is a perfect backstop"
        assumption; anything lower folds the complement into
        `false_negatives` in expectation.
      - precision: of transactions ACTED ON (declined outright, or
        actually routed to review within capacity), the fraction that are
        actually fraud — a property of the flagging/screening step itself,
        not of how effectively the downstream action worked.
      - false_positives: legitimate transactions incorrectly blocked, via
        EITHER channel — declined outright, OR (in expectation)
        incorrectly declined by the review process itself at rate
        `costs.review_false_positive_rate`. A legitimate transaction that
        is reviewed and not incorrectly declined is not counted here.
      - false_negatives: fraud transactions APPROVED outright, PLUS (in
        expectation) fraud transactions reviewed but not caught
        (`1 - review_fraud_catch_rate`), PLUS fraud transactions that fell
        into the review band but exceeded review capacity and were
        therefore never reviewed at all (see `review_overflow_to_approve`).

    Review capacity (`costs.review_capacity_frac`, methodology correction):
    if the number of transactions this threshold pair would route to
    review exceeds `floor(review_capacity_frac * n)`, the highest-risk
    transactions (by `y_prob`, descending) are prioritized for the
    capacity that exists. Transactions that fall in the review band but
    don't fit within capacity default to **approve**, not decline — an
    explicit, documented choice: the review band is by definition below
    the decline threshold, so a capacity-starved reviewed transaction
    reverts to the less severe action rather than being silently upgraded
    to an automatic block. `review_capacity_frac=None` (the default)
    reproduces the original unlimited-capacity behavior exactly.
    """
    y_true = np.asarray(y_true).astype(bool)
    y_prob = np.asarray(y_prob)
    n = len(y_true)
    actions = classify_actions(y_prob, review_threshold, decline_threshold)

    is_declined = actions == "decline"
    is_review_desired = actions == "review"
    is_approved_raw = actions == "approve"

    review_capacity_limit: int | None = None
    is_overflow_to_approve = np.zeros(n, dtype=bool)
    if costs.review_capacity_frac is not None:
        review_capacity_limit = int(np.floor(costs.review_capacity_frac * n))
        desired_idx = np.flatnonzero(is_review_desired)
        if len(desired_idx) > review_capacity_limit:
            # Prioritize the highest-risk candidates for the capacity that
            # exists; stable-sort by descending score so ties break by
            # original (score-sorted-irrelevant) order deterministically.
            order = np.argsort(-y_prob[desired_idx], kind="stable")
            keep_idx = desired_idx[order[:review_capacity_limit]]
            overflow_idx = desired_idx[order[review_capacity_limit:]]
            is_reviewed = np.zeros(n, dtype=bool)
            is_reviewed[keep_idx] = True
            is_overflow_to_approve[overflow_idx] = True
        else:
            is_reviewed = is_review_desired.copy()
    else:
        is_reviewed = is_review_desired.copy()

    is_approved = is_approved_raw | is_overflow_to_approve
    is_flagged = is_declined | is_reviewed  # transactions actually acted on

    # Decline channel: treated as a perfect, unconditional block.
    fraud_declined = y_true & is_declined
    legit_declined = (~y_true) & is_declined

    # Review channel: imperfect catch rate on fraud, imperfect error rate
    # on legit — both folded in as expectations, not simulated draws, so
    # results are deterministic and directly hand-computable.
    reviewed_fraud = int((y_true & is_reviewed).sum())
    reviewed_legit = int((~y_true & is_reviewed).sum())
    reviewed_fraud_expected_caught = costs.review_fraud_catch_rate * reviewed_fraud
    reviewed_fraud_expected_missed = (1 - costs.review_fraud_catch_rate) * reviewed_fraud
    review_error_false_positives = costs.review_false_positive_rate * reviewed_legit

    # Approve channel (including capacity overflow): always missed if fraud.
    fraud_approved = int((y_true & is_approved).sum())

    n_fraud = int(y_true.sum())
    expected_fraud_caught = float(fraud_declined.sum()) + reviewed_fraud_expected_caught
    expected_fraud_missed = float(fraud_approved) + reviewed_fraud_expected_missed
    fraud_recall = float(expected_fraud_caught / n_fraud) if n_fraud else float("nan")

    n_flagged = int(is_flagged.sum())
    precision = float((y_true & is_flagged).sum() / n_flagged) if n_flagged else float("nan")

    expected_false_positives = float(legit_declined.sum()) + review_error_false_positives

    total_cost = (
        expected_fraud_missed * costs.cost_per_missed_fraud
        + expected_false_positives * costs.cost_per_false_positive_decline
        + int(is_reviewed.sum()) * costs.cost_per_manual_review
    )

    return {
        "review_threshold": review_threshold,
        "decline_threshold": decline_threshold,
        "n_rows": n,
        "fraud_recall": fraud_recall,
        "precision": precision,
        "false_positives": expected_false_positives,
        "false_negatives": expected_fraud_missed,
        "review_volume": int(is_reviewed.sum()),
        "decline_volume": int(is_declined.sum()),
        "review_capacity_limit": review_capacity_limit,
        "review_overflow_to_approve": int(is_overflow_to_approve.sum()),
        "reviewed_fraud_expected_caught": float(reviewed_fraud_expected_caught),
        "reviewed_fraud_expected_missed": float(reviewed_fraud_expected_missed),
        "review_error_false_positives": float(review_error_false_positives),
        "pct_reviewed": float(is_reviewed.sum() / n) if n else float("nan"),
        "pct_declined": float(is_declined.sum() / n) if n else float("nan"),
        "pct_affected": float((is_reviewed.sum() + is_declined.sum()) / n) if n else float("nan"),
        "estimated_illustrative_total_cost": float(total_cost),
    }


def threshold_pair_grid(review_thresholds: list[float], decline_thresholds: list[float]) -> list[tuple[float, float]]:
    """All (review, decline) combinations with review < decline. Pairs that
    violate the ordering requirement are silently dropped here (not an
    error) since the two threshold lists are typically generated
    independently and are expected to overlap.
    """
    return [(r, d) for r in review_thresholds for d in decline_thresholds if r < d]


def evaluate_policy_grid(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    review_thresholds: list[float],
    decline_thresholds: list[float],
    costs: CostAssumptions,
) -> pd.DataFrame:
    """Evaluates every valid (review, decline) pair, sorted by ascending
    illustrative total cost. This is exploration, not threshold selection:
    callers should look at the whole table (fraud_recall/precision/volumes
    alongside cost), not just take row 0 and declare it a production
    policy — the lowest-cost row is lowest-cost only under the specific,
    hypothetical cost assumptions passed in.
    """
    rows = [
        evaluate_decision_policy(y_true, y_prob, r, d, costs)
        for r, d in threshold_pair_grid(review_thresholds, decline_thresholds)
    ]
    return pd.DataFrame(rows).sort_values("estimated_illustrative_total_cost").reset_index(drop=True)


def sensitivity_analysis(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    review_thresholds: list[float],
    decline_thresholds: list[float],
    base_costs: CostAssumptions,
    cost_variations: list[dict],
) -> pd.DataFrame:
    """For each cost variation, finds the lowest-illustrative-cost policy
    on the same threshold grid and reports it alongside the varied cost
    inputs. `cost_variations` is a list of dicts like
    {"label": "fraud_loss_2x", "cost_per_missed_fraud": 1000.0} — any
    CostAssumptions field not present in a variation falls back to
    `base_costs`.

    Purpose: show that the argmin-cost (review, decline) pair moves as the
    underlying cost assumptions move — i.e. the "best" threshold is a
    function of assumed business economics, not a fixed property of the
    model.
    """
    rows = []
    for variation in cost_variations:
        label = variation.get("label", "unlabeled")
        overrides = {k: v for k, v in variation.items() if k != "label"}
        varied_costs = replace(base_costs, **overrides)
        grid = evaluate_policy_grid(y_true, y_prob, review_thresholds, decline_thresholds, varied_costs)
        best = grid.iloc[0].to_dict()
        best["variation_label"] = label
        best["cost_per_missed_fraud"] = varied_costs.cost_per_missed_fraud
        best["cost_per_false_positive_decline"] = varied_costs.cost_per_false_positive_decline
        best["cost_per_manual_review"] = varied_costs.cost_per_manual_review
        best["review_capacity_frac"] = varied_costs.review_capacity_frac
        best["review_fraud_catch_rate"] = varied_costs.review_fraud_catch_rate
        best["review_false_positive_rate"] = varied_costs.review_false_positive_rate
        rows.append(best)
    return pd.DataFrame(rows)
