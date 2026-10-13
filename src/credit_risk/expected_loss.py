"""Module C, Milestone 4, Section C: illustrative LGD/EAD expected-loss
framework.

`Expected Loss = risk_probability x LGD x EAD`, computed per loan from:
  - `risk_probability`: Model A's or Model B's predicted probability of
    `terminal_adverse_outcome` (see schema.py -- NOT a 12-month PD; see
    docs/moduleC_milestone4_credit_decisioning.md).
  - `LGD` (loss given default): a single configurable scalar, illustrative
    only (see configs/credit_risk_expected_loss_assumptions.json's
    `_disclaimer`). Deliberately NOT derived from this dataset's real
    `recoveries`/`total_rec_prncp` columns -- doing so would make an
    illustrative assumption look like a measured historical LGD.
  - `EAD` (exposure at default): proxied by `loan_amnt` (see
    `EAD_FIELD`/`EAD_DEFINITION_NOTE` below and the config's
    `_ead_definition_note`).

Nothing produced by this module is a regulatory capital estimate, a
CECL/IFRS 9 allowance calculation, a Basel expected-loss figure, or
LendingClub's own loss experience. It exists only to make Model A vs
Model B's economic risk-ranking comparison (Section D) and the
decisioning/cost framework (Sections E-G) concrete and auditable.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src.credit_risk.config import PROJECT_ROOT

DEFAULT_EXPECTED_LOSS_ASSUMPTIONS_FILE = (
    PROJECT_ROOT / "configs" / "credit_risk_expected_loss_assumptions.json"
)

EXPECTED_LOSS_DISCLAIMER = (
    "ILLUSTRATIVE, synthetic expected-loss framework only -- not a regulatory "
    "capital, CECL/IFRS 9, or Basel expected-loss estimate, and not LendingClub's "
    "own loss experience. See configs/credit_risk_expected_loss_assumptions.json."
)

EAD_FIELD = "loan_amnt"
EAD_DEFINITION_NOTE = (
    "EAD is proxied by `loan_amnt`, an application-time-safe Model A raw "
    "feature. Every loan in this dataset is a closed-end, fully-amortizing "
    "installment loan disbursed in full at origination (not a revolving "
    "line), so the originally funded amount is a defensible upper bound on "
    "what is outstanding at any point in the loan's life. See "
    "configs/credit_risk_expected_loss_assumptions.json's "
    "_ead_definition_note."
)


class ExpectedLossError(ValueError):
    pass


@dataclass(frozen=True)
class ExpectedLossAssumptions:
    lgd: float

    def __post_init__(self) -> None:
        if not (0.0 <= self.lgd <= 1.0):
            raise ExpectedLossError(f"lgd must be in [0, 1]; got {self.lgd}")


def load_expected_loss_assumptions(
    config_file: Path = DEFAULT_EXPECTED_LOSS_ASSUMPTIONS_FILE,
) -> ExpectedLossAssumptions:
    with open(config_file) as f:
        cfg = json.load(f)
    return ExpectedLossAssumptions(lgd=float(cfg["lgd"]))


def compute_expected_loss(y_prob: np.ndarray, ead: np.ndarray, lgd: float) -> np.ndarray:
    """Elementwise `Expected Loss = y_prob * lgd * ead`, per loan.

    `y_prob` and `ead` must be the same shape. `lgd` is a single scalar
    applied uniformly (this module does not support a per-loan LGD --
    that would require dataset-derived recovery modeling, explicitly out
    of scope here per the module docstring). Raises `ExpectedLossError`
    (not a silent NaN/garbage result) on shape mismatch, an out-of-range
    `lgd`, an out-of-range `y_prob`, or a negative `ead` -- all signal a
    caller bug, not a legitimate illustrative input.
    """
    y_prob = np.asarray(y_prob, dtype=float)
    ead = np.asarray(ead, dtype=float)
    if y_prob.shape != ead.shape:
        raise ExpectedLossError(
            f"y_prob and ead must have the same shape; got {y_prob.shape} vs {ead.shape}"
        )
    if not (0.0 <= lgd <= 1.0):
        raise ExpectedLossError(f"lgd must be in [0, 1]; got {lgd}")
    if y_prob.size and (np.nanmin(y_prob) < 0.0 or np.nanmax(y_prob) > 1.0):
        raise ExpectedLossError("y_prob values must be within [0, 1]")
    if ead.size and np.nanmin(ead) < 0.0:
        raise ExpectedLossError("ead values must be non-negative")
    return y_prob * lgd * ead
