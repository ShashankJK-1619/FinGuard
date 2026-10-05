from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.payments.config import SyntheticAssumptions


def _toy_real_transactions(n: int = 300, seed: int = 0) -> pd.DataFrame:
    """Small deterministic stand-in for `load_real_transactions`'s output
    shape, so generator tests don't require the raw IEEE-CIS CSVs."""
    rng = np.random.default_rng(seed)
    product_cds = rng.choice(["C", "H", "R", "S", "W"], size=n)
    card4 = rng.choice(["ct_a", "ct_b", "ct_c"], size=n)
    card6 = rng.choice(["cc_a", "cc_b"], size=n)
    device_type = rng.choice(["mobile", "desktop", None], size=n, p=[0.4, 0.4, 0.2])
    p_email = rng.choice(["gmail.com", "yahoo.com", None], size=n, p=[0.5, 0.3, 0.2])
    r_email = p_email.copy()
    mismatch_mask = rng.random(n) < 0.1
    r_email = np.where(mismatch_mask, "other.com", r_email)
    email_domain_match = np.where(
        pd.isna(p_email) & pd.isna(r_email),
        "BOTH_MISSING",
        np.where(
            pd.isna(p_email) | pd.isna(r_email),
            "ONE_MISSING",
            np.where(p_email == r_email, "MATCH", "MISMATCH"),
        ),
    )
    # TransactionDT starting well above 0 so the attempt-offset backward
    # construction has room to work with in the common case (a dedicated
    # test separately exercises the near-zero boundary).
    transaction_dt = np.sort(rng.integers(100_000, 100_000 + n * 60, size=n))
    amt = rng.gamma(2.0, 50.0, size=n)
    is_fraud = rng.choice([0, 1], size=n, p=[0.97, 0.03])

    return pd.DataFrame(
        {
            "TransactionID": np.arange(1, n + 1),
            "TransactionDT": transaction_dt,
            "TransactionAmt": amt,
            "isFraud": is_fraud,
            "ProductCD": product_cds,
            "card4": card4,
            "card6": card6,
            "P_emaildomain": p_email,
            "R_emaildomain": r_email,
            "DeviceType": device_type,
            "email_domain_match": email_domain_match,
        }
    )


@pytest.fixture
def toy_real_transactions():
    return _toy_real_transactions()


@pytest.fixture
def toy_assumptions() -> SyntheticAssumptions:
    return SyntheticAssumptions(
        random_seed=42,
        abandonment_rate=0.2,
        decline_before_success_probs=(0.6, 0.25, 0.1, 0.04, 0.01),
        abandonment_attempt_count_probs=(0.5, 0.3, 0.15, 0.05),
        decline_reason_weights={
            "insufficient_funds": 0.35,
            "risk_block": 0.25,
            "issuer_timeout": 0.15,
            "invalid_card_details": 0.15,
            "other": 0.10,
        },
        session_lead_time_seconds_min=1.0,
        session_lead_time_seconds_max=120.0,
        retry_gap_seconds_min=5.0,
        retry_gap_seconds_max=600.0,
    )
