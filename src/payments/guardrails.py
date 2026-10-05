"""Module A, Milestone 1: rule-based fraud guardrails (NOT a trained model).

See docs/payments_data_model.md Section 9. Every rule here is a fixed
threshold comparison — there is no `.fit()` call anywhere in this module —
per the explicit instruction not to train any new ML model in this
milestone. Thresholds come from `configs/payments_guardrail_rules.json`
via `src.payments.config.load_guardrail_rules`, not hardcoded.

Backtest scope limitation (repeated from the design doc): only real-linked
sessions carry a real `isFraud` label. The backtest below runs on that
subset only and is a descriptive crosstab (how often each rule fires
among known-fraud vs. known-legitimate real-linked sessions) — not a
predictive-performance claim comparable to Module B's ROC-AUC/PR-AUC.
"""
from __future__ import annotations

import duckdb
import pandas as pd

from src.payments.config import GuardrailRules

FLAG_COLUMNS = ["velocity_flag", "high_amount_flag", "mismatched_email_flag", "combined_flag"]

_FLAGS_QUERY_TEMPLATE = """
SELECT
    session_id,
    is_synthetic_abandonment,
    isFraud,
    attempt_count >= {velocity_attempt_threshold} AS velocity_flag,
    TransactionAmt >= {high_amount_threshold} AS high_amount_flag,
    email_domain_match = 'MISMATCH' AS mismatched_email_flag
FROM payment_sessions
"""


def compute_high_amount_threshold(con: duckdb.DuckDBPyConnection, rules: GuardrailRules) -> float:
    """Percentile is computed over REAL-linked sessions' TransactionAmt
    only (docs/payments_data_model.md Sec 9) — abandonment sessions'
    amounts are donor-cloned (Sec 4), so including them would just
    resample the same real distribution with duplicates, not add
    information."""
    query = f"""
        SELECT QUANTILE_CONT(TransactionAmt, {rules.high_amount_percentile}) AS q
        FROM payment_sessions
        WHERE is_synthetic_abandonment = FALSE
    """
    return float(con.execute(query).df()["q"].iloc[0])


def compute_guardrail_flags(con: duckdb.DuckDBPyConnection, rules: GuardrailRules) -> pd.DataFrame:
    high_amount_threshold = compute_high_amount_threshold(con, rules)
    query = _FLAGS_QUERY_TEMPLATE.format(
        velocity_attempt_threshold=rules.velocity_attempt_threshold,
        high_amount_threshold=high_amount_threshold,
    )
    df = con.execute(query).df()
    df["combined_flag"] = df["velocity_flag"] | (df["high_amount_flag"] & df["mismatched_email_flag"])
    return df


def backtest_guardrails_against_fraud_label(flags_df: pd.DataFrame) -> pd.DataFrame:
    """Descriptive crosstab on the real-linked subset only (Sec 9).
    Abandonment-only sessions (`is_synthetic_abandonment == True`) have no
    `isFraud` label and are excluded here, not silently treated as
    non-fraud."""
    real = flags_df[~flags_df["is_synthetic_abandonment"]].copy()
    real["isFraud"] = real["isFraud"].astype(bool)
    n = len(real)
    n_fraud = int(real["isFraud"].sum())

    rows = []
    for flag_col in FLAG_COLUMNS:
        flagged = real[flag_col].astype(bool)
        n_flagged = int(flagged.sum())
        true_positives = int((real["isFraud"] & flagged).sum())
        rows.append(
            {
                "rule": flag_col,
                "n_real_linked_sessions": n,
                "flag_rate_overall": n_flagged / n if n else float("nan"),
                "flag_rate_among_fraud": float(real.loc[real["isFraud"], flag_col].mean()) if n_fraud else float("nan"),
                "flag_rate_among_legit": (
                    float(real.loc[~real["isFraud"], flag_col].mean()) if (n - n_fraud) else float("nan")
                ),
                "precision": true_positives / n_flagged if n_flagged else float("nan"),
                "recall": true_positives / n_fraud if n_fraud else float("nan"),
            }
        )
    return pd.DataFrame(rows)
