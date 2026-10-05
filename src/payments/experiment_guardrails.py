"""Module A, Milestone 2: fraud/risk guardrail metrics + the ship/no-ship
decision rule.

See docs/moduleA_milestone2_experimentation.md Section H and
docs/payments_experiment_metric_dictionary.md for the four guardrail
metrics' numerator/denominator/scope. All four are "higher = worse":

- fraud_rate: isFraud rate among REAL-LINKED sessions only (no fraud
  label exists for synthetic-abandonment sessions — same scope limitation
  as Milestone 1's guardrail backtest).
- fraudulent_completed_session_rate: isFraud rate among real-linked
  sessions that reached experiment_final_outcome == 'authorized'. In this
  milestone's data this is numerically identical to fraud_rate, because
  every real-linked session is 'authorized' by Milestone 1 construction
  (a real transaction IS the authorized terminal attempt of its session)
  and the synthetic treatment-response mechanism never touches
  real-linked sessions (experiment_effects.py) — stated explicitly so the
  coincidence isn't mistaken for a new finding.
- legitimate_decline_proxy_rate: among real-linked, NOT-fraudulent
  sessions, the share with at least one attempt whose
  decline_reason_code == 'risk_block'. An illustrative proxy for
  "friction imposed on legitimate customers by the risk-decline reason",
  NOT a trained model's false-positive rate (there is no model in this
  project to compute a true confusion matrix against) — and, because
  decline_reason_code is sampled independently of isFraud in Milestone 1
  (configs/payments_synthetic_assumptions.json), expect no real
  between-arm signal here either, the same kind of methodological
  artifact as Milestone 1's velocity_flag.
- excessive_retry_rate: share of ALL sessions (real-linked + abandonment)
  with attempt_count over the configured threshold.

`evaluate_decision_rule` implements the illustrative "ship only if primary
is significant AND positive AND clears a minimum practical uplift AND no
guardrail breaches its predeclared tolerance" rule from Sec H. All
tolerances come from `ExperimentAssumptions`, never hardcoded here.
"""
from __future__ import annotations

import duckdb
import pandas as pd

from src.payments.config import ExperimentAssumptions
from src.payments.schema import ASSIGNMENT_COL, ASSIGNMENT_CONTROL, ASSIGNMENT_TREATMENT

GUARDRAIL_METRICS = [
    "fraud_rate",
    "fraudulent_completed_session_rate",
    "legitimate_decline_proxy_rate",
    "excessive_retry_rate",
]

_FRAUD_GUARDRAILS_QUERY = """
SELECT
    assignment,
    COUNT(*) AS n_real_linked,
    AVG(CASE WHEN isFraud = 1 THEN 1.0 ELSE 0.0 END) AS fraud_rate,
    SUM(CASE WHEN experiment_final_outcome = 'authorized' AND isFraud = 1 THEN 1 ELSE 0 END) * 1.0
        / NULLIF(SUM(CASE WHEN experiment_final_outcome = 'authorized' THEN 1 ELSE 0 END), 0)
        AS fraudulent_completed_session_rate
FROM payment_experiment_sessions
WHERE is_synthetic_abandonment = FALSE
GROUP BY assignment
ORDER BY assignment
"""

_LEGITIMATE_DECLINE_PROXY_QUERY = """
WITH risk_block_sessions AS (
    SELECT DISTINCT session_id FROM payment_attempts WHERE decline_reason_code = 'risk_block'
)
SELECT
    s.assignment,
    COUNT(*) AS n_real_legit,
    AVG(CASE WHEN rb.session_id IS NOT NULL THEN 1.0 ELSE 0.0 END) AS legitimate_decline_proxy_rate
FROM payment_experiment_sessions s
LEFT JOIN risk_block_sessions rb ON s.session_id = rb.session_id
WHERE s.is_synthetic_abandonment = FALSE AND s.isFraud = 0
GROUP BY s.assignment
ORDER BY s.assignment
"""

_EXCESSIVE_RETRY_QUERY_TEMPLATE = """
SELECT
    assignment,
    COUNT(*) AS n_sessions,
    AVG(CASE WHEN attempt_count > {threshold} THEN 1.0 ELSE 0.0 END) AS excessive_retry_rate
FROM payment_experiment_sessions
GROUP BY assignment
ORDER BY assignment
"""


def compute_guardrail_metrics(con: duckdb.DuckDBPyConnection, assumptions: ExperimentAssumptions) -> pd.DataFrame:
    """One row per arm with all four guardrail metrics (plus their
    supporting denominator counts)."""
    fraud = con.execute(_FRAUD_GUARDRAILS_QUERY).df()
    proxy = con.execute(_LEGITIMATE_DECLINE_PROXY_QUERY).df()
    excessive = con.execute(
        _EXCESSIVE_RETRY_QUERY_TEMPLATE.format(threshold=assumptions.excessive_retry_attempt_threshold)
    ).df()

    merged = fraud.merge(proxy, on="assignment", how="outer").merge(excessive, on="assignment", how="outer")
    return merged.sort_values("assignment").reset_index(drop=True)


def evaluate_decision_rule(
    primary_test_result: dict, guardrail_df: pd.DataFrame, assumptions: ExperimentAssumptions
) -> dict:
    """Illustrative ship/no-ship decision rule (Sec H). Ship only if:
      1. the primary metric's two-proportion test is statistically
         significant AND the uplift is positive;
      2. the absolute uplift clears `assumptions.min_practical_uplift_abs`;
      3. no guardrail's (treatment - control) absolute difference exceeds
         its predeclared tolerance.
    """
    reasons: list[str] = []

    primary_significant_and_positive = (
        primary_test_result["is_statistically_significant"] and primary_test_result["absolute_uplift"] > 0
    )
    if not primary_significant_and_positive:
        reasons.append(
            "Primary metric does not show statistically significant, positive evidence "
            f"(p_value={primary_test_result['p_value']:.4g}, "
            f"absolute_uplift={primary_test_result['absolute_uplift']:.4f})."
        )

    practical_uplift_clears_threshold = primary_test_result["absolute_uplift"] >= assumptions.min_practical_uplift_abs
    if not practical_uplift_clears_threshold:
        reasons.append(
            f"Absolute uplift {primary_test_result['absolute_uplift']:.4f} is below the predeclared minimum "
            f"practical-significance threshold {assumptions.min_practical_uplift_abs:.4f}."
        )

    tolerance_map = {
        "fraud_rate": assumptions.fraud_rate_guardrail_tolerance_abs,
        "fraudulent_completed_session_rate": assumptions.fraudulent_completed_rate_guardrail_tolerance_abs,
        "legitimate_decline_proxy_rate": assumptions.legitimate_decline_proxy_guardrail_tolerance_abs,
        "excessive_retry_rate": assumptions.excessive_retry_rate_guardrail_tolerance_abs,
    }
    indexed = guardrail_df.set_index(ASSIGNMENT_COL)
    guardrail_breaches = []
    for metric, tolerance in tolerance_map.items():
        control_value = float(indexed.loc[ASSIGNMENT_CONTROL, metric])
        treatment_value = float(indexed.loc[ASSIGNMENT_TREATMENT, metric])
        diff = treatment_value - control_value
        breached = diff > tolerance
        guardrail_breaches.append(
            {
                "metric": metric,
                "control": control_value,
                "treatment": treatment_value,
                "absolute_diff": diff,
                "tolerance": tolerance,
                "breached": bool(breached),
            }
        )
        if breached:
            reasons.append(f"Guardrail '{metric}' increased by {diff:.4f}, exceeding tolerance {tolerance:.4f}.")

    any_guardrail_breach = any(g["breached"] for g in guardrail_breaches)
    ship = primary_significant_and_positive and practical_uplift_clears_threshold and not any_guardrail_breach

    return {
        "decision": "ship" if ship else "no-ship",
        "primary_significant_and_positive": primary_significant_and_positive,
        "practical_uplift_clears_threshold": practical_uplift_clears_threshold,
        "guardrail_breaches": guardrail_breaches,
        "any_guardrail_breach": any_guardrail_breach,
        "reasons": reasons,
    }
