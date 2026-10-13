"""Centralized, source-cited metrics for FinGuard's portfolio visuals.

Design rule (FinGuard Portfolio Production, Step 3, Section J):
    - Every chart-generation script in `src/portfolio/` imports its numbers
      from *this* module only. No chart script re-types a metric from a
      doc or JSON file directly, so there is exactly one place a number
      could ever drift out of sync with its source.
    - This module *tries* to read each metric from the real generated
      artifact under `data/interim/` first (those files are reproducible
      by re-running the relevant milestone script against the raw data).
      `data/interim/` is gitignored, so a fresh clone without the raw
      LendingClub/IEEE-CIS CSVs will not have these files. In that case
      this module falls back to the literal value recorded below, which
      was copied from the artifact (or doc) cited in that constant's
      `source` field at the time this module was written.
    - If a regenerated artifact's value and the fallback above should
      ever diverge, that is a signal the docs/fallbacks here are stale
      and need to be refreshed from the regenerated artifact.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
FIGURES_DIR = PROJECT_ROOT / "docs" / "figures"


def _load_json(relative_path: str) -> dict | None:
    """Best-effort load of a gitignored data/interim/*.json artifact.

    Returns None (not an exception) when the artifact is absent, which is
    the expected state in a fresh clone without raw data downloaded.
    """
    path = PROJECT_ROOT / relative_path
    if not path.exists():
        return None
    with open(path, "r") as f:
        return json.load(f)


def _dig(d: dict | None, *keys: str) -> Any:
    """Nested dict lookup; returns None at the first missing/absent key."""
    cur = d
    for k in keys:
        if cur is None or k not in cur:
            return None
        cur = cur[k]
    return cur


def _resolve(relative_path: str, keys: tuple[str, ...], fallback: Any) -> Any:
    """Read artifact[keys...] if the artifact exists and has that path, else fallback."""
    value = _dig(_load_json(relative_path), *keys)
    return value if value is not None else fallback


@dataclass(frozen=True)
class Metric:
    value: Any
    source: str


# ---------------------------------------------------------------------------
# Fraud dataset scale
# Source: docs/data_profile.md (train_transaction row count / isFraud rate)
# ---------------------------------------------------------------------------
FRAUD_N_TRANSACTIONS = Metric(590_540, "docs/data_profile.md (train_transaction row count)")
FRAUD_PREVALENCE_PCT = Metric(3.5, "docs/data_profile.md / README Key Results (isFraud prevalence, rounded)")

# ---------------------------------------------------------------------------
# Fraud model comparison — temporal benchmark, calibrated head-to-head.
# Source: data/interim/milestone5_metrics.json -> head_to_head_calibrated_metrics
#         .temporal_benchmark.{logistic, nonlinear_preferred_no_time_index}
#         Also documented in docs/milestone5_nonlinear_comparison.md.
# ---------------------------------------------------------------------------
_M5 = "data/interim/milestone5_metrics.json"

FRAUD_LOGISTIC_ROC_AUC = Metric(
    _resolve(_M5, ("head_to_head_calibrated_metrics", "temporal_benchmark", "logistic", "roc_auc"), 0.7317373035074971),
    f"{_M5} -> head_to_head_calibrated_metrics.temporal_benchmark.logistic.roc_auc",
)
FRAUD_LOGISTIC_PR_AUC = Metric(
    _resolve(_M5, ("head_to_head_calibrated_metrics", "temporal_benchmark", "logistic", "pr_auc"), 0.12017932219703062),
    f"{_M5} -> head_to_head_calibrated_metrics.temporal_benchmark.logistic.pr_auc",
)
FRAUD_NONLINEAR_ROC_AUC = Metric(
    _resolve(_M5, ("head_to_head_calibrated_metrics", "temporal_benchmark", "nonlinear_preferred_no_time_index", "roc_auc"), 0.779923625253101),
    f"{_M5} -> head_to_head_calibrated_metrics.temporal_benchmark.nonlinear_preferred_no_time_index.roc_auc",
)
FRAUD_NONLINEAR_PR_AUC = Metric(
    _resolve(_M5, ("head_to_head_calibrated_metrics", "temporal_benchmark", "nonlinear_preferred_no_time_index", "pr_auc"), 0.15356920252255282),
    f"{_M5} -> head_to_head_calibrated_metrics.temporal_benchmark.nonlinear_preferred_no_time_index.pr_auc",
)

# ---------------------------------------------------------------------------
# Fraud decisioning tradeoff — illustrative cost-based policy, both splits.
# Source: data/interim/milestone5_metrics.json ->
#         {logistic_decision_policy, nonlinear_preferred_no_time_index_decision_policy}
#         .{best_policy_selection (= validation), temporal_benchmark_policy_evaluation}
#         Also documented in docs/milestone5_nonlinear_comparison.md Section 6.
# ---------------------------------------------------------------------------
FRAUD_LOGISTIC_POLICY_VAL = Metric(
    _resolve(_M5, ("logistic_decision_policy", "best_policy_selection"), {
        "fraud_recall": 0.1804, "estimated_illustrative_total_cost": 831242.25,
    }),
    f"{_M5} -> logistic_decision_policy.best_policy_selection",
)
FRAUD_NONLINEAR_POLICY_VAL = Metric(
    _resolve(_M5, ("nonlinear_preferred_no_time_index_decision_policy", "best_policy_selection"), {
        "fraud_recall": 0.3371, "estimated_illustrative_total_cost": 687769.75,
    }),
    f"{_M5} -> nonlinear_preferred_no_time_index_decision_policy.best_policy_selection",
)
FRAUD_LOGISTIC_POLICY_TB = Metric(
    _resolve(_M5, ("logistic_decision_policy", "temporal_benchmark_policy_evaluation"), {
        "fraud_recall": 0.1704718417047184, "estimated_illustrative_total_cost": 1658498.0,
    }),
    f"{_M5} -> logistic_decision_policy.temporal_benchmark_policy_evaluation",
)
FRAUD_NONLINEAR_POLICY_TB = Metric(
    _resolve(_M5, ("nonlinear_preferred_no_time_index_decision_policy", "temporal_benchmark_policy_evaluation"), {
        "fraud_recall": 0.3155251141552512, "estimated_illustrative_total_cost": 1417903.0,
    }),
    f"{_M5} -> nonlinear_preferred_no_time_index_decision_policy.temporal_benchmark_policy_evaluation",
)

# ---------------------------------------------------------------------------
# Payments experiment (synthetic treatment layered on real IEEE-CIS transactions).
# Source: data/interim/payments_milestone2_experiment_metrics.json ->
#         {primary_metric_test, experiment_assumptions}
#         Also documented in docs/moduleA_milestone2_experimentation.md.
# ---------------------------------------------------------------------------
_PAY = "data/interim/payments_milestone2_experiment_metrics.json"

PAYMENTS_PRIMARY_METRIC_TEST = Metric(
    _resolve(_PAY, ("primary_metric_test",), {
        "control_rate": 0.8692435229492044,
        "treatment_rate": 0.8763152092428452,
        "absolute_uplift": 0.007071686293640855,
        "ci_low": 0.00548671733651792,
        "ci_high": 0.008656655250763791,
        "p_value": 2.236486793079287e-18,
        "is_statistically_significant": True,
    }),
    f"{_PAY} -> primary_metric_test",
)
PAYMENTS_MIN_PRACTICAL_UPLIFT_ABS = Metric(
    _resolve(_PAY, ("experiment_assumptions", "min_practical_uplift_abs"), 0.01),
    f"{_PAY} -> experiment_assumptions.min_practical_uplift_abs",
)
PAYMENTS_N_SESSIONS = Metric(
    _resolve(_PAY, ("diagnostics", "crossover_check", "n_sessions"), 679_121),
    f"{_PAY} -> diagnostics.crossover_check.n_sessions",
)

# ---------------------------------------------------------------------------
# Credit risk — risk bands, Model A, temporal benchmark.
# Source: data/interim/credit_risk_milestone4_metrics.json ->
#         models.model_a_application_time_safe.risk_band_summary_benchmark
#         Also documented in docs/milestone4_calibration_decisioning.md
#         [note: credit risk doc is docs/moduleC_milestone4_credit_decisioning.md]
# ---------------------------------------------------------------------------
_CR4 = "data/interim/credit_risk_milestone4_metrics.json"

CREDIT_RISK_BANDS_BENCHMARK = Metric(
    _resolve(_CR4, ("models", "model_a_application_time_safe", "risk_band_summary_benchmark"), [
        {"risk_band": "low", "n": 30838, "mean_predicted_probability": 0.06415009402181308, "observed_adverse_outcome_rate": 0.07296193008625722},
        {"risk_band": "medium", "n": 52181, "mean_predicted_probability": 0.12954848894173052, "observed_adverse_outcome_rate": 0.12644449128993313},
        {"risk_band": "high", "n": 51785, "mean_predicted_probability": 0.25326939066080006, "observed_adverse_outcome_rate": 0.235126001737955},
    ]),
    f"{_CR4} -> models.model_a_application_time_safe.risk_band_summary_benchmark",
)

# ---------------------------------------------------------------------------
# Credit risk — calibration slope drift, Model A & Model B, validation -> benchmark.
# Source: data/interim/credit_risk_milestone4_metrics.json ->
#         models.{model_a_application_time_safe, model_b_with_lender_grade_signal}
#         .{calibration_slope_intercept_validation, calibration_slope_intercept_benchmark}
#         Also documented in docs/moduleC_milestone4_credit_decisioning.md.
# ---------------------------------------------------------------------------
CREDIT_MODEL_A_SLOPE_VALIDATION = Metric(
    _resolve(_CR4, ("models", "model_a_application_time_safe", "calibration_slope_intercept_validation", "slope"), 0.9494196509764506),
    f"{_CR4} -> models.model_a_application_time_safe.calibration_slope_intercept_validation.slope",
)
CREDIT_MODEL_A_SLOPE_BENCHMARK = Metric(
    _resolve(_CR4, ("models", "model_a_application_time_safe", "calibration_slope_intercept_benchmark", "slope"), 0.4211399929235683),
    f"{_CR4} -> models.model_a_application_time_safe.calibration_slope_intercept_benchmark.slope",
)
CREDIT_MODEL_B_SLOPE_VALIDATION = Metric(
    _resolve(_CR4, ("models", "model_b_with_lender_grade_signal", "calibration_slope_intercept_validation", "slope"), 0.8509674034286874),
    f"{_CR4} -> models.model_b_with_lender_grade_signal.calibration_slope_intercept_validation.slope",
)
CREDIT_MODEL_B_SLOPE_BENCHMARK = Metric(
    _resolve(_CR4, ("models", "model_b_with_lender_grade_signal", "calibration_slope_intercept_benchmark", "slope"), 0.498136922017927),
    f"{_CR4} -> models.model_b_with_lender_grade_signal.calibration_slope_intercept_benchmark.slope",
)
CREDIT_MODEL_A_PSI_VALIDATION = Metric(
    _resolve(_CR4, ("models", "model_a_application_time_safe", "psi_train_vs_validation"), 0.05082494854210042),
    f"{_CR4} -> models.model_a_application_time_safe.psi_train_vs_validation",
)
CREDIT_MODEL_A_PSI_BENCHMARK = Metric(
    _resolve(_CR4, ("models", "model_a_application_time_safe", "psi_train_vs_benchmark"), 0.06437630914803036),
    f"{_CR4} -> models.model_a_application_time_safe.psi_train_vs_benchmark",
)

# ---------------------------------------------------------------------------
# Test suite status.
# Source: docs/moduleC_milestone4_credit_decisioning.md / docs/decision_log.md
# (pytest output is not persisted as a JSON artifact, so this is a doc-cited
# constant, not an artifact-backed one; re-verify with `pytest -q` if stale.)
# ---------------------------------------------------------------------------
TEST_SUITE_PASSED = Metric(371, "docs/moduleC_milestone4_credit_decisioning.md ('371 passed, 1 skipped')")
TEST_SUITE_SKIPPED = Metric(1, "docs/moduleC_milestone4_credit_decisioning.md ('371 passed, 1 skipped')")
TEST_SUITE_FAILED = Metric(0, "docs/moduleC_milestone4_credit_decisioning.md ('371 passed, 1 skipped')")
