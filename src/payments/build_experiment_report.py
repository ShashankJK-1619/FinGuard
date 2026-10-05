"""Module A, Milestone 2: end-to-end orchestration for the payments
experimentation layer.

Builds on Milestone 1's synthetic session/attempt layer
(src.payments.generate_sessions.build_payments_dataset) -> assigns a
synthetic control/treatment experiment on top of it
(src.payments.experiment_assignment) -> applies a transparent synthetic
treatment-response mechanism (src.payments.experiment_effects) -> runs
experiment-health diagnostics (src.payments.experiment_diagnostics) ->
computes primary/secondary metrics (src.payments.experiment_metrics) ->
runs the two-proportion statistical test + power/MDE planning
(src.payments.experiment_stats / experiment_power) -> evaluates fraud/risk
guardrails and the ship/no-ship decision rule
(src.payments.experiment_guardrails) -> examines pre-specified segment
heterogeneity (src.payments.experiment_segments) -> writes
data/interim/payments_milestone2_experiment_metrics.json.

No ML model is trained anywhere in this script, and no real experiment
occurred — see docs/moduleA_milestone2_experimentation.md.
"""
from __future__ import annotations

import json

from src.data.config import load_paths
from src.payments.config import load_experiment_assumptions, load_synthetic_assumptions
from src.payments.duckdb_views import connect_with_payments_views, register_experiment_sessions_view
from src.payments.experiment_assignment import assign_experiment_arms
from src.payments.experiment_diagnostics import check_no_crossover, compute_pretreatment_covariate_balance, compute_srm_check
from src.payments.experiment_effects import apply_synthetic_treatment_effect
from src.payments.experiment_guardrails import compute_guardrail_metrics, evaluate_decision_rule
from src.payments.experiment_metrics import compute_primary_and_secondary_metrics
from src.payments.experiment_power import compute_achieved_power, compute_mde, compute_mde_table
from src.payments.experiment_segments import compute_all_segment_heterogeneity
from src.payments.experiment_stats import compute_two_proportion_test
from src.payments.generate_sessions import build_payments_dataset
from src.payments.schema import ASSIGNMENT_COL, ASSIGNMENT_CONTROL, ASSIGNMENT_TREATMENT


def main() -> dict:
    paths = load_paths()
    synthetic_assumptions = load_synthetic_assumptions()
    experiment_assumptions = load_experiment_assumptions()

    sessions_df, attempts_df = build_payments_dataset(synthetic_assumptions, paths=paths)
    print(f"payment_sessions (baseline): {len(sessions_df):,} rows")

    # --- B/C: synthetic randomization ---
    assigned_df = assign_experiment_arms(sessions_df, experiment_assumptions)
    # --- E: synthetic treatment-response mechanism ---
    experiment_df = apply_synthetic_treatment_effect(assigned_df, experiment_assumptions)

    # --- J: experiment-health diagnostics (checked BEFORE trusting any metric) ---
    crossover_check = check_no_crossover(experiment_df)
    srm_check = compute_srm_check(experiment_df, experiment_assumptions)
    print("\n== Experiment-health diagnostics ==")
    print(f"  crossover check: {crossover_check}")
    print(f"  SRM check: {srm_check}")
    if srm_check["srm_flag"]:
        print("  WARNING: SRM flagged — downstream results should not be trusted until investigated.")

    con = connect_with_payments_views(sessions_df, attempts_df)
    register_experiment_sessions_view(con, experiment_df)

    covariate_balance_df = compute_pretreatment_covariate_balance(con)
    print("\n== Pre-treatment covariate balance ==")
    print(covariate_balance_df.to_string(index=False))

    # --- D: primary + secondary metrics ---
    metrics_df = compute_primary_and_secondary_metrics(con)
    print("\n== Primary + secondary metrics (per arm) ==")
    print(metrics_df.to_string(index=False))

    indexed_metrics = metrics_df.set_index(ASSIGNMENT_COL)
    n_control = int(indexed_metrics.loc[ASSIGNMENT_CONTROL, "n_sessions"])
    n_completed_control = int(indexed_metrics.loc[ASSIGNMENT_CONTROL, "n_completed"])
    n_treatment = int(indexed_metrics.loc[ASSIGNMENT_TREATMENT, "n_sessions"])
    n_completed_treatment = int(indexed_metrics.loc[ASSIGNMENT_TREATMENT, "n_completed"])

    # --- F: two-proportion test on the primary metric ---
    primary_test_result = compute_two_proportion_test(
        n_control, n_completed_control, n_treatment, n_completed_treatment, experiment_assumptions.alpha
    )
    print("\n== Primary metric: two-proportion test (payment_completion_rate) ==")
    for k, v in primary_test_result.items():
        print(f"  {k}: {v}")

    # --- G: power / MDE planning ---
    baseline_rate = primary_test_result["control_rate"]
    n_per_arm_actual = min(n_control, n_treatment)
    mde_table_df = compute_mde_table(
        baseline_rate, experiment_assumptions.alpha, experiment_assumptions.target_power,
        experiment_assumptions.mde_reference_sample_sizes,
    )
    mde_at_actual_n = compute_mde(
        baseline_rate, n_per_arm_actual, experiment_assumptions.alpha, experiment_assumptions.target_power
    )
    achieved_power = compute_achieved_power(
        n_per_arm_actual, baseline_rate, primary_test_result["treatment_rate"], experiment_assumptions.alpha
    )
    print("\n== Power / MDE reference table (experiment-planning, not proof of effect) ==")
    print(mde_table_df.to_string(index=False))
    print(f"\n  MDE at actual n_per_arm ({n_per_arm_actual:,}): {mde_at_actual_n:.5f}")
    print(f"  Post-hoc achieved power at observed effect: {achieved_power:.4f} (diagnostic only, see docstring)")

    # --- H: guardrails + decision rule ---
    guardrail_df = compute_guardrail_metrics(con, experiment_assumptions)
    print("\n== Guardrail metrics (per arm) ==")
    print(guardrail_df.to_string(index=False))

    decision = evaluate_decision_rule(primary_test_result, guardrail_df, experiment_assumptions)
    print("\n== Ship/no-ship decision ==")
    print(f"  decision: {decision['decision']}")
    for reason in decision["reasons"]:
        print(f"  - {reason}")

    # --- I: pre-specified segment heterogeneity ---
    segment_df = compute_all_segment_heterogeneity(con, experiment_assumptions)
    print("\n== Segment heterogeneity (4 pre-specified dimensions; no multiple-comparison correction — see docstring) ==")
    print(
        segment_df[
            ["segment_dimension", "segment_value", "n_control", "n_treatment", "control_rate", "treatment_rate",
             "absolute_uplift", "p_value", "is_statistically_significant"]
        ].to_string(index=False)
    )

    result = {
        "experiment_assumptions": {
            "experiment_seed": experiment_assumptions.experiment_seed,
            "treatment_allocation_fraction": experiment_assumptions.treatment_allocation_fraction,
            "eligible_min_attempt_count": experiment_assumptions.eligible_min_attempt_count,
            "recovery_uplift_prob": experiment_assumptions.recovery_uplift_prob,
            "alpha": experiment_assumptions.alpha,
            "target_power": experiment_assumptions.target_power,
            "min_practical_uplift_abs": experiment_assumptions.min_practical_uplift_abs,
            "excessive_retry_attempt_threshold": experiment_assumptions.excessive_retry_attempt_threshold,
        },
        "diagnostics": {"crossover_check": crossover_check, "srm_check": srm_check},
        "covariate_balance": covariate_balance_df.to_dict(orient="records"),
        "primary_and_secondary_metrics": metrics_df.to_dict(orient="records"),
        "primary_metric_test": primary_test_result,
        "power_mde": {
            "reference_table": mde_table_df.to_dict(orient="records"),
            "mde_at_actual_n_per_arm": mde_at_actual_n,
            "n_per_arm_actual": n_per_arm_actual,
            "achieved_power_post_hoc": achieved_power,
        },
        "guardrail_metrics": guardrail_df.to_dict(orient="records"),
        "decision": decision,
        "segment_heterogeneity": segment_df.to_dict(orient="records"),
    }

    out_path = paths.interim_dir / "payments_milestone2_experiment_metrics.json"
    paths.interim_dir.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2, default=str)
    print(f"\nWrote {out_path}")

    con.close()
    return result


if __name__ == "__main__":
    main()
