"""Module A, Milestone 1: end-to-end orchestration.

Loads real IEEE-CIS transactions -> generates the synthetic session/attempt
layer -> writes both as Parquet -> runs the SQL-first KPI/funnel/retry-
recovery/segmentation/guardrail analyses -> writes
data/interim/payments_milestone1_metrics.json. No ML model is trained
anywhere in this script. See docs/payments_data_model.md for methodology.
"""
from __future__ import annotations

import json

from src.data.config import load_paths
from src.payments.config import load_guardrail_rules, load_synthetic_assumptions
from src.payments.duckdb_views import connect_with_payments_views
from src.payments.funnel import compute_funnel
from src.payments.generate_sessions import build_payments_dataset
from src.payments.guardrails import backtest_guardrails_against_fraud_label, compute_guardrail_flags
from src.payments.kpi import compute_overall_kpis
from src.payments.retry_recovery import compute_retry_recovery
from src.payments.segmentation import compute_all_segments


def main() -> dict:
    paths = load_paths()
    assumptions = load_synthetic_assumptions()
    rules = load_guardrail_rules()

    sessions_df, attempts_df = build_payments_dataset(assumptions, paths=paths)
    print(f"payment_sessions: {len(sessions_df):,} rows ({sessions_df['is_synthetic_abandonment'].sum():,} synthetic-abandonment)")
    print(f"payment_attempts: {len(attempts_df):,} rows")

    con = connect_with_payments_views(sessions_df, attempts_df)

    overall_kpis = compute_overall_kpis(con)
    print("\n== Overall KPIs ==")
    for k, v in overall_kpis.items():
        print(f"  {k}: {v}")

    funnel_df = compute_funnel(con)
    print("\n== Funnel ==")
    print(funnel_df.to_string(index=False))

    retry_recovery_df = compute_retry_recovery(con)
    print("\n== Retry / recovery ==")
    print(retry_recovery_df.to_string(index=False))

    segments_df = compute_all_segments(con)
    print("\n== Segmentation (head) ==")
    print(segments_df.head(10).to_string(index=False))

    flags_df = compute_guardrail_flags(con, rules)
    backtest_df = backtest_guardrails_against_fraud_label(flags_df)
    print("\n== Guardrail backtest (real-linked sessions only) ==")
    print(backtest_df.to_string(index=False))

    result = {
        "assumptions": {
            "random_seed": assumptions.random_seed,
            "abandonment_rate": assumptions.abandonment_rate,
            "decline_before_success_probs": list(assumptions.decline_before_success_probs),
            "abandonment_attempt_count_probs": list(assumptions.abandonment_attempt_count_probs),
            "decline_reason_weights": assumptions.decline_reason_weights,
        },
        "guardrail_rules": {
            "velocity_attempt_threshold": rules.velocity_attempt_threshold,
            "high_amount_percentile": rules.high_amount_percentile,
        },
        "overall_kpis": overall_kpis,
        "funnel": funnel_df.to_dict(orient="records"),
        "retry_recovery": retry_recovery_df.to_dict(orient="records"),
        "segmentation": segments_df.to_dict(orient="records"),
        "guardrail_backtest": backtest_df.to_dict(orient="records"),
    }

    out_path = paths.interim_dir / "payments_milestone1_metrics.json"
    paths.interim_dir.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2, default=str)
    print(f"\nWrote {out_path}")

    con.close()
    return result


if __name__ == "__main__":
    main()
