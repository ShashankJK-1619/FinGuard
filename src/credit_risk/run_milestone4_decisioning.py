"""Module C, Milestone 4: credit decisioning, expected loss, and
probability governance, built on top of Milestone 3's Model A / Model B
logistic baselines.

HARD CONSTRAINT (restated here, not just in docs): the target
(`terminal_adverse_outcome`, schema.py) is NOT a 12-month PD. It is the
probability of an eventual adverse terminal outcome among ORIGINATED
loans with sufficient lifecycle resolution. Nothing in this script's
output should be read as a regulatory PD, a CECL/IFRS 9 allowance, or a
production underwriting/decisioning policy -- see
`SELECTION_BIAS_DISCLAIMER` below, which is embedded directly in this
script's own JSON output (Section H), not only in prose docs.

This script does NOT refit anything new and does NOT persist a model
pickle. It reuses Milestone 3's own fit/transform pipeline
(`run_milestone3_baseline._fit_and_evaluate_one_model`,
`_labeled_only`) to regenerate Model A/B's predicted probabilities on the
same frozen primary chronological split, then layers Milestone 4's new,
additive analysis on top:
  - Section A: probability-quality review (reliability bins already
    computed by Milestone 3's script are reused; this script adds the Cox
    calibration slope/intercept).
  - Section B: risk bands, boundaries fit on TRAIN ONLY.
  - Section C: illustrative expected loss (`EL = p * LGD * EAD`).
  - Section D: Model A vs Model B expected-loss comparison on benchmark.
  - Section E-G: illustrative three-way decisioning + cost framework +
    sensitivity analysis.
  - Section I: vintage score distribution / calibration-by-vintage / PSI
    drift diagnostics, kept separate from Milestone 3's frozen
    `vintage_analysis.py`.

Run: .venv/bin/python -m src.credit_risk.run_milestone4_decisioning
Writes: data/interim/credit_risk_milestone4_metrics.json (reproducibility
artifact; the prose report is
docs/moduleC_milestone4_credit_decisioning.md).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from src.credit_risk.calibration import calibration_slope_intercept
from src.credit_risk.chronological_split import (
    BENCHMARK,
    EXCLUDED_TOO_RECENT,
    TRAIN,
    VALIDATION,
    build_chronological_split,
    load_split_design,
)
from src.credit_risk.decision_cost_model import (
    evaluate_decision_policy as evaluate_credit_decision_policy,
)
from src.credit_risk.decision_cost_model import (
    load_credit_decision_cost_assumptions,
    sensitivity_analysis as decision_cost_sensitivity_analysis,
)
from src.credit_risk.decisioning import classify_decisions
from src.credit_risk.expected_loss import EAD_FIELD, compute_expected_loss, load_expected_loss_assumptions
from src.credit_risk.feature_loader import load_modeling_table
from src.credit_risk.feature_selection import load_feature_sets
from src.credit_risk.modeling import predict_probability
from src.credit_risk.raw_loader import get_connection, register_loans_view
from src.credit_risk.risk_bands import assign_risk_band, fit_risk_band_boundaries, summarize_risk_bands
from src.credit_risk.run_milestone3_baseline import _fit_and_evaluate_one_model, _labeled_only
from src.credit_risk.schema import ORIGINATION_DATE_COL, TERMINAL_ADVERSE_OUTCOME_LABEL_COL
from src.credit_risk.score_stability import (
    adverse_outcome_rate_by_band_over_time,
    calibration_by_vintage,
    compute_psi,
    psi_drift_category,
    score_distribution_by_vintage,
)
from src.credit_risk.target_construction import compute_terminal_adverse_outcome_target

PROJECT_ROOT = Path(__file__).resolve().parents[2]
METRICS_OUT_PATH = PROJECT_ROOT / "data" / "interim" / "credit_risk_milestone4_metrics.json"

SELECTION_BIAS_DISCLAIMER = (
    "This dataset ONLY observes loans that LendingClub actually originated. "
    "Rejected applicants have no repayment outcome recorded anywhere in "
    "this data, so nothing in this script -- probability estimates, risk "
    "bands, expected loss, or the illustrative decisioning/cost framework "
    "-- can validate what would happen to the FULL applicant population "
    "under a changed approval policy. Reject inference (estimating "
    "outcomes for applicants who were never funded) is explicitly out of "
    "scope for this milestone."
)

TARGET_FRAMING_NOTE = (
    "terminal_adverse_outcome is NOT a 12-month probability of default. It "
    "is the probability of an eventual adverse terminal outcome among "
    "ORIGINATED loans with sufficient lifecycle resolution, over each "
    "loan's own variable (36- or 60-month) maturity horizon."
)

# Section G sensitivity grid: illustrative-only variations layered on the
# benchmark split's fitted risk-band boundaries.
COST_SENSITIVITY_VARIATION_LABELS_NOTE = (
    "Each variation changes exactly one assumption (lgd, a threshold, or a "
    "flat cost) relative to the base case; none of these rows is reported "
    "as 'optimal' -- see decision_cost_model.sensitivity_analysis."
)


def _prob_and_y(fp, fitted, labeled_df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    X = fp.transform(labeled_df)
    prob = predict_probability(fitted, X)
    y = labeled_df[TERMINAL_ADVERSE_OUTCOME_LABEL_COL].astype(int).to_numpy()
    return prob, y


def _vintage_frame(
    fp, fitted, boundaries, buckets: dict[str, pd.DataFrame]
) -> pd.DataFrame:
    frames = []
    for bucket_name, labeled_df in buckets.items():
        prob, y = _prob_and_y(fp, fitted, labeled_df)
        band = assign_risk_band(prob, boundaries)
        frames.append(
            pd.DataFrame(
                {
                    ORIGINATION_DATE_COL: labeled_df[ORIGINATION_DATE_COL].to_numpy(),
                    "origination_year": pd.to_datetime(labeled_df[ORIGINATION_DATE_COL]).dt.year.to_numpy(),
                    "split_bucket": bucket_name,
                    "y_true": y,
                    "y_prob": prob,
                    "risk_band": band,
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


def _analyze_one_model(
    model_name: str,
    fp,
    fitted,
    prob_train: np.ndarray,
    y_train: np.ndarray,
    train_labeled: pd.DataFrame,
    validation_labeled: pd.DataFrame,
    benchmark_labeled: pd.DataFrame,
    excluded_labeled: pd.DataFrame,
    lgd: float,
) -> dict:
    prob_validation, y_validation = _prob_and_y(fp, fitted, validation_labeled)
    prob_benchmark, y_benchmark = _prob_and_y(fp, fitted, benchmark_labeled)

    ead_train = train_labeled[EAD_FIELD].astype(float).to_numpy()
    ead_validation = validation_labeled[EAD_FIELD].astype(float).to_numpy()
    ead_benchmark = benchmark_labeled[EAD_FIELD].astype(float).to_numpy()

    # Section A: Cox calibration slope/intercept, out-of-sample only.
    calib_slope_intercept_validation = calibration_slope_intercept(
        model_name, VALIDATION, y_validation, prob_validation
    )
    calib_slope_intercept_benchmark = calibration_slope_intercept(
        model_name, BENCHMARK, y_benchmark, prob_benchmark
    )

    # Section B: risk-band boundaries fit on TRAIN ONLY, applied elsewhere.
    boundaries = fit_risk_band_boundaries(model_name, prob_train)
    band_validation = assign_risk_band(prob_validation, boundaries)
    band_benchmark = assign_risk_band(prob_benchmark, boundaries)
    risk_band_summary_validation = summarize_risk_bands(y_validation, prob_validation, band_validation)
    risk_band_summary_benchmark = summarize_risk_bands(y_benchmark, prob_benchmark, band_benchmark)

    # Section C: illustrative expected loss.
    el_validation = compute_expected_loss(prob_validation, ead_validation, lgd)
    el_benchmark = compute_expected_loss(prob_benchmark, ead_benchmark, lgd)

    # Section E: decisioning policy reuses the SAME fitted boundaries as
    # the review/decline threshold pair (low/medium -> review,
    # medium/high -> decline) -- one development-only-fit threshold pair,
    # not two independently chosen ones.
    decisions_benchmark = classify_decisions(
        prob_benchmark, boundaries.low_medium_boundary, boundaries.medium_high_boundary
    )

    # Section I: vintage diagnostics across ALL labeled buckets.
    vintage_df = _vintage_frame(
        fp,
        fitted,
        boundaries,
        {
            TRAIN: train_labeled,
            VALIDATION: validation_labeled,
            BENCHMARK: benchmark_labeled,
            EXCLUDED_TOO_RECENT: excluded_labeled,
        },
    )
    score_dist_by_vintage = score_distribution_by_vintage(
        vintage_df["origination_year"].to_numpy(), vintage_df["y_prob"].to_numpy()
    )
    calib_by_vintage = calibration_by_vintage(
        vintage_df["origination_year"].to_numpy(), vintage_df["y_true"].to_numpy(), vintage_df["y_prob"].to_numpy()
    )
    adverse_rate_by_band_over_time = adverse_outcome_rate_by_band_over_time(
        vintage_df["origination_year"].to_numpy(), vintage_df["risk_band"].to_numpy(), vintage_df["y_true"].to_numpy()
    )
    psi_train_vs_validation = compute_psi(prob_train, prob_validation)
    psi_train_vs_benchmark = compute_psi(prob_train, prob_benchmark)

    return {
        "model_name": model_name,
        "boundaries": boundaries,
        "prob_validation": prob_validation,
        "prob_benchmark": prob_benchmark,
        "y_validation": y_validation,
        "y_benchmark": y_benchmark,
        "ead_benchmark": ead_benchmark,
        "el_validation": el_validation,
        "el_benchmark": el_benchmark,
        "decisions_benchmark": decisions_benchmark,
        "band_benchmark": band_benchmark,
        "calib_slope_intercept_validation": calib_slope_intercept_validation,
        "calib_slope_intercept_benchmark": calib_slope_intercept_benchmark,
        "risk_band_summary_validation": risk_band_summary_validation,
        "risk_band_summary_benchmark": risk_band_summary_benchmark,
        "score_dist_by_vintage": score_dist_by_vintage,
        "calib_by_vintage": calib_by_vintage,
        "adverse_rate_by_band_over_time": adverse_rate_by_band_over_time,
        "psi_train_vs_validation": psi_train_vs_validation,
        "psi_train_vs_benchmark": psi_train_vs_benchmark,
    }


def _ead_by_band_mean_el(el: np.ndarray, band: np.ndarray) -> dict:
    out = {}
    for b in ("low", "medium", "high"):
        mask = band == b
        out[b] = float(el[mask].mean()) if mask.any() else None
    return out


def main() -> int:
    con = get_connection()
    register_loans_view(con)

    model_a, model_b = load_feature_sets()
    design = load_split_design()
    expected_loss_assumptions = load_expected_loss_assumptions()
    decision_costs = load_credit_decision_cost_assumptions()
    lgd = expected_loss_assumptions.lgd

    print("=== Loading real data (Model B's superset of raw columns) ===")
    raw_df = load_modeling_table(con, model_b)
    print(f"Loaded {len(raw_df):,} real rows")

    with_target = compute_terminal_adverse_outcome_target(raw_df, data_as_of_date=design.data_as_of_date)
    with_split, split_summary = build_chronological_split(with_target, design)

    train_all = with_split[with_split["split_bucket"] == TRAIN]
    validation_all = with_split[with_split["split_bucket"] == VALIDATION]
    benchmark_all = with_split[with_split["split_bucket"] == BENCHMARK]
    excluded_all = with_split[with_split["split_bucket"] == EXCLUDED_TOO_RECENT]

    train_labeled = _labeled_only(train_all, design.label_status_column)
    validation_labeled = _labeled_only(validation_all, design.label_status_column)
    benchmark_labeled = _labeled_only(benchmark_all, design.label_status_column)
    excluded_labeled = _labeled_only(excluded_all, design.label_status_column)

    print(f"\n=== Illustrative LGD = {lgd}; EAD field = {EAD_FIELD} ===")

    analyses = {}
    for fs in (model_a, model_b):
        print(f"\n=== Re-fitting {fs.name} (Milestone 3 pipeline, not refit-tuned) ===")
        m3_result = _fit_and_evaluate_one_model(fs, train_labeled, validation_labeled, benchmark_labeled)
        analysis = _analyze_one_model(
            fs.name,
            m3_result["fitted_preprocessor"],
            m3_result["fitted_model"],
            m3_result["prob_train"],
            m3_result["y_train"],
            train_labeled,
            validation_labeled,
            benchmark_labeled,
            excluded_labeled,
            lgd,
        )
        analyses[fs.name] = analysis

        print(f"\n--- Section A: calibration slope/intercept ({fs.name}) ---")
        print("validation:", analysis["calib_slope_intercept_validation"])
        print("benchmark:", analysis["calib_slope_intercept_benchmark"])

        print(f"\n--- Section B: risk bands ({fs.name}, benchmark) ---")
        print(analysis["risk_band_summary_benchmark"].to_string(index=False))

        print(f"\n--- Section I: PSI (train vs validation / benchmark, {fs.name}) ---")
        print(
            f"train_vs_validation={analysis['psi_train_vs_validation']:.4f} "
            f"({psi_drift_category(analysis['psi_train_vs_validation'])}); "
            f"train_vs_benchmark={analysis['psi_train_vs_benchmark']:.4f} "
            f"({psi_drift_category(analysis['psi_train_vs_benchmark'])})"
        )

    # Section D: Model A vs Model B expected-loss comparison on benchmark.
    a = analyses[model_a.name]
    b = analyses[model_b.name]
    avg_el_a = float(a["el_benchmark"].mean())
    avg_el_b = float(b["el_benchmark"].mean())
    el_by_band_a = _ead_by_band_mean_el(a["el_benchmark"], a["band_benchmark"])
    el_by_band_b = _ead_by_band_mean_el(b["el_benchmark"], b["band_benchmark"])
    ranking_corr = spearmanr(a["el_benchmark"], b["el_benchmark"])
    section_d = {
        "split": BENCHMARK,
        "lgd": lgd,
        "ead_field": EAD_FIELD,
        "avg_expected_loss_model_a": avg_el_a,
        "avg_expected_loss_model_b": avg_el_b,
        "avg_expected_loss_difference_b_minus_a": avg_el_b - avg_el_a,
        "expected_loss_by_band_model_a": el_by_band_a,
        "expected_loss_by_band_model_b": el_by_band_b,
        "ranking_consistency_spearman_correlation": float(ranking_corr.statistic),
        "ranking_consistency_spearman_pvalue": float(ranking_corr.pvalue),
        "framing_note": (
            "This compares economic risk-ranking, not deployment readiness -- "
            "Model B is NOT recommended for deployment (see "
            "docs/credit_risk_model_feature_sets.md)."
        ),
    }
    print("\n=== Section D: Model A vs Model B expected loss (benchmark) ===")
    print(json.dumps(section_d, indent=2))

    # Section F/G: decisioning cost framework + sensitivity, Model A on benchmark.
    base_review_threshold = a["boundaries"].low_medium_boundary
    base_decline_threshold = a["boundaries"].medium_high_boundary
    base_policy_result = evaluate_credit_decision_policy(
        a["y_benchmark"],
        a["prob_benchmark"],
        a["ead_benchmark"],
        lgd,
        base_review_threshold,
        base_decline_threshold,
        decision_costs,
    )
    print("\n=== Section F: base decisioning-cost policy (Model A, benchmark) ===")
    print(json.dumps(base_policy_result, indent=2))

    sensitivity_variations = [
        {"label": "base"},
        {"label": "lgd_low", "lgd": max(lgd - 0.2, 0.0)},
        {"label": "lgd_high", "lgd": min(lgd + 0.15, 1.0)},
        {"label": "stricter_decline", "decline_threshold": max(base_decline_threshold - 0.05, base_review_threshold + 0.01)},
        {"label": "looser_decline", "decline_threshold": min(base_decline_threshold + 0.05, 0.99)},
        {"label": "costlier_review", "cost_per_manual_review": decision_costs.cost_per_manual_review * 3.0},
        {
            "label": "costlier_opportunity",
            "opportunity_cost_per_good_loan_declined": decision_costs.opportunity_cost_per_good_loan_declined * 3.0,
        },
    ]
    sensitivity_table = decision_cost_sensitivity_analysis(
        a["y_benchmark"],
        a["prob_benchmark"],
        a["ead_benchmark"],
        base_lgd=lgd,
        base_review_threshold=base_review_threshold,
        base_decline_threshold=base_decline_threshold,
        base_costs=decision_costs,
        variations=sensitivity_variations,
    )
    print("\n=== Section G: threshold/assumption sensitivity (Model A, benchmark) ===")
    print(sensitivity_table.to_string(index=False))

    # --- Persist reproducibility artifact ---
    out = {
        "target_framing_note": TARGET_FRAMING_NOTE,
        "selection_bias_disclaimer": SELECTION_BIAS_DISCLAIMER,
        "expected_loss_assumptions": {"lgd": lgd, "ead_field": EAD_FIELD},
        "decision_cost_assumptions": {
            "cost_per_manual_review": decision_costs.cost_per_manual_review,
            "opportunity_cost_per_good_loan_declined": decision_costs.opportunity_cost_per_good_loan_declined,
        },
        "models": {
            name: {
                "risk_band_boundaries": {
                    "low_medium_boundary": r["boundaries"].low_medium_boundary,
                    "medium_high_boundary": r["boundaries"].medium_high_boundary,
                    "n_train": r["boundaries"].n_train,
                },
                "calibration_slope_intercept_validation": r["calib_slope_intercept_validation"].__dict__,
                "calibration_slope_intercept_benchmark": r["calib_slope_intercept_benchmark"].__dict__,
                "risk_band_summary_validation": r["risk_band_summary_validation"].to_dict(orient="records"),
                "risk_band_summary_benchmark": r["risk_band_summary_benchmark"].to_dict(orient="records"),
                "avg_expected_loss_validation": float(r["el_validation"].mean()),
                "avg_expected_loss_benchmark": float(r["el_benchmark"].mean()),
                "score_distribution_by_vintage": r["score_dist_by_vintage"].to_dict(orient="records"),
                "calibration_by_vintage": r["calib_by_vintage"].to_dict(orient="records"),
                "adverse_outcome_rate_by_band_over_time": r["adverse_rate_by_band_over_time"].to_dict(
                    orient="records"
                ),
                "psi_train_vs_validation": r["psi_train_vs_validation"],
                "psi_train_vs_validation_category": psi_drift_category(r["psi_train_vs_validation"]),
                "psi_train_vs_benchmark": r["psi_train_vs_benchmark"],
                "psi_train_vs_benchmark_category": psi_drift_category(r["psi_train_vs_benchmark"]),
            }
            for name, r in analyses.items()
        },
        "model_a_vs_model_b_expected_loss_comparison": section_d,
        "decisioning_base_policy_model_a_benchmark": base_policy_result,
        "decisioning_sensitivity_model_a_benchmark": sensitivity_table.to_dict(orient="records"),
    }
    METRICS_OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(METRICS_OUT_PATH, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\nWrote {METRICS_OUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
