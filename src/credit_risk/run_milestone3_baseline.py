"""Module C, Milestone 3: credit-risk baseline logistic regression,
Model A vs Model B, on the PRIMARY chronological split
(`terminal_adverse_outcome`).

Ties together every new Milestone 3 module (feature_selection,
feature_loader, preprocessing, modeling, evaluation, calibration,
interpretation, ablation, vintage_analysis, subgroup_diagnostics) into one
reproducible run against the real LendingClub file. Fits each model
EXACTLY ONCE on the train split; validation/benchmark/excluded_too_recent
are only ever transformed/scored, never refit on.

Run: .venv/bin/python -m src.credit_risk.run_milestone3_baseline
Writes: data/interim/credit_risk_milestone3_metrics.json (reproducibility
artifact; the prose report is docs/moduleC_milestone3_credit_baseline.md).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.credit_risk.ablation import compute_ablation
from src.credit_risk.calibration import compare_calibration, fit_calibrated_variants, reliability_curve
from src.credit_risk.chronological_split import (
    BENCHMARK,
    EXCLUDED_TOO_RECENT,
    TRAIN,
    VALIDATION,
    build_chronological_split,
    load_split_design,
)
from src.credit_risk.evaluation import compute_classification_metrics
from src.credit_risk.feature_loader import feature_matrix_columns, load_modeling_table
from src.credit_risk.feature_selection import FeatureSet, load_feature_sets
from src.credit_risk.interpretation import extract_coefficients, reference_levels_from_onehot, top_coefficients
from src.credit_risk.modeling import fit_logistic_regression, predict_probability
from src.credit_risk.preprocessing import FittedPreprocessor, fit_preprocessing_pipeline
from src.credit_risk.raw_loader import get_connection, register_loans_view
from src.credit_risk.schema import (
    LABEL_STATUS_CENSORED,
    ORIGINATION_DATE_COL,
    TERMINAL_ADVERSE_OUTCOME_LABEL_COL,
)
from src.credit_risk.subgroup_diagnostics import (
    NON_PROTECTED_SUBGROUP_FIELDS,
    compute_subgroup_diagnostics,
    row_missingness_rate,
)
from src.credit_risk.target_construction import compute_terminal_adverse_outcome_target
from src.credit_risk.vintage_analysis import compute_vintage_stability

PROJECT_ROOT = Path(__file__).resolve().parents[2]
METRICS_OUT_PATH = PROJECT_ROOT / "data" / "interim" / "credit_risk_milestone3_metrics.json"

THRESHOLD_NAME = "train_prevalence_matched"


def _labeled_only(df: pd.DataFrame, label_status_column: str) -> pd.DataFrame:
    return df[df[label_status_column] != LABEL_STATUS_CENSORED].copy()


def _fit_and_evaluate_one_model(
    feature_set: FeatureSet,
    train_labeled: pd.DataFrame,
    validation_labeled: pd.DataFrame,
    benchmark_labeled: pd.DataFrame,
) -> dict:
    numeric_cols, categorical_cols = feature_matrix_columns(feature_set)
    target_col = TERMINAL_ADVERSE_OUTCOME_LABEL_COL

    fp: FittedPreprocessor = fit_preprocessing_pipeline(train_labeled, numeric_cols, categorical_cols)
    X_train = fp.transform(train_labeled)
    X_validation = fp.transform(validation_labeled)
    X_benchmark = fp.transform(benchmark_labeled)

    y_train = train_labeled[target_col].astype(int).to_numpy()
    y_validation = validation_labeled[target_col].astype(int).to_numpy()
    y_benchmark = benchmark_labeled[target_col].astype(int).to_numpy()

    fitted = fit_logistic_regression(X_train, y_train, fp.output_feature_names, name=feature_set.name)

    threshold_value = float(y_train.mean())  # Section E: named, train-prevalence-matched threshold

    prob_train = predict_probability(fitted, X_train)
    prob_validation = predict_probability(fitted, X_validation)
    prob_benchmark = predict_probability(fitted, X_benchmark)

    validation_metrics = compute_classification_metrics(
        y_validation, prob_validation, threshold_value, THRESHOLD_NAME, VALIDATION, feature_set.name
    )
    benchmark_metrics = compute_classification_metrics(
        y_benchmark, prob_benchmark, threshold_value, THRESHOLD_NAME, BENCHMARK, feature_set.name
    )

    # Section F: calibration, fit entirely within train via 5-fold CV, assessed on validation.
    sigmoid_model, isotonic_model = fit_calibrated_variants(fitted.model, X_train, y_train)
    sigmoid_prob_validation = sigmoid_model.predict_proba(X_validation)[:, 1]
    isotonic_prob_validation = isotonic_model.predict_proba(X_validation)[:, 1]
    calibration_comparison = compare_calibration(
        feature_set.name,
        n_train=len(y_train),
        y_validation=y_validation,
        uncalibrated_prob_validation=prob_validation,
        sigmoid_prob_validation=sigmoid_prob_validation,
        isotonic_prob_validation=isotonic_prob_validation,
    )
    chosen_prob_validation = (
        sigmoid_prob_validation if calibration_comparison.chosen_method == "sigmoid" else isotonic_prob_validation
    )
    reliability_uncalibrated = reliability_curve(y_validation, prob_validation)
    reliability_calibrated = reliability_curve(y_validation, chosen_prob_validation)

    categorical_onehot = fp.column_transformer.named_transformers_["categorical"].named_steps["onehot"]
    reference_levels = reference_levels_from_onehot(categorical_cols, categorical_onehot)
    coefficient_table = extract_coefficients(fitted, reference_levels)

    return {
        "feature_set": feature_set,
        "fitted_preprocessor": fp,
        "fitted_model": fitted,
        "threshold_value": threshold_value,
        "y_train": y_train,
        "prob_train": prob_train,
        "validation_metrics": validation_metrics,
        "benchmark_metrics": benchmark_metrics,
        "calibration_comparison": calibration_comparison,
        "reliability_uncalibrated": reliability_uncalibrated,
        "reliability_calibrated": reliability_calibrated,
        "coefficient_table": coefficient_table,
    }


def main() -> int:
    con = get_connection()
    register_loans_view(con)

    model_a, model_b = load_feature_sets()
    design = load_split_design()  # default = PRIMARY terminal_adverse_outcome split

    print("=== Loading real data (Model B's superset of raw columns) ===")
    raw_df = load_modeling_table(con, model_b)
    print(f"Loaded {len(raw_df):,} real rows")

    with_target = compute_terminal_adverse_outcome_target(raw_df, data_as_of_date=design.data_as_of_date)
    with_split, split_summary = build_chronological_split(with_target, design)

    print("\n=== Section A: primary split summary (frozen, unaltered) ===")
    print(split_summary.to_string(index=False))

    train_all = with_split[with_split["split_bucket"] == TRAIN]
    validation_all = with_split[with_split["split_bucket"] == VALIDATION]
    benchmark_all = with_split[with_split["split_bucket"] == BENCHMARK]
    excluded_all = with_split[with_split["split_bucket"] == EXCLUDED_TOO_RECENT]

    train_labeled = _labeled_only(train_all, design.label_status_column)
    validation_labeled = _labeled_only(validation_all, design.label_status_column)
    benchmark_labeled = _labeled_only(benchmark_all, design.label_status_column)
    excluded_labeled = _labeled_only(excluded_all, design.label_status_column)

    results = {}
    for fs in (model_a, model_b):
        print(f"\n=== Fitting {fs.name} ===")
        results[fs.name] = _fit_and_evaluate_one_model(
            fs, train_labeled, validation_labeled, benchmark_labeled
        )
        print("Validation metrics:", json.dumps(results[fs.name]["validation_metrics"].as_dict(), indent=2))
        print("Benchmark metrics:", json.dumps(results[fs.name]["benchmark_metrics"].as_dict(), indent=2))
        cc = results[fs.name]["calibration_comparison"]
        print(
            "Section F calibration (fit on train, assessed on validation):",
            json.dumps(
                {
                    "n_train": cc.n_train,
                    "uncalibrated_brier_validation": cc.uncalibrated_brier_validation,
                    "sigmoid_brier_validation": cc.sigmoid_brier_validation,
                    "isotonic_brier_validation": cc.isotonic_brier_validation,
                    "chosen_method": cc.chosen_method,
                    "chosen_method_reason": cc.chosen_method_reason,
                },
                indent=2,
            ),
        )

    # Section G: Model A coefficient interpretation + Model B's grade/sub_grade/int_rate rows
    coef_a = results[model_a.name]["coefficient_table"]
    top_a = top_coefficients(coef_a, n=10)
    print("\n=== Section G: Model A top-10 |coefficient| ===")
    print(top_a.to_string(index=False))

    coef_b = results[model_b.name]["coefficient_table"]
    grade_rows = coef_b.table[
        coef_b.table["feature"].str.startswith(("grade_", "sub_grade_", "int_rate"))
    ]
    print("\n=== Section G: Model B grade/sub_grade/int_rate coefficients ===")
    print(grade_rows.to_string(index=False))

    # Section H: ablation on validation + benchmark
    ablation_validation = compute_ablation(
        VALIDATION, results[model_a.name]["validation_metrics"], results[model_b.name]["validation_metrics"]
    )
    ablation_benchmark = compute_ablation(
        BENCHMARK, results[model_a.name]["benchmark_metrics"], results[model_b.name]["benchmark_metrics"]
    )
    print("\n=== Section H: ablation (validation) ===")
    print(json.dumps(ablation_validation.as_dict(), indent=2))
    print("\n=== Section H: ablation (benchmark) ===")
    print(json.dumps(ablation_benchmark.as_dict(), indent=2))

    # Section I: vintage/stability analysis using Model A, across ALL labeled origination years
    model_a_result = results[model_a.name]
    fp_a = model_a_result["fitted_preprocessor"]
    fitted_a = model_a_result["fitted_model"]

    vintage_frames = []
    for bucket_name, labeled_df in (
        (TRAIN, train_labeled),
        (VALIDATION, validation_labeled),
        (BENCHMARK, benchmark_labeled),
        (EXCLUDED_TOO_RECENT, excluded_labeled),
    ):
        X = fp_a.transform(labeled_df)
        prob = predict_probability(fitted_a, X)
        vintage_frames.append(
            pd.DataFrame(
                {
                    ORIGINATION_DATE_COL: labeled_df[ORIGINATION_DATE_COL].to_numpy(),
                    "split_bucket": bucket_name,
                    "y_true": labeled_df[TERMINAL_ADVERSE_OUTCOME_LABEL_COL].astype(int).to_numpy(),
                    "y_prob": prob,
                }
            )
        )
    vintage_input = pd.concat(vintage_frames, ignore_index=True)
    vintage_stability = compute_vintage_stability(
        vintage_input[ORIGINATION_DATE_COL], vintage_input["split_bucket"], vintage_input["y_true"], vintage_input["y_prob"]
    )
    print("\n=== Section I: vintage/stability (Model A, descriptive only) ===")
    print(vintage_stability.to_string(index=False))

    # Section J: subgroup/governance diagnostics on validation, Model A
    numeric_cols_a, _ = feature_matrix_columns(model_a)
    missingness_val = row_missingness_rate(validation_labeled, numeric_cols_a)
    prob_validation_a = predict_probability(
        fitted_a, fp_a.transform(validation_labeled)
    )
    subgroup_tables = {}
    for field in NON_PROTECTED_SUBGROUP_FIELDS:
        subgroup_tables[field] = compute_subgroup_diagnostics(
            validation_labeled,
            field,
            validation_labeled[TERMINAL_ADVERSE_OUTCOME_LABEL_COL].astype(int).to_numpy(),
            prob_validation_a,
            model_a_result["threshold_value"],
            missingness_val,
        )
        print(f"\n=== Section J: subgroup diagnostics -- {field} (Model A, validation) ===")
        print(subgroup_tables[field].to_string(index=False))

    # --- Persist reproducibility artifact ---
    out = {
        "split_summary": split_summary.to_dict(orient="records"),
        "models": {
            name: {
                "validation_metrics": r["validation_metrics"].as_dict(),
                "benchmark_metrics": r["benchmark_metrics"].as_dict(),
                "threshold_value": r["threshold_value"],
                "calibration": {
                    "n_train": r["calibration_comparison"].n_train,
                    "uncalibrated_brier_validation": r["calibration_comparison"].uncalibrated_brier_validation,
                    "sigmoid_brier_validation": r["calibration_comparison"].sigmoid_brier_validation,
                    "isotonic_brier_validation": r["calibration_comparison"].isotonic_brier_validation,
                    "chosen_method": r["calibration_comparison"].chosen_method,
                    "chosen_method_reason": r["calibration_comparison"].chosen_method_reason,
                },
                "reliability_curve_uncalibrated": [b.__dict__ for b in r["reliability_uncalibrated"]],
                "reliability_curve_calibrated": [b.__dict__ for b in r["reliability_calibrated"]],
                "coefficients_top20": r["coefficient_table"].table.head(20).to_dict(orient="records"),
                "intercept": r["coefficient_table"].intercept,
            }
            for name, r in results.items()
        },
        "ablation": {
            VALIDATION: ablation_validation.as_dict(),
            BENCHMARK: ablation_benchmark.as_dict(),
        },
        "vintage_stability_model_a": vintage_stability.to_dict(orient="records"),
        "subgroup_diagnostics_model_a_validation": {
            field: table.to_dict(orient="records") for field, table in subgroup_tables.items()
        },
    }
    METRICS_OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(METRICS_OUT_PATH, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\nWrote {METRICS_OUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
