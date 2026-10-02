from __future__ import annotations

import numpy as np
import pytest

from src.models.schema import TARGET_COL
from src.models.split import chronological_split, chronological_split_two_way
from src.models.train_calibrated import (
    CALIBRATION_FIT_FRAC_OF_VALIDATION,
    DECLINE_THRESHOLD_GRID,
    REVIEW_THRESHOLD_GRID,
    build_cost_variations,
    calibration_comparison,
    fit_baseline_pipeline,
    select_best_calibration_method,
    time_feature_ablation,
)
from src.models.cost_model import load_cost_assumptions
from tests.conftest import requires_baseline_features
from tests.models.test_train_baseline import _toy_model_df


def _toy_splits():
    """Builds the full train / cal_fit / selection / temporal_benchmark
    chain used by the corrected Milestone 4 pipeline, out of the toy model
    table, mirroring what src.models.train_calibrated.main() does on real
    data. Uses a larger-than-default toy table (n=3000, not 400) because
    quartering an already-10%-fraud-rate table can otherwise leave one
    quarter (e.g. the `selection` split) with zero fraud rows by chance,
    which makes ROC-AUC undefined there — not a bug in the code under
    test, just an artifact of too few rows."""
    df = _toy_model_df(n=3000)
    train_df, val_df, temporal_benchmark_df, _ = chronological_split(df, train_frac=0.6, val_frac=0.2)
    cal_fit_df, select_df, _ = chronological_split_two_way(val_df, first_frac=CALIBRATION_FIT_FRAC_OF_VALIDATION)
    return train_df, cal_fit_df, select_df, temporal_benchmark_df


def test_calibration_comparison_returns_all_methods_both_splits():
    train_df, cal_fit_df, select_df, temporal_benchmark_df = _toy_splits()
    pipeline = fit_baseline_pipeline(train_df)
    y_cal_fit = cal_fit_df[TARGET_COL].to_numpy()
    y_select = select_df[TARGET_COL].to_numpy()
    y_temporal_benchmark = temporal_benchmark_df[TARGET_COL].to_numpy()
    raw_cal_fit = pipeline.predict_proba(cal_fit_df)[:, 1]
    raw_select = pipeline.predict_proba(select_df)[:, 1]
    raw_temporal_benchmark = pipeline.predict_proba(temporal_benchmark_df)[:, 1]

    result = calibration_comparison(
        y_cal_fit, raw_cal_fit, y_select, raw_select, y_temporal_benchmark, raw_temporal_benchmark
    )

    assert set(result) == {"uncalibrated", "sigmoid", "isotonic"}
    for method in result:
        assert set(result[method]) == {"selection", "temporal_benchmark"}
        for split in ("selection", "temporal_benchmark"):
            assert "roc_auc" in result[method][split]
            assert "reliability_table" in result[method][split]


def test_calibration_comparison_selection_results_unaffected_by_temporal_benchmark_mutation():
    """No future leakage: calibrators are fit on (y_cal_fit, raw_cal_fit)
    only, and `selection` metrics are computed independently of whatever is
    passed as the temporal benchmark. Mutating temporal-benchmark-side
    inputs must not change any selection-side number."""
    train_df, cal_fit_df, select_df, temporal_benchmark_df = _toy_splits()
    pipeline = fit_baseline_pipeline(train_df)
    y_cal_fit = cal_fit_df[TARGET_COL].to_numpy()
    y_select = select_df[TARGET_COL].to_numpy()
    y_temporal_benchmark = temporal_benchmark_df[TARGET_COL].to_numpy()
    raw_cal_fit = pipeline.predict_proba(cal_fit_df)[:, 1]
    raw_select = pipeline.predict_proba(select_df)[:, 1]
    raw_temporal_benchmark = pipeline.predict_proba(temporal_benchmark_df)[:, 1]

    result_a = calibration_comparison(
        y_cal_fit, raw_cal_fit, y_select, raw_select, y_temporal_benchmark, raw_temporal_benchmark
    )

    mutated_temporal_benchmark = np.full_like(raw_temporal_benchmark, 0.999)
    result_b = calibration_comparison(
        y_cal_fit, raw_cal_fit, y_select, raw_select, y_temporal_benchmark, mutated_temporal_benchmark
    )

    for method in result_a:
        a_sel = result_a[method]["selection"]
        b_sel = result_b[method]["selection"]
        assert a_sel["roc_auc"] == pytest.approx(b_sel["roc_auc"])
        assert a_sel["brier_score"] == pytest.approx(b_sel["brier_score"])


def test_select_best_calibration_method_only_reads_requested_split():
    fake_comparison = {
        "sigmoid": {"selection": {"brier_score": 0.10}, "temporal_benchmark": {"brier_score": 999.0}},
        "isotonic": {"selection": {"brier_score": 0.05}, "temporal_benchmark": {"brier_score": 0.0}},
    }
    assert select_best_calibration_method(fake_comparison, split="selection") == "isotonic"


def test_select_best_calibration_method_prefers_sigmoid_when_effectively_tied():
    """Objective 6: a bare argmin would flip to isotonic on a
    noise-level (1%) difference. The corrected selection rule must stick
    with the simpler sigmoid calibrator unless isotonic's improvement
    clears the minimum relative-improvement bar (default 3%)."""
    tied_comparison = {
        "sigmoid": {"selection": {"brier_score": 0.100}},
        "isotonic": {"selection": {"brier_score": 0.099}},  # 1% relative improvement
    }
    assert select_best_calibration_method(tied_comparison, split="selection") == "sigmoid"


def test_select_best_calibration_method_switches_to_isotonic_on_clear_improvement():
    clear_win_comparison = {
        "sigmoid": {"selection": {"brier_score": 0.100}},
        "isotonic": {"selection": {"brier_score": 0.090}},  # 10% relative improvement
    }
    assert select_best_calibration_method(clear_win_comparison, split="selection") == "isotonic"


def test_time_feature_ablation_returns_expected_structure():
    df = _toy_model_df()
    train_df, val_df, temporal_benchmark_df, _ = chronological_split(df, train_frac=0.6, val_frac=0.2)
    result = time_feature_ablation(train_df, val_df, temporal_benchmark_df)

    assert set(result) == {"with_time_drift_features", "without_time_drift_features"}
    for variant in result:
        for split in ("validation", "temporal_benchmark"):
            metrics = result[variant][split]
            assert 0.0 <= metrics["roc_auc"] <= 1.0
            assert 0.0 <= metrics["pr_auc"] <= 1.0
            assert 0.0 <= metrics["brier_score"] <= 1.0


def test_build_cost_variations_preserves_unvaried_fields():
    costs = load_cost_assumptions()
    variations = build_cost_variations(costs)
    labels = {v["label"] for v in variations}
    assert labels == {
        "fraud_loss_2x",
        "fraud_loss_half",
        "false_positive_cost_2x",
        "false_positive_cost_half",
        "review_cost_2x",
        "review_cost_half",
        "review_cost_exceeds_decline_cost",
        "fraud_loss_near_zero",
        "review_capacity_tightened",
        "review_catch_rate_low",
        "review_false_positive_rate_high",
    }
    fraud_2x = next(v for v in variations if v["label"] == "fraud_loss_2x")
    assert fraud_2x["cost_per_missed_fraud"] == pytest.approx(costs.cost_per_missed_fraud * 2)

    capacity_variation = next(v for v in variations if v["label"] == "review_capacity_tightened")
    assert capacity_variation["review_capacity_frac"] == pytest.approx(0.01)
    catch_rate_variation = next(v for v in variations if v["label"] == "review_catch_rate_low")
    assert catch_rate_variation["review_fraud_catch_rate"] == pytest.approx(0.4)
    fp_rate_variation = next(v for v in variations if v["label"] == "review_false_positive_rate_high")
    assert fp_rate_variation["review_false_positive_rate"] == pytest.approx(0.30)


@requires_baseline_features
def test_full_milestone4_pipeline_on_real_baseline_table_produces_sane_metrics():
    from src.models.train_baseline import load_baseline_table
    from src.models.cost_model import evaluate_policy_grid

    df = load_baseline_table()
    train_df, val_df, temporal_benchmark_df, _ = chronological_split(df)
    cal_fit_df, select_df, _ = chronological_split_two_way(val_df, first_frac=CALIBRATION_FIT_FRAC_OF_VALIDATION)

    pipeline = fit_baseline_pipeline(train_df)
    y_cal_fit = cal_fit_df[TARGET_COL].to_numpy()
    y_select = select_df[TARGET_COL].to_numpy()
    y_temporal_benchmark = temporal_benchmark_df[TARGET_COL].to_numpy()
    raw_cal_fit = pipeline.predict_proba(cal_fit_df)[:, 1]
    raw_select = pipeline.predict_proba(select_df)[:, 1]
    raw_temporal_benchmark = pipeline.predict_proba(temporal_benchmark_df)[:, 1]

    comparison = calibration_comparison(
        y_cal_fit, raw_cal_fit, y_select, raw_select, y_temporal_benchmark, raw_temporal_benchmark
    )

    # Ranking (ROC-AUC) should be ~unchanged by calibration: sigmoid is a
    # strictly monotonic transform of the raw score so its AUC matches
    # exactly; isotonic can introduce ties (flat regions) that nudge AUC by
    # a small amount, so it gets a looser tolerance, not an exact match.
    uncalibrated_auc = comparison["uncalibrated"]["selection"]["roc_auc"]
    assert comparison["sigmoid"]["selection"]["roc_auc"] == pytest.approx(uncalibrated_auc, abs=1e-9)
    assert comparison["isotonic"]["selection"]["roc_auc"] == pytest.approx(uncalibrated_auc, abs=0.01)

    for method in ("sigmoid", "isotonic"):
        assert comparison[method]["selection"]["brier_score"] < comparison["uncalibrated"]["selection"]["brier_score"]

    costs = load_cost_assumptions()
    grid = evaluate_policy_grid(y_select, raw_select, REVIEW_THRESHOLD_GRID, DECLINE_THRESHOLD_GRID, costs)
    assert len(grid) > 0
    assert (grid["estimated_illustrative_total_cost"].diff().dropna() >= 0).all()

    ablation = time_feature_ablation(train_df, val_df, temporal_benchmark_df)
    assert ablation["with_time_drift_features"]["temporal_benchmark"]["roc_auc"] > 0.5
    assert ablation["without_time_drift_features"]["temporal_benchmark"]["roc_auc"] > 0.5
