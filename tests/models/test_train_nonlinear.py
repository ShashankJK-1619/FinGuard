from __future__ import annotations

import numpy as np
import pytest

from src.models.schema import BINARY_FEATURES, CATEGORICAL_FEATURES, NUMERIC_FEATURES, TARGET_COL
from src.models.split import chronological_split, chronological_split_two_way
from src.models.train_calibrated import CALIBRATION_FIT_FRAC_OF_VALIDATION
from src.models.train_nonlinear import (
    HGB_PARAM_CANDIDATES,
    build_nonlinear_pipeline,
    build_tree_preprocessor,
    categorical_feature_mask,
    compare_feature_importance,
    decision_policy_comparison,
    fit_nonlinear_pipeline,
    permutation_importance_table,
    time_feature_ablation_nonlinear,
    tune_hyperparameters,
)
from src.models.train_baseline import extract_coefficients
from src.models.train_calibrated import fit_baseline_pipeline
from src.models.cost_model import load_cost_assumptions
from tests.conftest import requires_baseline_features
from tests.models.test_train_baseline import _toy_model_df

# Cheap, deterministic candidates for tests that exercise the tuning loop —
# HGB_PARAM_CANDIDATES itself is already modest (4 candidates), but tests
# shrink max_iter further so the suite stays fast.
_FAST_PARAM_CANDIDATES = [
    {"max_iter": 20, "max_leaf_nodes": 15, "learning_rate": 0.2, "l2_regularization": 0.0},
    {"max_iter": 30, "max_leaf_nodes": 31, "learning_rate": 0.1, "l2_regularization": 0.0},
]


def _toy_splits():
    """Same construction as tests/models/test_train_calibrated.py::_toy_splits
    — n=3000 so every quarter-ish split retains both classes."""
    df = _toy_model_df(n=3000)
    train_df, val_df, temporal_benchmark_df, _ = chronological_split(df, train_frac=0.6, val_frac=0.2)
    cal_fit_df, select_df, _ = chronological_split_two_way(val_df, first_frac=CALIBRATION_FIT_FRAC_OF_VALIDATION)
    return train_df, cal_fit_df, select_df, temporal_benchmark_df


def test_categorical_feature_mask_matches_transformer_column_order():
    mask = categorical_feature_mask()
    assert len(mask) == len(NUMERIC_FEATURES) + len(CATEGORICAL_FEATURES) + len(BINARY_FEATURES)
    assert mask[: len(NUMERIC_FEATURES)] == [False] * len(NUMERIC_FEATURES)
    assert mask[len(NUMERIC_FEATURES) : len(NUMERIC_FEATURES) + len(CATEGORICAL_FEATURES)] == [True] * len(
        CATEGORICAL_FEATURES
    )
    assert mask[len(NUMERIC_FEATURES) + len(CATEGORICAL_FEATURES) :] == [False] * len(BINARY_FEATURES)


def test_tree_preprocessor_output_width_matches_mask_length():
    df = _toy_model_df(n=200)
    preprocessor = build_tree_preprocessor()
    transformed = preprocessor.fit_transform(df)
    assert transformed.shape[1] == len(categorical_feature_mask())


def test_build_nonlinear_pipeline_fit_and_predict_proba_are_well_formed():
    train_df, _, select_df, _ = _toy_splits()
    pipeline = build_nonlinear_pipeline(hgb_params=_FAST_PARAM_CANDIDATES[0])
    pipeline.fit(train_df, train_df[TARGET_COL])
    probs = pipeline.predict_proba(select_df)[:, 1]

    assert probs.shape[0] == len(select_df)
    assert (probs >= 0).all() and (probs <= 1).all()


def test_nonlinear_pipeline_handles_missing_numeric_and_categorical_values_natively():
    """HistGradientBoostingClassifier should fit and predict even when
    numeric/categorical columns contain NaN — no imputation happens in
    build_tree_preprocessor for numeric columns by design (see module
    docstring), so this is a direct check that missing values don't crash
    the pipeline the way they would a bare sklearn estimator."""
    df = _toy_model_df(n=300)
    df.loc[df.index[:20], NUMERIC_FEATURES[0]] = np.nan
    df.loc[df.index[:20], CATEGORICAL_FEATURES[0]] = np.nan
    train_df, val_df, _, _ = chronological_split(df, train_frac=0.6, val_frac=0.2)

    pipeline = build_nonlinear_pipeline(hgb_params=_FAST_PARAM_CANDIDATES[0])
    pipeline.fit(train_df, train_df[TARGET_COL])
    probs = pipeline.predict_proba(val_df)[:, 1]
    assert np.isfinite(probs).all()


def test_tune_hyperparameters_only_uses_passed_dataframe():
    """Leakage-safety smoke test: tune_hyperparameters must not require (or
    accept) anything beyond train_df — selection/temporal_benchmark are
    never passed in, mirroring Requirement B."""
    train_df, _, _, _ = _toy_splits()
    result = tune_hyperparameters(train_df, param_candidates=_FAST_PARAM_CANDIDATES, n_folds=2)

    assert set(result) == {
        "candidates",
        "n_folds",
        "selection_metric",
        "best_params",
        "best_mean_pr_auc",
        "tuning_runtime_seconds",
    }
    assert len(result["candidates"]) == len(_FAST_PARAM_CANDIDATES)
    assert result["best_params"] in _FAST_PARAM_CANDIDATES
    assert result["tuning_runtime_seconds"] >= 0
    for candidate in result["candidates"]:
        assert len(candidate["fold_pr_auc"]) == 2


def test_tune_hyperparameters_picks_the_candidate_with_highest_mean_pr_auc():
    train_df, _, _, _ = _toy_splits()
    result = tune_hyperparameters(train_df, param_candidates=_FAST_PARAM_CANDIDATES, n_folds=2)
    best_from_candidates = max(result["candidates"], key=lambda r: r["mean_pr_auc"])
    assert result["best_params"] == best_from_candidates["params"]
    assert result["best_mean_pr_auc"] == pytest.approx(best_from_candidates["mean_pr_auc"])


def test_time_feature_ablation_nonlinear_returns_expected_structure():
    train_df, _, select_df, temporal_benchmark_df = _toy_splits()
    # select_df stands in for a validation split here — this test only checks
    # structural shape, not real validation semantics.
    result = time_feature_ablation_nonlinear(train_df, select_df, temporal_benchmark_df, _FAST_PARAM_CANDIDATES[0])

    assert set(result) == {"with_time_drift_features", "without_time_drift_features"}
    for variant in result:
        for split in ("validation", "temporal_benchmark"):
            metrics = result[variant][split]
            assert 0.0 <= metrics["roc_auc"] <= 1.0
            assert 0.0 <= metrics["pr_auc"] <= 1.0
            assert 0.0 <= metrics["brier_score"] <= 1.0


def test_permutation_importance_table_covers_every_raw_feature():
    train_df, _, select_df, _ = _toy_splits()
    pipeline = fit_nonlinear_pipeline(train_df, _FAST_PARAM_CANDIDATES[0])
    table = permutation_importance_table(pipeline, select_df, n_repeats=2)

    expected_features = set(NUMERIC_FEATURES) | set(CATEGORICAL_FEATURES) | set(BINARY_FEATURES)
    assert set(table["feature"]) == expected_features
    assert {"feature", "importance_mean", "importance_std", "is_masked_code"} <= set(table.columns)
    # sorted descending by importance_mean
    assert (table["importance_mean"].diff().dropna() <= 1e-12).all()


def test_compare_feature_importance_reports_overlap_and_disjoint_sets():
    train_df, _, select_df, _ = _toy_splits()
    logistic_pipeline = fit_baseline_pipeline(train_df)
    coef_table = extract_coefficients(logistic_pipeline)
    nonlinear_pipeline = fit_nonlinear_pipeline(train_df, _FAST_PARAM_CANDIDATES[0])
    importance_table = permutation_importance_table(nonlinear_pipeline, select_df, n_repeats=2)

    result = compare_feature_importance(coef_table, importance_table, top_n=5)

    assert set(result) == {
        "top_n",
        "logistic_top_features",
        "nonlinear_top_features",
        "agree_in_both_top_n",
        "only_in_logistic_top_n",
        "only_in_nonlinear_top_n",
    }
    assert len(result["nonlinear_top_features"]) == 5
    assert set(result["agree_in_both_top_n"]) <= set(result["logistic_top_features"])
    assert set(result["agree_in_both_top_n"]) <= set(result["nonlinear_top_features"])


def test_decision_policy_comparison_returns_expected_keys_and_touches_temporal_benchmark_once():
    train_df, _, select_df, temporal_benchmark_df = _toy_splits()
    pipeline = fit_nonlinear_pipeline(train_df, _FAST_PARAM_CANDIDATES[0])
    y_select = select_df[TARGET_COL].to_numpy()
    y_tb = temporal_benchmark_df[TARGET_COL].to_numpy()
    probs_select = pipeline.predict_proba(select_df)[:, 1]
    probs_tb = pipeline.predict_proba(temporal_benchmark_df)[:, 1]
    costs = load_cost_assumptions()

    result = decision_policy_comparison(y_select, probs_select, y_tb, probs_tb, costs)

    assert set(result) == {
        "threshold_cost_grid_selection",
        "best_policy_selection",
        "temporal_benchmark_policy_evaluation",
    }
    assert len(result["threshold_cost_grid_selection"]) > 0
    best = result["best_policy_selection"]
    tb_eval = result["temporal_benchmark_policy_evaluation"]
    assert tb_eval["review_threshold"] == pytest.approx(best["review_threshold"])
    assert tb_eval["decline_threshold"] == pytest.approx(best["decline_threshold"])


@requires_baseline_features
def test_full_milestone5_pipeline_on_real_baseline_table_produces_sane_metrics():
    from src.models.train_baseline import load_baseline_table

    df = load_baseline_table()
    train_df, val_df, temporal_benchmark_df, _ = chronological_split(df)
    cal_fit_df, select_df, _ = chronological_split_two_way(val_df, first_frac=CALIBRATION_FIT_FRAC_OF_VALIDATION)

    # Keep this real-data test fast: use the cheapest HGB_PARAM_CANDIDATES
    # entry directly rather than running the full tune_hyperparameters
    # search (that full search is exercised in main() / the toy-data tests
    # above, not here — this test is a sanity check on real data shape and
    # scale, not a re-run of the tuning procedure).
    pipeline = fit_nonlinear_pipeline(train_df, HGB_PARAM_CANDIDATES[0])
    y_select = select_df[TARGET_COL].to_numpy()
    y_temporal_benchmark = temporal_benchmark_df[TARGET_COL].to_numpy()
    probs_select = pipeline.predict_proba(select_df)[:, 1]
    probs_temporal_benchmark = pipeline.predict_proba(temporal_benchmark_df)[:, 1]

    from sklearn.metrics import roc_auc_score

    assert roc_auc_score(y_select, probs_select) > 0.6
    assert roc_auc_score(y_temporal_benchmark, probs_temporal_benchmark) > 0.6

    costs = load_cost_assumptions()
    result = decision_policy_comparison(
        y_select, probs_select, y_temporal_benchmark, probs_temporal_benchmark, costs
    )
    assert len(result["threshold_cost_grid_selection"]) > 0
