"""Milestone 4: probability calibration, cost-based decisioning, and a
time-feature ablation, all on top of Milestone 3's chronological split and
class-weighted logistic-regression baseline.

Reuses `src.models.split.chronological_split` with its Milestone 3 default
fractions — the train/validation/temporal-benchmark boundaries are
byte-for-byte the same split Milestone 3 used, not a new split.

**Terminology correction**: Milestone 3 called the final 20% segment
"holdout" and presented it as a single, final, untouched check. By the end
of Milestone 4 it has been inspected twice during iterative model
development — once for Milestone 3's baseline metrics, once for
Milestone 4's calibration/decisioning evaluation. It is therefore no
longer a pristine, never-touched final test set, and this module calls it
a **"temporal benchmark"** instead of "holdout" throughout: a fixed,
later-in-time slice still useful for checking that earlier choices
(calibration method, decision thresholds) generalize forward in time, but
NOT a number that should be quoted as an unbiased estimate of future
production performance. See docs/milestone4_calibration_decisioning.md.

**Methodology correction (calibration/selection split)**: the first
version of this module fit each calibrator on the validation period and
then reported that calibrator's Brier score / reliability table on THE
SAME validation period — an in-sample, optimistically-biased evaluation
(especially severe for isotonic regression, which can nearly perfectly
fit the data it was trained on). This version instead splits Milestone
3's validation period itself, chronologically, into two disjoint pieces
via `src.models.split.chronological_split_two_way`:
  - `cal_fit`: the earlier half — calibrators are `.fit()` here, and only
    here.
  - `selection`: the later half — calibration quality (Brier/reliability)
    is evaluated here, and the cost-based threshold/cost grid + sensitivity
    analysis are also selected here. Both uses are legitimate because
    `selection` was never passed to any `.fit()` call.
No calibrator or decision threshold in this module is ever fit, chosen,
or evaluated using data it (or anything derived from it) was fit on. The
temporal benchmark is touched exactly twice: once to EVALUATE calibration
quality out-of-sample at a later point in time, and once to EVALUATE (not
re-select) the selection-chosen lowest-cost decision policy. See
tests/models/test_train_calibrated.py.

Run: .venv/bin/python -m src.models.train_calibrated
Requires: data/processed/baseline_features_v1.parquet (Milestone 2).
Writes: data/interim/milestone4_metrics.json.
"""
from __future__ import annotations

import json
from dataclasses import asdict

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from src.data.config import load_paths
from src.models.calibration import CALIBRATION_METHODS, apply_calibrator, fit_calibrator, reliability_table
from src.models.cost_model import (
    COST_ASSUMPTIONS_DISCLAIMER,
    CostAssumptions,
    evaluate_decision_policy,
    evaluate_policy_grid,
    load_cost_assumptions,
    sensitivity_analysis,
)
from src.models.schema import NUMERIC_FEATURES, NUMERIC_FEATURES_NO_TIME_DRIFT, TARGET_COL
from src.models.split import chronological_split, chronological_split_two_way
from src.models.train_baseline import build_model_pipeline, evaluate_probs, load_baseline_table

# Exploration grids for the three-action policy — deliberately not a single
# "final" threshold pair. See src/models/cost_model.py::evaluate_policy_grid.
REVIEW_THRESHOLD_GRID = [0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50]
DECLINE_THRESHOLD_GRID = [0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]

# How the validation period is subdivided between calibration-fit and
# selection (calibration-evaluation + threshold/cost selection). 0.5 keeps
# both halves roughly comparable in size; see module docstring.
CALIBRATION_FIT_FRAC_OF_VALIDATION = 0.5

_METRIC_KEYS = ("roc_auc", "pr_auc", "brier_score")


def fit_baseline_pipeline(train_df: pd.DataFrame, numeric_features: list[str] | None = None) -> Pipeline:
    """Fits the Milestone 3 pipeline architecture on TRAIN ONLY. Passing
    `numeric_features=schema.NUMERIC_FEATURES_NO_TIME_DRIFT` builds the
    Milestone 4 ablation variant instead of the full feature set.
    """
    pipeline = build_model_pipeline(numeric_features=numeric_features)
    pipeline.fit(train_df, train_df[TARGET_COL])
    return pipeline


def calibration_comparison(
    y_cal_fit: np.ndarray,
    raw_probs_cal_fit: np.ndarray,
    y_select: np.ndarray,
    raw_probs_select: np.ndarray,
    y_temporal_benchmark: np.ndarray,
    raw_probs_temporal_benchmark: np.ndarray,
) -> dict:
    """Fits each calibration method in `CALIBRATION_METHODS` on the
    CALIBRATION-FIT slice only, then evaluates the resulting probabilities
    on two splits that calibration fitting never saw: `selection` (used
    for both this comparison and threshold/cost selection — see module
    docstring) and `temporal_benchmark` (Milestone 3's renamed holdout).
    Objectives 1-4 of Milestone 4 live here: separating ranking (ROC-AUC/
    PR-AUC, which calibration cannot change — see module docstring of
    src/models/calibration.py) from probability quality (Brier score,
    reliability table) — now computed strictly out-of-sample from the
    calibrator's own fit data (methodology correction; see module
    docstring above).
    """
    results = {}
    for method in CALIBRATION_METHODS:
        calibrator = fit_calibrator(method, raw_probs_cal_fit, y_cal_fit)
        select_probs = apply_calibrator(calibrator, raw_probs_select)
        tb_probs = apply_calibrator(calibrator, raw_probs_temporal_benchmark)

        results[method] = {
            "selection": {
                **evaluate_probs(y_select, select_probs),
                "reliability_table": reliability_table(y_select, select_probs),
            },
            "temporal_benchmark": {
                **evaluate_probs(y_temporal_benchmark, tb_probs),
                "reliability_table": reliability_table(y_temporal_benchmark, tb_probs),
            },
        }
    return results


def select_best_calibration_method(
    comparison: dict,
    split: str = "selection",
    metric: str = "brier_score",
    min_relative_improvement_for_isotonic: float = 0.03,
) -> str:
    """Chooses between "sigmoid" and "isotonic" using the OUT-OF-SAMPLE
    `selection` split by default — never the data either calibrator was
    fit on, and never `temporal_benchmark`, so this selection cannot leak
    temporal-benchmark information into the chosen method.

    Methodology correction: rather than a bare argmin over all calibrated
    methods (which would flip to isotonic on a noise-level difference),
    this prefers the simpler sigmoid calibrator UNLESS isotonic shows a
    CLEAR improvement — defined as at least
    `min_relative_improvement_for_isotonic` (default 3%) relative
    reduction in the chosen metric (lower is better, e.g. Brier score).
    Sigmoid is strictly simpler (two parameters, no flat regions, cannot
    introduce ties that nudge ROC-AUC — see src/models/calibration.py), so
    it is the right default when the two methods are effectively tied.
    """
    sigmoid_score = comparison["sigmoid"][split][metric]
    isotonic_score = comparison["isotonic"][split][metric]
    if sigmoid_score <= 0:
        return "sigmoid"
    relative_improvement = (sigmoid_score - isotonic_score) / sigmoid_score
    return "isotonic" if relative_improvement > min_relative_improvement_for_isotonic else "sigmoid"


def time_feature_ablation(train_df: pd.DataFrame, val_df: pd.DataFrame, temporal_benchmark_df: pd.DataFrame) -> dict:
    """Compares the full feature set against `NUMERIC_FEATURES_NO_TIME_DRIFT`
    (day_index/week_index removed) on identical train/val/temporal-benchmark
    data, isolating how much ranking performance comes from the two
    temporal drift proxies schema.py flags as carrying extrapolation risk.
    No calibrator is fit anywhere in this function, so the
    fit-vs-evaluate-on-same-data concern that motivated the calibration
    methodology correction (see module docstring) does not apply here:
    both pipelines are fit on TRAIN only and only ever evaluated on
    val/temporal_benchmark.
    """
    full_pipeline = fit_baseline_pipeline(train_df, numeric_features=NUMERIC_FEATURES)
    no_time_pipeline = fit_baseline_pipeline(train_df, numeric_features=NUMERIC_FEATURES_NO_TIME_DRIFT)

    def _summary(pipeline: Pipeline, df: pd.DataFrame) -> dict:
        y_true = df[TARGET_COL].to_numpy()
        y_prob = pipeline.predict_proba(df)[:, 1]
        metrics = evaluate_probs(y_true, y_prob)
        return {k: metrics[k] for k in _METRIC_KEYS}

    return {
        "with_time_drift_features": {
            "validation": _summary(full_pipeline, val_df),
            "temporal_benchmark": _summary(full_pipeline, temporal_benchmark_df),
        },
        "without_time_drift_features": {
            "validation": _summary(no_time_pipeline, val_df),
            "temporal_benchmark": _summary(no_time_pipeline, temporal_benchmark_df),
        },
    }


def build_cost_variations(base_costs: CostAssumptions) -> list[dict]:
    """One isolated variation per cost input, so the sensitivity table
    (Milestone 4 requirement E) can be read one row at a time: "if X
    assumption moves, does the lowest-cost policy move".

    The +/-2x variations alone turned out to be too mild on this dataset:
    manual review being much cheaper than a declined legitimate
    transaction means simply doubling or halving any one input never
    changes which action is cheapest at the margin — all six produce the
    identical argmin-cost policy and only the reported dollar total moves.
    That is itself a genuine, reportable finding about this cost
    configuration, but two more extreme, targeted variations are added to
    actually cross the relevant break-even points:
      - `review_cost_exceeds_decline_cost`: raises manual review above the
        decline cost, so review is no longer unconditionally the cheaper
        way to act on a suspicious transaction.
      - `fraud_loss_near_zero`: drops missed-fraud cost low enough that
        acting on anything stops being worth it at all.

    Two further variations exercise the Milestone 4 methodology
    correction's realistic-review-model parameters specifically:
      - `review_capacity_tightened`: a much smaller review capacity,
        intended to actually trigger the prioritize-highest-risk /
        overflow-to-approve behavior described in
        src/models/cost_model.py, not just assume it theoretically exists.
      - `review_catch_rate_low`: a materially worse review catch rate, to
        show recall degrading once review is modeled as an unreliable
        (not perfect) backstop, independent of its capacity limit.
    """
    return [
        {"label": "fraud_loss_2x", "cost_per_missed_fraud": base_costs.cost_per_missed_fraud * 2},
        {"label": "fraud_loss_half", "cost_per_missed_fraud": base_costs.cost_per_missed_fraud * 0.5},
        {
            "label": "false_positive_cost_2x",
            "cost_per_false_positive_decline": base_costs.cost_per_false_positive_decline * 2,
        },
        {
            "label": "false_positive_cost_half",
            "cost_per_false_positive_decline": base_costs.cost_per_false_positive_decline * 0.5,
        },
        {"label": "review_cost_2x", "cost_per_manual_review": base_costs.cost_per_manual_review * 2},
        {"label": "review_cost_half", "cost_per_manual_review": base_costs.cost_per_manual_review * 0.5},
        {
            "label": "review_cost_exceeds_decline_cost",
            "cost_per_manual_review": base_costs.cost_per_false_positive_decline * 2.4,
        },
        {"label": "fraud_loss_near_zero", "cost_per_missed_fraud": 5.0},
        {"label": "review_capacity_tightened", "review_capacity_frac": 0.01},
        {"label": "review_catch_rate_low", "review_fraud_catch_rate": 0.4},
    ]


def main() -> int:
    paths = load_paths()
    df = load_baseline_table(paths)

    train_df, val_df, temporal_benchmark_df, split_report = chronological_split(df)
    print("=== Split report (train/validation boundaries identical to Milestone 3; final segment renamed temporal_benchmark) ===")
    print(json.dumps(split_report, indent=2))

    cal_fit_df, select_df, val_subsplit_report = chronological_split_two_way(
        val_df, first_frac=CALIBRATION_FIT_FRAC_OF_VALIDATION
    )
    print(
        "\n=== Validation-period sub-split (methodology correction): "
        "calibration-fit (earlier half) vs. selection (later half, out-of-sample) ==="
    )
    print(json.dumps(val_subsplit_report, indent=2))

    pipeline = fit_baseline_pipeline(train_df)
    y_cal_fit = cal_fit_df[TARGET_COL].to_numpy()
    y_select = select_df[TARGET_COL].to_numpy()
    y_temporal_benchmark = temporal_benchmark_df[TARGET_COL].to_numpy()
    raw_probs_cal_fit = pipeline.predict_proba(cal_fit_df)[:, 1]
    raw_probs_select = pipeline.predict_proba(select_df)[:, 1]
    raw_probs_temporal_benchmark = pipeline.predict_proba(temporal_benchmark_df)[:, 1]

    comparison = calibration_comparison(
        y_cal_fit, raw_probs_cal_fit, y_select, raw_probs_select, y_temporal_benchmark, raw_probs_temporal_benchmark
    )
    print("\n=== Calibration comparison: ROC-AUC / PR-AUC / Brier (selection & temporal_benchmark, both out-of-sample from calibration fit) ===")
    comparison_summary = {
        method: {
            split_name: {k: comparison[method][split_name][k] for k in _METRIC_KEYS}
            for split_name in ("selection", "temporal_benchmark")
        }
        for method in comparison
    }
    print(json.dumps(comparison_summary, indent=2))

    best_method = select_best_calibration_method(comparison)
    print(f"\nBest calibration method (selection-split Brier, prefers sigmoid unless isotonic clearly better): {best_method}")

    best_calibrator = fit_calibrator(best_method, raw_probs_cal_fit, y_cal_fit)
    calibrated_probs_select = apply_calibrator(best_calibrator, raw_probs_select)
    calibrated_probs_temporal_benchmark = apply_calibrator(best_calibrator, raw_probs_temporal_benchmark)

    ablation = time_feature_ablation(train_df, val_df, temporal_benchmark_df)
    print("\n=== Time-feature ablation (day_index/week_index) ===")
    print(json.dumps(ablation, indent=2))

    costs = load_cost_assumptions()
    print(f"\n=== Cost assumptions ===\n{COST_ASSUMPTIONS_DISCLAIMER}")
    print(json.dumps(asdict(costs), indent=2))

    threshold_grid = evaluate_policy_grid(
        y_select, calibrated_probs_select, REVIEW_THRESHOLD_GRID, DECLINE_THRESHOLD_GRID, costs
    )
    print(f"\n=== Threshold/cost tradeoff grid (SELECTION SPLIT ONLY, {best_method}-calibrated probabilities) ===")
    print(threshold_grid.to_string(index=False))

    cost_variations = build_cost_variations(costs)
    sensitivity = sensitivity_analysis(
        y_select, calibrated_probs_select, REVIEW_THRESHOLD_GRID, DECLINE_THRESHOLD_GRID, costs, cost_variations
    )
    print("\n=== Sensitivity analysis: lowest-cost policy per cost variation (SELECTION SPLIT ONLY) ===")
    print(sensitivity.to_string(index=False))

    best_policy = threshold_grid.iloc[0]
    temporal_benchmark_policy_eval = evaluate_decision_policy(
        y_temporal_benchmark,
        calibrated_probs_temporal_benchmark,
        float(best_policy["review_threshold"]),
        float(best_policy["decline_threshold"]),
        costs,
    )
    print(
        "\n=== Temporal benchmark evaluation (touched once for this policy; this slice was ALSO "
        "touched once already for Milestone 3's baseline metrics and once already for this "
        "milestone's calibration comparison above — it is NOT a pristine, never-touched final "
        "test set, and is not re-optimized on here) ==="
    )
    print(json.dumps(temporal_benchmark_policy_eval, indent=2))

    out = {
        "split_report": split_report,
        "validation_period_subsplit_report": val_subsplit_report,
        "calibration_comparison": comparison,
        "best_calibration_method": best_method,
        "time_feature_ablation": ablation,
        "cost_assumptions": {"disclaimer": COST_ASSUMPTIONS_DISCLAIMER, **asdict(costs)},
        "threshold_cost_grid_selection": threshold_grid.to_dict(orient="records"),
        "sensitivity_analysis_selection": sensitivity.to_dict(orient="records"),
        "temporal_benchmark_policy_evaluation": temporal_benchmark_policy_eval,
        "methodology_note": (
            "The final chronological segment (called 'holdout' in Milestone 3) is called "
            "'temporal_benchmark' from Milestone 4 onward: it has now been inspected during "
            "iterative model development (Milestone 3 baseline metrics, Milestone 4 calibration "
            "comparison and policy evaluation) and must not be presented as a pristine, "
            "never-touched final unbiased test set. Calibration methods are fit only on the "
            "earlier half of the validation period ('cal_fit') and evaluated/selected only on "
            "the later half ('selection') or the temporal benchmark — never on cal_fit itself."
        ),
    }
    out_path = paths.interim_dir / "milestone4_metrics.json"
    paths.interim_dir.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
