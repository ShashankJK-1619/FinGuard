"""Visual 4 -- fraud decisioning tradeoff: recall vs. illustrative cost,
logistic baseline vs. preferred nonlinear model, under the project's
existing illustrative cost-based policy. Shown on the temporal benchmark
(the chronologically held-out, forward-looking split); the validation-split
figures are cited in the caption for the full 14.5-17.3% range.
"""
from __future__ import annotations

import matplotlib.pyplot as plt

from src.portfolio.metrics_config import (
    FRAUD_LOGISTIC_POLICY_TB,
    FRAUD_LOGISTIC_POLICY_VAL,
    FRAUD_NONLINEAR_POLICY_TB,
    FRAUD_NONLINEAR_POLICY_VAL,
)
from src.portfolio.style import GRAY, NAVY, TEAL, add_source_note, annotate_bars, apply_style, save_figure


def build() -> None:
    apply_style()

    log_tb = FRAUD_LOGISTIC_POLICY_TB.value
    nl_tb = FRAUD_NONLINEAR_POLICY_TB.value
    log_val = FRAUD_LOGISTIC_POLICY_VAL.value
    nl_val = FRAUD_NONLINEAR_POLICY_VAL.value

    recall_log = log_tb["fraud_recall"]
    recall_nl = nl_tb["fraud_recall"]
    cost_log = log_tb["estimated_illustrative_total_cost"]
    cost_nl = nl_tb["estimated_illustrative_total_cost"]
    cost_reduction_tb = (cost_log - cost_nl) / cost_log * 100
    cost_reduction_val = (
        (log_val["estimated_illustrative_total_cost"] - nl_val["estimated_illustrative_total_cost"])
        / log_val["estimated_illustrative_total_cost"]
        * 100
    )
    recall_ratio_tb = recall_nl / recall_log

    fig, (ax_recall, ax_cost) = plt.subplots(1, 2, figsize=(11, 5.3))

    labels = ["Logistic\nbaseline", "Preferred\nnonlinear"]

    bars_recall = ax_recall.bar(labels, [recall_log, recall_nl], color=[NAVY, TEAL], width=0.5)
    annotate_bars(ax_recall, bars_recall, fmt="{:.1%}", fontsize=12)
    ax_recall.set_ylim(0, max(recall_log, recall_nl) * 1.5)
    ax_recall.set_ylabel("Fraud recall (under illustrative policy)")
    ax_recall.set_title(f"Fraud recall  —  ~{recall_ratio_tb:.2f}x", fontsize=12.5)

    bars_cost = ax_cost.bar(labels, [cost_log / 1000, cost_nl / 1000], color=[NAVY, TEAL], width=0.5)
    annotate_bars(ax_cost, bars_cost, fmt="${:,.0f}k", fontsize=12)
    ax_cost.set_ylim(0, cost_log / 1000 * 1.3)
    ax_cost.set_ylabel("Illustrative total cost ($k)")
    ax_cost.set_title(f"Illustrative cost  —  -{cost_reduction_tb:.1f}%", fontsize=12.5)

    fig.suptitle(
        "Fraud Decisioning Tradeoff — Illustrative Cost-Based Policy (Temporal Benchmark)",
        fontsize=14.5,
        fontweight="bold",
        y=1.05,
    )
    fig.text(
        0.5,
        0.975,
        f"Across both evaluation splits the preferred nonlinear model cuts illustrative cost by "
        f"{min(cost_reduction_tb, cost_reduction_val):.1f}-{max(cost_reduction_tb, cost_reduction_val):.1f}% "
        "while roughly doubling fraud recall",
        ha="center",
        fontsize=10.2,
        color=GRAY,
    )
    fig.text(
        0.5,
        0.065,
        "Illustrative cost assumptions; not real company economics.",
        ha="center",
        fontsize=9.5,
        color=GRAY,
        style="italic",
        fontweight="bold",
    )

    add_source_note(
        fig,
        "Source: data/interim/milestone5_metrics.json -> {logistic,nonlinear_preferred_no_time_index}_decision_policy; "
        "docs/milestone5_nonlinear_comparison.md Section 6",
    )

    fig.tight_layout(rect=(0, 0.1, 1, 0.9))
    out = save_figure(fig, "fraud_decisioning_tradeoff.png")
    print(f"Wrote {out}")


if __name__ == "__main__":
    build()
