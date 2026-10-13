"""Visual 3 -- fraud model comparison, logistic baseline vs. preferred
nonlinear model, on the chronologically held-out temporal benchmark.

PR-AUC is emphasized (larger panel, first) because fraud prevalence in
this slice is ~3.4% -- ROC-AUC alone can look flattering under heavy
class imbalance, so PR-AUC is the more decision-relevant metric here.
"""
from __future__ import annotations

import matplotlib.pyplot as plt

from src.portfolio.metrics_config import (
    FRAUD_LOGISTIC_PR_AUC,
    FRAUD_LOGISTIC_ROC_AUC,
    FRAUD_NONLINEAR_PR_AUC,
    FRAUD_NONLINEAR_ROC_AUC,
)
from src.portfolio.style import GRAY, NAVY, TEAL, add_source_note, annotate_bars, apply_style, save_figure


def build() -> None:
    apply_style()

    logistic_roc = FRAUD_LOGISTIC_ROC_AUC.value
    logistic_pr = FRAUD_LOGISTIC_PR_AUC.value
    nonlinear_roc = FRAUD_NONLINEAR_ROC_AUC.value
    nonlinear_pr = FRAUD_NONLINEAR_PR_AUC.value

    fig, (ax_pr, ax_roc) = plt.subplots(1, 2, figsize=(11, 5.3), gridspec_kw={"width_ratios": [1.15, 1]})

    labels = ["Logistic\nbaseline", "Preferred\nnonlinear"]

    # PR-AUC panel (primary / emphasized).
    bars_pr = ax_pr.bar(labels, [logistic_pr, nonlinear_pr], color=[NAVY, TEAL], width=0.5)
    annotate_bars(ax_pr, bars_pr, fmt="{:.3f}", fontsize=12)
    relative_lift_pct = (nonlinear_pr / logistic_pr - 1) * 100
    ax_pr.set_ylim(0, max(nonlinear_pr, logistic_pr) * 1.35)
    ax_pr.set_ylabel("PR-AUC")
    ax_pr.set_title(f"PR-AUC (primary metric)  —  +{relative_lift_pct:.0f}% relative", fontsize=12.5)

    # ROC-AUC panel (secondary).
    bars_roc = ax_roc.bar(labels, [logistic_roc, nonlinear_roc], color=[NAVY, TEAL], width=0.5)
    annotate_bars(ax_roc, bars_roc, fmt="{:.3f}", fontsize=12)
    ax_roc.set_ylim(0, 1.0)
    ax_roc.set_ylabel("ROC-AUC")
    ax_roc.set_title("ROC-AUC (secondary)", fontsize=12.5)

    fig.suptitle(
        "Fraud Model Comparison — Chronological Temporal Benchmark",
        fontsize=15,
        fontweight="bold",
        y=1.04,
    )
    fig.text(
        0.5,
        0.965,
        "PR-AUC is emphasized because fraud prevalence is only ~3.4% in this held-out slice — "
        "ROC-AUC alone can overstate performance under class imbalance",
        ha="center",
        fontsize=10.2,
        color=GRAY,
    )

    add_source_note(
        fig,
        "Source: data/interim/milestone5_metrics.json -> head_to_head_calibrated_metrics.temporal_benchmark; "
        "docs/milestone5_nonlinear_comparison.md",
    )

    fig.tight_layout(rect=(0, 0.03, 1, 0.92))
    out = save_figure(fig, "fraud_model_comparison.png")
    print(f"Wrote {out}")


if __name__ == "__main__":
    build()
