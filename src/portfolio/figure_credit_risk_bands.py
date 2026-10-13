"""Visual 5 -- credit risk bands (Model A, temporal benchmark).

Shows mean predicted probability vs. observed adverse-outcome rate by
risk band (low/medium/high), boundaries fit on train only. Model A
(application-time-safe features only) is used per the task's default,
since the Milestone 4 report documents no stronger reason to show Model B
here as well. The outcome is explicitly labeled as the eventual adverse
terminal outcome among originated loans -- NOT default_12m / a 12-month PD.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from src.portfolio.metrics_config import CREDIT_RISK_BANDS_BENCHMARK
from src.portfolio.style import LAVENDER, TEXT_SECONDARY, VIOLET, add_source_note, apply_style, save_figure


def build() -> None:
    apply_style()
    bands = CREDIT_RISK_BANDS_BENCHMARK.value
    order = ["low", "medium", "high"]
    bands_sorted = sorted(bands, key=lambda b: order.index(b["risk_band"]))

    labels = [b["risk_band"].capitalize() for b in bands_sorted]
    predicted = [b["mean_predicted_probability"] for b in bands_sorted]
    observed = [b["observed_adverse_outcome_rate"] for b in bands_sorted]
    ns = [b["n"] for b in bands_sorted]

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(9.5, 5.8))
    bars1 = ax.bar(x - width / 2, predicted, width, label="Mean predicted probability", color=LAVENDER)
    bars2 = ax.bar(x + width / 2, observed, width, label="Observed adverse-outcome rate", color=VIOLET)

    # Lavender progression: alpha increases low -> medium -> high so risk
    # severity is legible even without relying on color alone (labels too).
    risk_alphas = [0.55, 0.78, 1.0]
    for bars in (bars1, bars2):
        for bar, a in zip(bars, risk_alphas):
            bar.set_alpha(a)

    for bars in (bars1, bars2):
        for bar in bars:
            h = bar.get_height()
            ax.annotate(
                f"{h:.1%}",
                xy=(bar.get_x() + bar.get_width() / 2, h),
                xytext=(0, 5),
                textcoords="offset points",
                ha="center",
                fontsize=10.5,
                fontweight="bold",
            )

    for i, n in enumerate(ns):
        ax.annotate(f"n={n:,}", xy=(x[i], -0.028), ha="center", fontsize=9.5, color=TEXT_SECONDARY, annotation_clip=False)

    ax.set_xticks(x)
    ax.set_xticklabels([f"{lab} risk" for lab in labels])
    ax.set_ylabel("Rate")
    ax.set_ylim(0, max(max(predicted), max(observed)) * 1.35)
    ax.legend(loc="upper left")

    fig.suptitle("Credit Risk Bands — Model A, Temporal Benchmark", fontsize=15, fontweight="bold", y=1.04)
    fig.text(
        0.5,
        0.965,
        "Outcome = eventual adverse terminal outcome among originated loans (variable 36/60-month horizon) — "
        "not default_12m, not a 12-month PD",
        ha="center",
        fontsize=10,
        color=TEXT_SECONDARY,
    )

    add_source_note(
        fig,
        "Source: data/interim/credit_risk_milestone4_metrics.json -> "
        "models.model_a_application_time_safe.risk_band_summary_benchmark; "
        "docs/moduleC_milestone4_credit_decisioning.md",
    )

    fig.tight_layout(rect=(0, 0.04, 1, 0.9))
    out = save_figure(fig, "credit_risk_bands.png")
    print(f"Wrote {out}")


if __name__ == "__main__":
    build()
