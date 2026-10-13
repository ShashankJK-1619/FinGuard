"""Visual 2 -- payments A/B experiment result.

Shows the synthetic experiment's completion-rate uplift against its
predeclared practical-significance threshold, and the resulting NO-SHIP
call. All treatment behavior is synthetic/simulated, layered on real
IEEE-CIS transaction rows -- this was never a live production experiment.
"""
from __future__ import annotations

import matplotlib.pyplot as plt

from src.portfolio.metrics_config import (
    PAYMENTS_MIN_PRACTICAL_UPLIFT_ABS,
    PAYMENTS_PRIMARY_METRIC_TEST,
)
from src.portfolio.style import (
    AMBER_MUTED,
    LAVENDER,
    LAVENDER_LIGHT,
    PERIWINKLE,
    TEXT_SECONDARY,
    VIOLET,
    add_source_note,
    apply_style,
    save_figure,
)


def build() -> None:
    apply_style()
    test = PAYMENTS_PRIMARY_METRIC_TEST.value
    threshold = PAYMENTS_MIN_PRACTICAL_UPLIFT_ABS.value

    control_rate = test["control_rate"]
    treatment_rate = test["treatment_rate"]
    uplift = test["absolute_uplift"]
    ci_low = test["ci_low"]
    ci_high = test["ci_high"]
    p_value = test["p_value"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5.2), gridspec_kw={"width_ratios": [1, 1.1]})

    # Panel 1: control vs treatment completion rate.
    bars = ax1.bar(
        ["Control", "Treatment\n(synthetic)"],
        [control_rate, treatment_rate],
        color=[LAVENDER, PERIWINKLE],
        width=0.55,
    )
    for bar, val in zip(bars, [control_rate, treatment_rate]):
        ax1.annotate(
            f"{val:.4f}",
            xy=(bar.get_x() + bar.get_width() / 2, val),
            xytext=(0, 6),
            textcoords="offset points",
            ha="center",
            fontsize=11,
            fontweight="bold",
        )
    ax1.set_ylim(0.80, 0.90)
    ax1.set_ylabel("Payment completion rate")
    ax1.set_title("Completion rate: control vs. treatment", fontsize=12.5)

    # Panel 2: uplift vs. predeclared practical-significance threshold.
    ax2.axhline(threshold * 100, color=AMBER_MUTED, linestyle="--", linewidth=1.6, label="Predeclared threshold (+1.0 pp)")
    ax2.errorbar(
        [0],
        [uplift * 100],
        yerr=[[uplift * 100 - ci_low * 100], [ci_high * 100 - uplift * 100]],
        fmt="o",
        markersize=11,
        color=VIOLET,
        ecolor=LAVENDER_LIGHT,
        elinewidth=2,
        capsize=6,
        label="Observed uplift (95% CI)",
    )
    ax2.annotate(
        f"+{uplift * 100:.2f} pp\n(95% CI: +{ci_low * 100:.2f} to +{ci_high * 100:.2f} pp)",
        xy=(0, uplift * 100),
        xytext=(0.35, uplift * 100),
        fontsize=10.5,
        va="center",
    )
    ax2.set_xlim(-1, 1.6)
    ax2.set_ylim(0, 1.3)
    ax2.set_xticks([])
    ax2.set_ylabel("Absolute uplift (percentage points)")
    ax2.set_title("Observed uplift vs. practical threshold", fontsize=12.5)
    ax2.legend(loc="lower right")

    p_str = "p < 0.001" if p_value < 0.001 else f"p = {p_value:.3f}"
    fig.suptitle(
        "Payments A/B Experiment — Statistically Significant, but NO-SHIP",
        fontsize=15,
        fontweight="bold",
        y=1.03,
    )
    fig.text(
        0.5,
        0.965,
        f"Synthetic treatment behavior simulated on real IEEE-CIS transactions, not a live production experiment — "
        f"{p_str}, uplift below the predeclared +1.0 pp practical-significance threshold",
        ha="center",
        fontsize=10.3,
        color=TEXT_SECONDARY,
    )

    add_source_note(
        fig,
        "Source: data/interim/payments_milestone2_experiment_metrics.json (primary_metric_test, experiment_assumptions); "
        "docs/moduleA_milestone2_experimentation.md",
    )

    fig.tight_layout(rect=(0, 0.03, 1, 0.93))
    out = save_figure(fig, "payments_experiment_result.png")
    print(f"Wrote {out}")


if __name__ == "__main__":
    build()
