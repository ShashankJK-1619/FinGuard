"""Visual 6 -- credit calibration drift (the strongest governance visual).

Shows the Cox calibration slope for Model A and Model B, validation vs.
temporal benchmark, against the slope=1.0 perfect-calibration reference
line, plus each model's PSI (train vs. validation / benchmark) to make
visible that score-distribution stability (low PSI) did NOT protect
probability calibration from degrading substantially out-of-time.
"""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from src.portfolio.metrics_config import (
    CREDIT_MODEL_A_PSI_BENCHMARK,
    CREDIT_MODEL_A_PSI_VALIDATION,
    CREDIT_MODEL_A_SLOPE_BENCHMARK,
    CREDIT_MODEL_A_SLOPE_VALIDATION,
    CREDIT_MODEL_B_SLOPE_BENCHMARK,
    CREDIT_MODEL_B_SLOPE_VALIDATION,
)
from src.portfolio.style import BORDER, CAUTION, GRID, TEXT_SECONDARY, VIOLET, add_source_note, apply_style, save_figure


def build() -> None:
    apply_style()

    models = ["Model A", "Model B"]
    val_slopes = [CREDIT_MODEL_A_SLOPE_VALIDATION.value, CREDIT_MODEL_B_SLOPE_VALIDATION.value]
    bench_slopes = [CREDIT_MODEL_A_SLOPE_BENCHMARK.value, CREDIT_MODEL_B_SLOPE_BENCHMARK.value]

    x = np.arange(len(models))
    width = 0.32

    fig, ax = plt.subplots(figsize=(9.5, 6))
    ax.set_xlim(-0.6, 1.6)
    ax.axhline(1.0, color=TEXT_SECONDARY, linestyle="--", linewidth=1.3, zorder=1)
    ax.annotate(
        "Perfect calibration (slope = 1.0)",
        xy=(0.02, 1.0),
        xycoords=ax.get_yaxis_transform(),
        xytext=(0, 5),
        textcoords="offset points",
        ha="left",
        fontsize=9.5,
        color=TEXT_SECONDARY,
    )

    bars_val = ax.bar(x - width / 2, val_slopes, width, label="Validation", color=VIOLET, zorder=2)
    bars_bench = ax.bar(x + width / 2, bench_slopes, width, label="Temporal benchmark (out-of-time)", color=CAUTION, zorder=2)

    for bars in (bars_val, bars_bench):
        for bar in bars:
            h = bar.get_height()
            ax.annotate(
                f"{h:.2f}",
                xy=(bar.get_x() + bar.get_width() / 2, h),
                xytext=(0, 5),
                textcoords="offset points",
                ha="center",
                fontsize=12,
                fontweight="bold",
            )

    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=12.5)
    ax.set_ylabel("Calibration slope (Cox, higher = closer to perfectly calibrated)")
    ax.set_ylim(0, 1.15)
    ax.legend(loc="upper right")

    fig.suptitle("Credit Calibration Drift — Validation vs. Temporal Benchmark", fontsize=15, fontweight="bold", y=1.04)
    fig.text(
        0.5,
        0.965,
        "Stable score distributions do not guarantee stable probability calibration",
        ha="center",
        fontsize=11,
        color=TEXT_SECONDARY,
        style="italic",
    )

    psi_val = CREDIT_MODEL_A_PSI_VALIDATION.value
    psi_bench = CREDIT_MODEL_A_PSI_BENCHMARK.value
    ax.text(
        0.02,
        0.03,
        f"For reference, Model A's score-distribution PSI stayed low and stable over the same period\n"
        f"(train vs. validation: {psi_val:.3f}; train vs. benchmark: {psi_bench:.3f} — both well under common 0.1/0.25 drift flags)\n"
        f"while its calibration slope fell from {CREDIT_MODEL_A_SLOPE_VALIDATION.value:.2f} to {CREDIT_MODEL_A_SLOPE_BENCHMARK.value:.2f}.",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=9.3,
        color=TEXT_SECONDARY,
        bbox=dict(boxstyle="round,pad=0.4", facecolor=GRID, edgecolor=BORDER),
    )

    add_source_note(
        fig,
        "Source: data/interim/credit_risk_milestone4_metrics.json -> "
        "models.{model_a_application_time_safe,model_b_with_lender_grade_signal}."
        "{calibration_slope_intercept_validation,calibration_slope_intercept_benchmark,psi_train_vs_validation,psi_train_vs_benchmark}; "
        "docs/moduleC_milestone4_credit_decisioning.md",
    )

    fig.tight_layout(rect=(0, 0.04, 1, 0.9))
    out = save_figure(fig, "credit_calibration_drift.png")
    print(f"Wrote {out}")


if __name__ == "__main__":
    build()
