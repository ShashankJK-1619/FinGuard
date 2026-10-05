"""Module A, Milestone 2: two-proportion statistical inference.

See docs/moduleA_milestone2_experimentation.md Section F. Uses
`statsmodels.stats.proportion` (already a project dependency, see
requirements.txt — no new dependency added) rather than hand-rolling a
z-test:
  - `proportions_ztest` for the two-sided hypothesis test (H0: the two
    population proportions are equal).
  - `confint_proportions_2indep` (Wald method) for the confidence
    interval on the difference of two proportions (treatment - control).

Statistical significance vs. practical significance (explained again in
the milestone report, not just here): a p-value below alpha only says the
observed difference is unlikely under the null of no difference — it says
nothing about whether that difference is large enough to matter
operationally. With enough sample size, a trivially small uplift (e.g.
+0.01pp) can be "significant" while being commercially irrelevant; that is
exactly why the ship/no-ship decision rule (experiment_guardrails.py)
requires BOTH statistical significance AND a separately predeclared
minimum practical uplift, not significance alone.
"""
from __future__ import annotations

from statsmodels.stats.proportion import confint_proportions_2indep, proportions_ztest


def compute_two_proportion_test(
    n_control: int,
    n_completed_control: int,
    n_treatment: int,
    n_completed_treatment: int,
    alpha: float,
) -> dict:
    """Two-sided two-proportion z-test + Wald CI for (treatment - control).
    `n_*` are per-arm session counts; `n_completed_*` are the counts of
    sessions with the outcome of interest (e.g. payment_completion_rate's
    numerator) in each arm."""
    if n_control == 0 or n_treatment == 0:
        raise ValueError(f"both arms must be non-empty; got n_control={n_control}, n_treatment={n_treatment}")

    control_rate = n_completed_control / n_control
    treatment_rate = n_completed_treatment / n_treatment
    absolute_uplift = treatment_rate - control_rate
    relative_uplift = absolute_uplift / control_rate if control_rate else float("nan")

    zstat, p_value = proportions_ztest(
        count=[n_completed_treatment, n_completed_control],
        nobs=[n_treatment, n_control],
    )
    ci_low, ci_high = confint_proportions_2indep(
        count1=n_completed_treatment,
        nobs1=n_treatment,
        count2=n_completed_control,
        nobs2=n_control,
        method="wald",
        compare="diff",
        alpha=alpha,
    )
    return {
        "n_control": n_control,
        "n_completed_control": n_completed_control,
        "control_rate": control_rate,
        "n_treatment": n_treatment,
        "n_completed_treatment": n_completed_treatment,
        "treatment_rate": treatment_rate,
        "absolute_uplift": absolute_uplift,
        "relative_uplift": relative_uplift,
        "zstat": float(zstat),
        "p_value": float(p_value),
        "alpha": alpha,
        "ci_low": float(ci_low),
        "ci_high": float(ci_high),
        "is_statistically_significant": bool(p_value < alpha),
    }
