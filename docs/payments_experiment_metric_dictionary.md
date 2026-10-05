# Module A, Milestone 2 — Payments Experiment Metric Dictionary

All metrics below are computed per `assignment` arm (`control` /
`treatment`) over the experiment session table
(`src.payments.duckdb_views.register_experiment_sessions_view`, DuckDB
view `payment_experiment_sessions`). "Session" always means one row of
the Milestone 1 `payment_sessions` table, carrying the Milestone 2
`assignment`, `retry_eligible`, `treatment_effect_applied`, and
`experiment_final_outcome` columns (`src/payments/schema.py`). See
`docs/moduleA_milestone2_experimentation.md` for full scenario context —
**every row below describes a synthetic scenario; none of these numbers
are measurements of any real payments product.**

## Primary metric

| Metric | Numerator | Denominator | Grain | Direction | Interpretation |
|---|---|---|---|---|---|
| `payment_completion_rate` | sessions with `experiment_final_outcome == 'authorized'` | all sessions in the arm | session | higher is better | The headline measure of checkout success under this milestone's synthetic experiment — whether the session ultimately authorized, accounting for both the baseline funnel and the synthetic treatment-response mechanism. |

## Secondary metrics

| Metric | Numerator | Denominator | Grain | Direction | Interpretation |
|---|---|---|---|---|---|
| `first_attempt_authorization_rate` | sessions whose attempt #1 had `outcome == 'authorized'` | all sessions in the arm | session (keyed off attempt #1) | higher is better | How often the very first authorization attempt succeeds. By construction, unaffected by the treatment mechanism (which only ever changes a session's *final* outcome after retries) — expected to be statistically indistinguishable between arms, and a useful internal-consistency check if it is not. |
| `retry_rate` | sessions with `attempt_count > 1` | all sessions in the arm | session | context-dependent (not inherently better/worse — a secondary diagnostic, not an optimization target on its own) | How often a session needed more than one attempt. Driven entirely by Milestone 1's baseline retry-sampling, not by the treatment mechanism — expected to be stable across arms. |
| `retry_recovery_rate` | sessions with `attempt_count > 1` AND `experiment_final_outcome == 'authorized'` | sessions with `attempt_count > 1` | session | higher is better | Of sessions that needed a retry, how many still completed. This is the metric most directly, mechanically moved by the treatment effect (§4 of the milestone doc), since the mechanism only ever flips already-retried, previously-abandoned treatment sessions to `authorized`. |
| `attempts_per_completed_session` | sum of `attempt_count` over sessions with `experiment_final_outcome == 'authorized'` | count of sessions with `experiment_final_outcome == 'authorized'` | session | lower is better (fewer attempts needed per success is more efficient) | Average effort (in attempts) behind each eventual success. Expected to tick up slightly in treatment, since the mechanism specifically converts previously-abandoned, higher-`attempt_count` sessions into completions. |

## Guardrail metrics

All guardrail metrics are evaluated as `treatment − control`
against a configured absolute tolerance
(`configs/payments_experiment_assumptions.json`); a difference exceeding
the tolerance in the "worse" direction is a **breach**, regardless of
how the primary metric performed (`src/payments/experiment_guardrails.py
::evaluate_decision_rule`).

| Metric | Numerator | Denominator | Grain | Direction | Interpretation |
|---|---|---|---|---|---|
| `fraud_rate` | real-linked sessions with `isFraud == 1` | all real-linked sessions in the arm | session (real-linked subset only — `isFraud` does not exist for synthetic-abandonment sessions) | lower is better | Baseline fraud incidence among sessions with a real fraud label. The treatment mechanism never touches real-linked sessions (§4 of the milestone doc), so this is expected to be statistically indistinguishable between arms — a between-arm difference here would suggest an assignment or pipeline bug, not a real safety finding. |
| `fraudulent_completed_session_rate` | real-linked sessions with `isFraud == 1` AND `experiment_final_outcome == 'authorized'` | all real-linked sessions in the arm | session (real-linked subset only) | lower is better | Fraud that not only occurred but *completed*. Expected to equal `fraud_rate` exactly in this run, since every real-linked session is authorized by construction and the treatment mechanism never alters real-linked sessions — a stated structural property (§7 of the milestone doc), not a new finding. |
| `legitimate_decline_proxy_rate` | real-linked, non-fraudulent (`isFraud == 0`) sessions with at least one attempt where `decline_reason_code == 'risk_block'` | all real-linked, non-fraudulent sessions in the arm | session (real-linked, non-fraudulent subset only) | lower is better | A **proxy** for "legitimate customers wrongly blocked by a risk rule" — not a real false-positive-rate measurement from any model or manual review process. `decline_reason_code` is sampled independently of `isFraud` in Milestone 1, so no real between-arm signal is expected here; reported with this caveat every time it appears. |
| `excessive_retry_rate` | sessions with `attempt_count >= excessive_retry_attempt_threshold` (default 3) | all sessions in the arm | session | lower is better | A guardrail against a redesigned retry flow perversely encouraging excessive repeated attempts (a plausible real risk-ops concern for this kind of UX change) rather than genuinely helping users succeed sooner. |

## Experiment-health diagnostics (not ship/no-ship inputs, but reported alongside)

| Metric | What it measures | Interpretation |
|---|---|---|
| `n_sessions` / `n_unique_session_ids` (crossover check) | Row count vs. unique `session_id` count | Must be equal; any gap means the same unit appears more than once (duplicate row or same unit in both arms) and `ExperimentDiagnosticError` is raised rather than silently continuing. |
| `srm_p_value` / `srm_flag` (SRM check) | One-sample z-test of observed treatment fraction vs. configured `treatment_allocation_fraction` | `srm_flag = True` (p < alpha) would indicate a broken or biased randomization and should block trusting any downstream result, independent of what the primary metric shows. |
| Pre-treatment covariate balance (mean `TransactionAmt`, `ProductCD=='W'` share, `card6=='debit'` share, `DeviceType=='mobile'` share) | Whether arms look alike on fields fixed before any treatment response | All of these are pre-treatment by construction; large between-arm gaps would indicate a broken randomization, not a real covariate imbalance "finding." |

## Segment heterogeneity (pre-specified dimensions only)

Computed identically to the primary metric's two-proportion test, but
grouped by one of exactly four pre-registered dimensions
(`SEGMENT_DIMENSIONS` in `src/payments/experiment_segments.py`):
`DeviceType`, `ProductCD`, `card6`, `retry_eligible`. See
`docs/moduleA_milestone2_experimentation.md` §9 for the multiple-testing
caveat — no dimension beyond these four is computed, and no individual
segment-level p-value below the table should be read as an independently
confirmed discovery.
