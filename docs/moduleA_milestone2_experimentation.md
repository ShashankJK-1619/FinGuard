# Module A, Milestone 2 — Payments Experimentation & Product Decisioning

## 0. Scope boundary

This milestone builds a **synthetic A/B-test analysis layer** on top of
the Module A, Milestone 1 payment session/attempt model
(`docs/payments_data_model.md`). It does **not** train any machine-learning
model — the guardrail/decision logic below is fixed, documented,
rule-based arithmetic over configured thresholds, exactly like Milestone
1's fraud guardrails. It does not touch Module B's split, features, or
models.

**No real experiment occurred.** There is no real A/B test, no real
treatment, and no real "observed business impact" anywhere in this
milestone. Everything describing an "experiment" below is a **synthetic
scenario invented for this project**, built by (1) deterministically
assigning each already-synthetic-or-real payment session to a "control"
or "treatment" arm, and (2) applying a small, configured, synthetic
treatment-response mechanism to a subset of treatment-arm sessions. The
statistical machinery (two-proportion test, power/MDE, guardrails,
segment heterogeneity, SRM/balance diagnostics) is the real, standard
machinery a payments/product team would use — only the underlying data
is synthetic, and that is stated everywhere.

## 1. The scenario (Section A)

**Hypothesis**: *"A revised checkout retry experience — for example,
clearer in-session messaging and a smoother retry flow after an initial
authorization decline — increases the fraction of payment sessions that
ultimately complete (reach `authorized`), without materially increasing
fraud, fraudulent-completion, legitimate-decline-proxy, or
excessive-retry-rate guardrail metrics."*

This is phrased as a **product hypothesis about a hypothetical UX
change**, not as a description of any real company's actual retry
experience, actual experiment, or actual results. The mechanism used to
simulate "what might happen if the hypothesis were true" is described in
full, with no hidden steps, in §4 below.

## 2. Randomization unit (Section B)

**Chosen unit: `session_id` (session-level randomization).**

- Module A's payment sessions are built on IEEE-CIS real transactions
  plus synthetic abandonment sessions (`docs/payments_data_model.md` §2b,
  §3). There is **no real, documented customer identifier** anywhere in
  the dataset Module A uses — Module B's own data dictionary
  (`docs/data_dictionary.md`) leaves `card4`/`card6`/email-domain fields
  as opaque categorical keys with undisclosed per-column meaning, and
  explicitly forbids treating them as a reliable identity key. Building a
  "customer" proxy out of these fields (e.g. hashing `card4+card6+
  P_emaildomain`) would risk two problems at once: (1) presenting an
  invented identity concept as if it were a real one, which this
  project's working style explicitly disallows, and (2) an unknown and
  unauditable number of hash collisions/misses given the undisclosed
  column semantics. **Session-level randomization is therefore the
  honest choice available with this dataset — not the generally
  preferred choice.** This is recorded as a limitation (§9), not hidden.
- **Why session-level is defensible here anyway**: each payment session
  in this synthetic model is already a self-contained unit — exactly one
  checkout attempt sequence, with its own `attempt_count` and one
  terminal `final_outcome`. The treatment (a revised retry *experience*)
  is naturally experienced once per checkout attempt, so assigning the
  session the moment it starts, and keeping that session's every later
  attempt in the same arm, matches the proposed UX change's actual unit
  of delivery.
- **How contamination could occur with session-level randomization**: if
  the *same underlying person* opens more than one session (e.g. retries
  checkout in a new session after abandoning an earlier one, or on a
  different device), and that person's two sessions land in different
  arms, that person would experience both the old and new retry
  experience across their two attempts. This is a real contamination
  risk whenever the true causal unit (the customer) is more granular
  than the statistical randomization unit — exactly the scenario Section
  B calls out. **Because this dataset has no reliable customer key, this
  risk cannot be measured or bounded here**, only stated. A production
  implementation of this experiment would need a real, stable customer
  or device identifier and would very likely prefer customer-level
  randomization specifically to avoid this failure mode.
- **Why attempt-level randomization would be inappropriate**: a session's
  attempts are not independent trials — they are a sequential retry
  sequence where every attempt after the first is causally downstream of
  the previous attempt's decline (same card, same amount, same session).
  Randomizing the arm *per attempt* would mean the same checkout session
  could flip between the old and new retry experience attempt-to-attempt,
  which is not a coherent version of the proposed product change (a user
  cannot experience "half of a redesigned retry flow"), and would make
  any resulting metric uninterpretable as a treatment effect — it would
  mix within-session arm-switching noise into the estimate.
- **Validated properties** (see `tests/payments/test_experiment_assignment.py`):
  assignment is a pure, deterministic function of `(session_id,
  experiment_seed)` via SHA-256 hashing (`src/payments/
  experiment_assignment.py::unit_hash_uniform`) — stable across reruns,
  invariant to row order, unaffected by mutating any outcome field
  (`final_outcome`, `isFraud`, `attempt_count`) for the same
  `session_id` (no post-treatment information enters assignment),
  approximately balanced at the configured
  `treatment_allocation_fraction` (default 0.5), and produces a
  different assignment under a different `experiment_seed`.

## 3. Assignment mechanism detail (Section C)

`src/payments/experiment_assignment.py::assign_experiment_arms`:

```
draw = SHA256(f"{ASSIGNMENT_SALT}:{experiment_seed}:{session_id}") → uniform float in [0, 1)
assignment = "treatment" if draw < treatment_allocation_fraction else "control"
```

- **Deterministic & reproducible**: same `(session_id, experiment_seed)`
  always hashes to the same draw — no `numpy.random.Generator` state,
  no dependency on call order or DataFrame row order.
- **No post-treatment information**: the draw is a function of
  `session_id` and `experiment_seed` only — never a function of
  `final_outcome`, `isFraud`, `attempt_count`, or any other
  outcome/post-assignment field.
- **Approximately balanced**: validated at both toy scale (±0.15
  tolerance given a few hundred rows,
  `test_assignment_is_approximately_balanced`) and at full real-data
  scale, where the Sample Ratio Mismatch (SRM) check (§7) is the
  rigorous version of this same balance question.
- A separate, independently salted hash (`EFFECT_SALT`, §4) drives the
  synthetic treatment-response draw, so the assignment stream and the
  treatment-effect stream are statistically independent of one another
  even though both derive from the same `session_id`.

## 4. Synthetic treatment-effect mechanism (Section E)

Three layers are kept visibly separate, by design, in every table this
milestone produces:

1. **Baseline data** — the real-linked and synthetic-abandonment sessions
   built in Milestone 1, untouched. `final_outcome` is this layer.
2. **Synthetic randomization** — the `assignment` column from §3. Purely
   a labeling exercise; does not change any session's outcome by itself.
3. **Synthetic treatment response** — `src/payments/experiment_effects.py
   ::apply_synthetic_treatment_effect`, the **only** place any outcome is
   changed, and only under all of these conditions simultaneously:
   - `assignment == "treatment"`
   - `is_synthetic_abandonment == True` (the session has **no real
     transaction** — real transaction history, including every
     real-linked session's `final_outcome`, is never altered, in any
     arm)
   - `attempt_count >= eligible_min_attempt_count` (default 2) — i.e. the
     session already attempted a retry, labeled `retry_eligible`
   - An independent Bernoulli draw (via `EFFECT_SALT`) succeeds with
     probability `recovery_uplift_prob` (default 0.10)

   When all four hold, `experiment_final_outcome` is flipped from
   `abandoned` to `authorized` and `treatment_effect_applied = True` is
   recorded. Every other session's `experiment_final_outcome` is simply
   copied from `final_outcome` unchanged.

`recovery_uplift_prob` is explicitly a **modeling choice**, recorded with
a disclaimer in `configs/payments_experiment_assumptions.json`, not a
measured or claimed real-world effect size. It was fixed before any
analysis was run on real data and was **not** adjusted afterward — see
§10 for the explicit no-tuning-after-looking-at-results commitment.

This single mechanism is deliberately narrow: it only ever *helps*
already-struggling (retried, abandoned) treatment-arm sessions, never
touches control, never touches real transaction history, and leaves
`attempt_count`/`payment_attempts` rows themselves unmutated (a stated
simplification — attempts-table-level effects, e.g. simulating an actual
extra successful retry attempt row, were judged unnecessary to
demonstrate the statistical machinery this milestone targets).

## 5. Metric hierarchy (Section D)

See `docs/payments_experiment_metric_dictionary.md` for the full
numerator/denominator/grain/direction/tier table. Computed in
`src/payments/experiment_metrics.py` (primary/secondary, DuckDB SQL) and
`src/payments/experiment_guardrails.py` (guardrails, DuckDB SQL).

## 6. Statistical analysis (Sections F–G)

- **Two-proportion test** (`src/payments/experiment_stats.py::
  compute_two_proportion_test`): control rate, treatment rate, absolute
  uplift (`treatment - control`), relative uplift, a two-sided z-test
  (`statsmodels.stats.proportion.proportions_ztest`), p-value, and a Wald
  confidence interval on the difference
  (`confint_proportions_2indep(..., method="wald", compare="diff")`).
  Verified against hand-computable toy numbers in
  `tests/payments/test_experiment_analytics.py` (e.g. control 100/1000 vs.
  treatment 150/1000 → uplift exactly 0.05, p < 0.05, CI excludes 0; and
  identical rates → uplift exactly 0, p-value exactly 1.0, CI straddles
  0).
- **Statistical vs. practical significance, explicitly distinguished**:
  a result can be statistically significant (p < alpha) while the
  absolute uplift is too small to be worth the cost/risk of shipping —
  this is exactly why Section H's decision rule requires **both**
  statistical significance **and** a minimum practical uplift
  (`min_practical_uplift_abs`), not significance alone. §8 below shows
  this distinction is not hypothetical — it is the actual outcome of
  this milestone's real-data run.
- **Power / MDE** (`src/payments/experiment_power.py`): `compute_mde_table`
  reports, for a fixed `alpha`/`target_power`/baseline rate, the minimum
  detectable absolute effect at a reference grid of per-arm sample sizes
  (`mde_reference_sample_sizes` in the config) — monotonically decreasing
  as sample size grows (verified in
  `test_mde_shrinks_as_sample_size_grows`). This table is explicitly
  framed as **experiment-planning information** ("given this much
  traffic, what's the smallest effect we could reliably detect") — it is
  **not** a proof of the detected effect, and `compute_achieved_power` is
  likewise labeled a *diagnostic* post-hoc calculation, not a measure of
  how "proven" the finding is (post-hoc power is a known-weak statistic;
  it is reported here only because it was in-scope, not as decision input).

## 7. Guardrails and decision rule (Section H)

- `GUARDRAIL_METRICS = ["fraud_rate", "fraudulent_completed_session_rate",
  "legitimate_decline_proxy_rate", "excessive_retry_rate"]`
  (`src/payments/experiment_guardrails.py`), each with its own configured
  absolute-tolerance threshold in
  `configs/payments_experiment_assumptions.json`.
- **`legitimate_decline_proxy_rate`** is explicitly a **proxy**, not a
  model's false-positive rate: it is the share of real-linked,
  **non-fraudulent** sessions with at least one attempt whose
  `decline_reason_code == "risk_block"`. Because `decline_reason_code` is
  sampled independently of `isFraud` in Milestone 1's generator, this
  metric is expected to show **no real between-arm signal** — any
  observed between-arm difference here is sampling noise, not evidence
  of a fairness or over-blocking effect, and is reported with that
  caveat every time it appears (mirroring Milestone 1's
  `velocity_flag` "methodological artifact" callout style).
- **`fraudulent_completed_session_rate` is expected to equal
  `fraud_rate` exactly in this run**: every real-linked session is
  always `authorized` by construction (Milestone 1), and the treatment
  mechanism (§4) never touches real-linked sessions — so "fraud rate
  among all real-linked sessions" and "fraud rate among *completed*
  real-linked sessions" are the same real-linked population here. This
  is flagged as an expected structural property of this synthetic setup,
  not a new finding.
- **Decision rule** (`evaluate_decision_rule`): `ship` requires **all
  three**:
  1. `primary_significant_and_positive` — the two-proportion test's
     p-value < alpha **and** absolute uplift > 0
  2. `practical_uplift_clears_threshold` — absolute uplift >=
     `min_practical_uplift_abs`
  3. `not any_guardrail_breach` — no guardrail metric's
     (treatment − control) difference exceeds its configured tolerance

  Any failure yields `no-ship`, with a human-readable reason attached per
  failing condition. Unit tests
  (`tests/payments/test_experiment_analytics.py`) directly demonstrate
  both failure branches: a winning, significant primary result is still
  blocked by a guardrail breach, and separately blocked by a uplift that
  is statistically significant but below the practical threshold — see
  §8 for the real-data instance of the latter.

## 8. Experiment-health diagnostics (Section J)

- **Crossover / duplicate-unit check** (`check_no_crossover`): raises
  `ExperimentDiagnosticError` loudly if any `session_id` appears more
  than once in the assigned/experiment table (covers both literal
  duplicate rows and the specific failure mode of the same unit appearing
  in both arms — directly regression-tested in
  `test_check_no_crossover_fails_on_same_unit_in_both_arms`).
- **SRM check** (`compute_srm_check`): a one-sample proportions z-test of
  the observed treatment fraction against
  `treatment_allocation_fraction`, flagging `srm_flag = True` when
  p-value < alpha. Verified to correctly *not* flag a normal, balanced
  split and to correctly flag a deliberately broken 90/10 split
  (`tests/payments/test_experiment_diagnostics.py`).
- **Pre-treatment covariate balance** (`compute_pretreatment_covariate_balance`):
  a DuckDB query comparing mean `TransactionAmt` and the `ProductCD=='W'`,
  `card6=='debit'`, `DeviceType=='mobile'` shares across arms — all
  **pre-treatment** fields (fixed at session-creation time, before any
  treatment-response logic runs), so large between-arm differences here
  would indicate a broken randomization, not a real finding.

## 9. Segment heterogeneity (Section I)

Pre-specified, fixed dimension list only —
`SEGMENT_DIMENSIONS = ["DeviceType", "ProductCD", "card6",
"retry_eligible"]` (`src/payments/experiment_segments.py`). Any other
column raises `ExperimentSegmentError` (regression-tested) — this is a
deliberate allow-list, not an open-ended "slice everything" tool.

**Multiple-testing risk, stated explicitly**: running the same
two-proportion test across 4 dimensions × several values each (14
segment rows in §8 of the report below) inflates the chance that at
least one segment shows p < 0.05 by chance alone, even if the true
subgroup effect were identical everywhere. **No multiple-comparison
correction (e.g. Bonferroni) is applied** — this is a deliberate choice
appropriate to a small, pre-registered set of business-relevant
dimensions, not a data-dredged search, but it means **no individual
segment row below should be read as its own independently confirmed
discovery** — only the pre-specified, whole-dimension pattern (e.g. "is
the effect concentrated in retry-eligible sessions, as the mechanism
would predict?") is a defensible read. No subgroup-specific causal claim
is made anywhere in this milestone.

## 10. No post-hoc tuning (explicit commitment)

`configs/payments_experiment_assumptions.json`'s `recovery_uplift_prob`,
`min_practical_uplift_abs`, and guardrail tolerances were fixed before
`src/payments/build_experiment_report.py` was run against the real IEEE-
CIS data for this milestone's report, and were **not** adjusted afterward
to change the ship/no-ship outcome. The real-data run's `no-ship` result
(driven by uplift falling below the practical threshold, not by any
guardrail breach — see the final report) is reported as-is.

## 11. Limitations (full list)

- **Session-level randomization, not customer-level**, because this
  dataset provides no real, documented customer identifier (§2) — any
  single real person who opens multiple sessions could be split across
  arms, contaminating the estimate in a way this dataset cannot measure.
- **The entire treatment-response mechanism is synthetic** (§4) — no
  real retry-experience redesign exists, was tested, or was observed to
  have any effect. `recovery_uplift_prob` is an illustrative modeling
  choice, not a measured effect size.
- **`legitimate_decline_proxy_rate` is a proxy construct**, not a real
  false-positive-rate measurement from any model or manual review
  process, and is expected to show no real between-arm signal in this
  synthetic setup (§7).
- **`payment_attempts` rows are not mutated** when the treatment effect
  flips a session's outcome — only the session-level
  `experiment_final_outcome` column changes. `attempts_per_completed_session`
  and `retry_recovery_rate` at the session level still move sensibly
  because they are computed from `attempt_count`/`final_outcome`
  relationships already present in the baseline data, but a hypothetical
  "attempt 3 now exists and is authorized" row is never added.
  attempts-level is sliced on this.
- **No multiple-comparison correction is applied to the segment
  heterogeneity table** (§9) — appropriate for 4 pre-registered
  dimensions, not safe to extend to an open-ended search without adding
  one.
- **Post-hoc achieved power is reported only as a diagnostic**, not as
  evidence of how "proven" the result is — it is a well-known weak
  statistic when computed after seeing the data (§6).
- **No real experiment occurred.** This entire milestone is a
  demonstration of experimentation statistics/engineering practice on
  top of Milestone 1's synthetic payments layer, not evidence about any
  real product, company, or user behavior.
