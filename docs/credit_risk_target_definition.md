# Module C, Milestone 1 — Target Definition, Censoring, and Loss Feasibility (Section D)

## Scope boundary

This document specifies the `default_12m` target definition implemented
in `src/credit_risk/target_definition.py` and validated only against toy
data in `tests/credit_risk/test_target_definition.py`. No real loan
performance data has been computed on yet — see
`docs/credit_risk_dataset_comparison.md` for the dataset recommendation
this logic will eventually be applied to, in a later milestone.

## D1. The exact `default_12m` definition

For a loan with `origination_date` o, a configured outcome window of
`outcome_window_months` months (default: 12), and a `data_as_of_date` d
(the latest date through which this loan's performance has actually been
observed):

```
observation_window_end = o + outcome_window_months (calendar months)
sufficient_followup     = d >= observation_window_end

default_12m = 1   if default_event_date is set AND default_event_date <= observation_window_end
            = 0   if default_12m is not 1 AND sufficient_followup is True
            = NA  otherwise (label_status = "censored_insufficient_followup")
```

`label_status` is one of three mutually exclusive values:
`"default"`, `"non_default"`, or `"censored_insufficient_followup"` —
every row gets exactly one, never left ambiguous.

This is a fixed-length performance-window definition (a loan either did
or did not reach a default-severity status within exactly N months of
its own origination date), not a fixed-calendar-date cross-section —
each loan's window is anchored to its own origination date, so two loans
originated a year apart are each judged against their own 12-month
clock, not a shared cutoff.

## D2. Window-boundary convention: inclusive

A default event landing *exactly* on `observation_window_end` counts as
"within the window" (`default_12m = 1`), not outside it. This is a
deliberate, documented choice, not an oversight: the alternative
(exclusive boundary) would silently reclassify a loan that defaulted on
day 365 of a 12-month window as a "survivor," which both overstates
model performance on boundary cases and is an arbitrary place to draw
the line given that "12 months" is itself a planning convention, not a
law of nature. `test_default_exactly_at_window_boundary_is_inclusive_default`
is the regression guard for this convention.

## D3. Censoring: why "no default observed yet" is not the same as "non-default"

The core requirement this logic enforces: **a loan that has not
defaulted *yet* but also has not been observed long enough to rule out a
default occurring later in the window must never be silently coded as a
confirmed non-default (`default_12m = 0`).** Doing so would bias the
label distribution toward the (recent, often still-performing)
loans that happen to sit near the end of whatever extract the data comes
from — undercounting defaults for the newest originations purely because
they haven't had time to default yet, not because they're actually
lower-risk. This is the lending-domain analogue of right-censoring in
survival analysis.

Concretely: a loan originated in month 0 with `data_as_of_date` at month
6 and no default event recorded is `label_status =
"censored_insufficient_followup"`, `default_12m = NA` — not `0`. Only
once `data_as_of_date` reaches or passes `observation_window_end` (here,
month 12) without a recorded default does the loan earn a confirmed `0`.
`test_no_default_with_insufficient_followup_is_censored_not_coded_zero`
is the direct regression guard.

A practical consequence for later milestones: any model-training sample
built from this logic must either exclude censored rows or handle them
with an explicit survival/censoring-aware method — training a plain
binary classifier on `default_12m` while silently dropping NA rows is
fine (it just restricts the sample to loans with resolved outcomes);
silently filling NA with 0 is the exact bug this design prevents.

## D4. Delinquency and default are separate severities, never conflated

`delinquency_event_date` is carried through the output for visibility
only — it is never read when computing `default_12m`
(`test_delinquency_event_alone_does_not_trigger_default`). This mirrors
the lifecycle distinction in `docs/credit_risk_data_model.md` Section A:
delinquency is explicitly recoverable (a borrower can cure and return to
current status), default is the configured severe/terminal state this
module's target is built around. A loan that experiences a lesser
delinquency event but neither defaults nor runs out of follow-up within
the window is `label_status = "non_default"`, same as a loan with no
events at all.

## D5. Configuration, not hardcoding

All of the above thresholds live in
`configs/credit_risk_target_definition.json`
(`TargetDefinitionAssumptions`, loaded via
`load_target_definition_assumptions`), validated loudly in
`__post_init__`:

| Field | Current value | Validated as |
|---|---|---|
| `outcome_window_months` | 12 | must be >= 1 |
| `default_event_statuses` | `["Charged Off", "Default"]` | must be non-empty, non-empty strings |
| `delinquency_event_statuses` | `["In Grace Period", "Late (16-30 days)", "Late (31-120 days)"]` | must be non-empty strings |
| `minimum_followup_days_tolerance` | 0 | must be >= 0 |

`default_event_statuses` and `delinquency_event_statuses` are enforced
**disjoint** — overlap raises `CreditRiskConfigError`, because default
and delinquency are tracked as separate severities by design (Sec D4);
a status cannot simultaneously be "the lesser recoverable state" and
"the severe terminal state" in this scheme.

The config's own `_disclaimer` field states explicitly that this status
vocabulary reflects the recommended candidate dataset's (LendingClub)
*publicly documented* `loan_status` values from secondary sources, and
**must be verified against the actual raw file's data dictionary**
before any real computation is run on it — this has not happened yet,
per this milestone's scope boundary.

## D6. Required input contract

`compute_default_12m_labels` requires `loan_id`, `origination_date`,
`data_as_of_date`, and `default_event_date` columns
(`test_rejects_missing_required_column`), and rejects any row where
`data_as_of_date < origination_date`
(`test_rejects_data_as_of_date_before_origination_date`) — a loan cannot
have negative days of follow-up; this is treated as a data-quality error
worth failing loudly on, not silently clamping.

## F. PD / LGD / EAD feasibility from the recommended dataset (planning-level, not yet computed)

Credit-loss modeling is conventionally decomposed into three components:
**PD** (probability of default), **LGD** (loss given default — the
fraction of exposure not recovered after a default), and **EAD**
(exposure at default — the outstanding balance at the moment of
default). None of these are modeled in this milestone; this section only
assesses, at a planning level, what the recommended LendingClub dataset
(Section C) would and would not support in a later milestone.

- **PD — feasible, and the primary fit for this dataset.** `default_12m`
  as defined above is a direct, derivable PD target from `loan_status`
  plus an extract/`data_as_of_date` for censoring. This is the
  dataset's strongest and most direct use.
- **LGD — feasible only as an illustrative, after-the-fact calculation,
  not a point-in-time-safe model input.** LendingClub's
  `recoveries`/`collection_recovery_fee`/`total_rec_prncp` fields (all
  flagged post-default in `configs/credit_risk_feature_eligibility.json`)
  could support a *retrospective* "of the loans that defaulted, what
  fraction of principal was ultimately recovered" calculation — useful
  as a descriptive loss-severity illustration — but by construction
  these fields only exist *after* default has occurred, so they can
  never be inputs to an application-time underwriting model. Any LGD
  "model" built from them would have to be a separate, explicitly
  post-default-conditional exercise (e.g. predicting ultimate severity
  *given* that default has already happened), not a PD-style
  underwriting feature pipeline.
- **EAD — feasible but limited and largely degenerate for this
  product.** For LendingClub's fixed-term, fully amortizing installment
  loans with no revolving redraw feature, exposure at any point in the
  loan's life is close to a deterministic function of origination terms
  and elapsed scheduled amortization (approximable from
  `scheduled_payments`, Section B4) rather than a genuinely uncertain
  quantity the way it would be for a revolving product (e.g. a credit
  line that can be redrawn before default). EAD modeling is more
  interesting for revolving-credit datasets than for this
  fixed-installment product; it is listed here as *technically
  computable* but low-value as a modeling exercise for this specific
  dataset.

**Summary**: this module's Milestone 1–2 modeling effort should treat PD
(`default_12m`) as the actual target; LGD/EAD are documented here as
dataset-feasibility findings, not as committed future milestones.

## G. Chronological split feasibility (planning-level, not yet implemented)

No split has been implemented in this milestone — this section only
records why a chronological (time-based) split is feasible and why it
is the correct choice to implement when a real dataset is downloaded,
mirroring the time-based-split reasoning already applied in Module B
(`docs/time_structure.md`).

- **Why chronological, not random**: `origination_date` is a real,
  ordered timestamp spanning LendingClub's multi-year release window
  (2007–2018Q4 per Section C). A random split would let the model see
  loans originated *after* the loans in its "test" set during training
  — directly backwards for a production underwriting use case, where a
  model can only ever be deployed on applications arriving after its
  training data ends. A chronological split (e.g. train on loans
  originated through some cutoff date, validate/test on loans
  originated after it) is the only split that honestly simulates
  real deployment.
- **A feasibility-specific wrinkle this dataset introduces**: because
  `default_12m` requires a full 12-month observation window
  (`sufficient_followup`, Sec D3) to resolve a confirmed label, loans
  originated too close to the dataset's own most recent extract date
  will be censored, not confirmed non-defaults. A chronological split
  must therefore pick its train/validation/test cutoffs *with the
  12-month resolution lag in mind* — e.g. a test set of "loans
  originated in the last 12 months of the file" would consist
  overwhelmingly of censored rows, which is a label-availability
  problem, not a leakage problem, but would still make that test set
  useless as constructed. This is flagged here as a concrete planning
  consideration for whoever implements the split in a later milestone,
  not solved in this one.
- **Feasibility verdict**: chronological splitting is feasible and
  recommended; the only real risk is picking cutoffs naively without
  accounting for the censoring lag above.
</content>
