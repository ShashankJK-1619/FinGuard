# Module C, Milestone 2, Section C — Real `loan_status` Mapping

## Why this doc exists

Milestone 1 designed the default/delinquency severity split against a
secondary-source-documented guess at LendingClub's `loan_status`
vocabulary. This milestone replaces that guess with the **empirically
observed** vocabulary from the real downloaded file (`SELECT DISTINCT
loan_status` against `loans_raw`, 2,260,668 clean rows) and maps every
single value explicitly, with a documented rationale per value. Source
of truth: `configs/credit_risk_status_mapping.json`; enforced by
`src/credit_risk/status_mapping.py`.

## Design discipline

- **Every real `loan_status` value is mapped explicitly.**
  `map_loan_status`/`map_status_series` raise `StatusMappingError` on any
  value not present in the config — a previously-unseen status string
  (e.g. introduced by a future LendingClub vintage) must be reviewed and
  added by a human. It is never silently bucketed into an
  "ambiguous_other" catch-all or dropped.
- **`category` (a descriptive severity bucket for reporting) is
  deliberately decoupled from `default_event_statuses`/
  `delinquency_event_statuses`** in `configs/
  credit_risk_target_definition.json`, which is what actually sets
  `terminal_adverse_outcome` (primary target)/`default_12m_last_pymnt_proxy`
  (secondary target). A status can
  share a category with another status and still be excluded from the
  label-triggering set — see `Late (31-120 days)` below.
- **`meets_credit_policy` is tracked separately from outcome category.**
  LendingClub's "Does not meet the credit policy." prefix describes
  *underwriting-policy vintage* (the loan was issued under criteria
  LendingClub later tightened beyond), not a different repayment
  outcome — conflating the two would misclassify otherwise-identical
  outcomes as different severities purely because of when/how the loan
  was underwritten.

## The 9 real values

| loan_status | category | meets_credit_policy | in default_event_statuses? | row count |
|---|---|---|---|---|
| Current | current_performing | true | no | 878,317 |
| In Grace Period | delinquent | true | no | 8,436 |
| Late (16-30 days) | delinquent | true | no | 4,349 |
| Late (31-120 days) | severe_delinquency_or_default | true | **no** (see note) | 21,467 |
| Default | severe_delinquency_or_default | true | yes | 40 |
| Charged Off | charged_off | true | yes | 268,559 |
| Fully Paid | fully_paid | true | no | 1,076,751 |
| Does not meet the credit policy. Status:Fully Paid | fully_paid | **false** | no | 1,988 |
| Does not meet the credit policy. Status:Charged Off | charged_off | **false** | yes | 761 |

Total: 2,260,668 rows (matches the full clean post-footer-filter row
count exactly — every row has a mapped status).

## Rationale highlights

- **`Late (31-120 days)` shares its category with `Default` but is
  excluded from the trigger set.** It is a severe, but not-yet-terminal,
  delinquency state approaching LendingClub's documented ~120-150 day
  charge-off policy threshold — the loan has not yet been charged off
  and could still cure or progress to an actual terminal outcome.
  Treating it as a confirmed default would pre-empt that eventual, still
  genuinely unknown resolution. See its `target_trigger_note` in
  `configs/credit_risk_status_mapping.json`.
- **`Default` (40 rows) is LendingClub's own explicit determination**,
  distinct from a mere lateness band, and is included in
  `default_event_statuses`.
- **`In Grace Period` / `Late (16-30 days)` are lesser delinquency
  signals**, carried for visibility only — consistent with Milestone
  1's delinquency/default separation principle, a lesser delinquency
  event never by itself flips a default label.
- **The two "Does not meet the credit policy." statuses map to the
  SAME category as their unprefixed counterpart** (`fully_paid` /
  `charged_off` respectively) and the charged-off variant is included in
  `default_event_statuses` exactly like plain `Charged Off` — it is a
  real charge-off, just one issued under since-tightened underwriting
  criteria.

## Consumers

- `src/credit_risk/target_construction.py::load_loans_base_table` joins
  every loan to its `status_category`/`meets_credit_policy` via
  `map_status_series`.
- `configs/credit_risk_target_definition.json`'s `default_event_statuses`
  / `delinquency_event_statuses` consume the raw `loan_status` strings
  directly (not the `category` field) when constructing
  `default_12m_last_pymnt_proxy`.
- `compute_terminal_adverse_outcome_target` (the **primary** target,
  post-governance-correction — see `docs/decision_log.md`) consumes
  `status_category` directly (`charged_off` / `fully_paid`) since it
  does not need the finer-grained trigger distinction the secondary
  12-month proxy target does.
