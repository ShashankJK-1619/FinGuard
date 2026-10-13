# Module C, Milestone 4, Section C — Illustrative Expected-Loss Framework (LGD × EAD)

## Why this doc exists

`configs/credit_risk_expected_loss_assumptions.json` is the
machine-readable source of truth (loaded by
`src/credit_risk/expected_loss.py::load_expected_loss_assumptions`).
This doc is its human-readable companion: it explains *why* the
expected-loss formula is shaped the way it is, and draws an explicit
line between this illustrative framework and any real accounting,
CECL, Basel, or regulatory expected-loss estimate.

## The formula, and what each term is (and is not)

```
Expected Loss = risk_probability × LGD × EAD
```

`src/credit_risk/expected_loss.py::compute_expected_loss(y_prob, ead, lgd)`.

- **`risk_probability`**: Model A or Model B's predicted probability of
  `terminal_adverse_outcome` (Milestone 2/3's target). Per
  `docs/moduleC_milestone4_credit_decisioning.md`, Section A, this is
  the **raw, uncalibrated** model output — recalibration did not
  improve it on validation, so none is applied here. It is **not** a
  12-month probability of default; it is a loan-specific-horizon
  (36/60-month) estimate of eventual adverse resolution.
- **`LGD`** (loss given default): a single configurable scalar, default
  **0.8**, stored under the `lgd` key in
  `configs/credit_risk_expected_loss_assumptions.json`. This is an
  **author-chosen illustrative assumption**, not derived from the real
  `recoveries` or `total_rec_prncp` fields in the LendingClub file.
  Deriving a real empirical LGD from those fields (e.g.
  `1 - recoveries / (principal at charge-off)`) would be a legitimate
  follow-up analysis, but is explicitly out of scope here — this
  milestone keeps LGD a single transparent knob specifically so that
  Section D's Model A vs. Model B comparison and Section G's
  sensitivity analysis can vary it cleanly without re-deriving it from
  outcome data that differs in coverage/quality across vintages.
- **`EAD`** (exposure at default): proxied by `loan_amnt`
  (`EAD_FIELD` constant in `expected_loss.py`) — the original funded
  loan amount at origination. Documented rationale:
  - These are **closed-end installment loans** with no revolving
    redraw feature (unlike a credit card line), so the original funded
    amount is a defensible, simple proxy for exposure at any point in
    the loan's life — there is no mechanism by which exposure could
    exceed the original principal through further draws.
  - `loan_amnt` (not `funded_amnt`/`funded_amnt_inv`) is used for
    consistency with Milestone 3's own feature-selection decision
    (`docs/credit_risk_model_feature_sets.md`), which already excluded
    `funded_amnt`/`funded_amnt_inv` as empirically near-duplicate
    (correlation 0.9998 / 0.9990 with `loan_amnt` on the full real
    file) — using a different amount field here than Model A/B's own
    feature set would be an unexplained inconsistency.
  - A more precise EAD proxy would account for principal already repaid
    by the time of an eventual adverse outcome (declining exposure over
    the loan's life) — this is a real simplification, flagged here,
    not hidden. `loan_amnt` is deliberately the simpler, conservative
    (upper-bound-leaning) choice for an illustrative framework, not a
    claim that it is the most precise possible EAD estimate.

## What this is not

- **Not a CECL/ASC 326 current-expected-credit-loss estimate.** CECL
  requires a reasonable-and-supportable forecast methodology, specific
  loss-rate derivation standards, and auditable documentation this
  milestone does not attempt to meet.
- **Not a Basel regulatory-capital expected-loss calculation.** Basel
  EL requires regulatory-approved PD/LGD/EAD models and downturn-LGD
  conventions this milestone does not implement.
- **Not a claim about real LendingClub portfolio losses.** LGD=0.8 is
  not fit to or validated against this portfolio's actual recovery
  experience.
- **Not a per-loan dollar prediction anyone should act on.** It is a
  transparent, reproducible arithmetic combination of a model output
  and two configurable assumptions, built to support Section D's
  model-to-model *comparison* and Section G's *sensitivity* analysis —
  both of which depend only on EL being computed consistently, not on
  it being numerically "correct" in an absolute sense.

## How Section D and Section G actually use this

Section D (Model A vs. Model B expected-loss comparison) holds LGD and
the EAD definition **identical** for both models and compares: average
EL, EL by risk band, EL by vintage, and rank-correlation between the two
models' per-loan EL arrays. Holding LGD/EAD fixed isolates the
comparison to the one thing that differs between the models — the
probability. Section G (sensitivity analysis,
`src/credit_risk/decision_cost_model.py::sensitivity_analysis`) varies
LGD directly (alongside decisioning thresholds and flat review/
opportunity costs) to show how exposed the downstream illustrative cost
figures are to this one configurable assumption — intentionally making
LGD's influence visible rather than burying a single hard-coded value
inside the cost arithmetic.

## Validation at load time

`CreditDecisionCostAssumptions`-style discipline is mirrored here:
`ExpectedLossAssumptions.__post_init__` raises `ExpectedLossError` if
`lgd` is outside `[0, 1]`; `compute_expected_loss` raises if `y_prob`,
`ead` are mismatched in shape, if `y_prob` falls outside `[0, 1]`, or if
`ead` contains a negative value — fail loudly rather than silently
producing a nonsensical negative or out-of-range expected-loss figure.
