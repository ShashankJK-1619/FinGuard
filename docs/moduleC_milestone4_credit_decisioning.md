# Module C, Milestone 4 — Credit Decisioning, Expected Loss & Probability Governance

## 0. Scope boundary

This milestone takes Model A (`model_a_application_time_safe`) and
Model B (`model_b_with_lender_grade_signal`) exactly as fit in Milestone
3 — same frozen primary chronological split, same preprocessing, same
`LogisticRegression(C=1.0)` specification, refit once each by directly
calling Milestone 3's own fitting function
(`run_milestone3_baseline._fit_and_evaluate_one_model`), never retuned
— and builds four new, additive layers on top: (A) a probability-quality
re-review, (B) illustrative risk bands, (C/D) an illustrative LGD×EAD
expected-loss framework and a Model A vs. Model B economic-ranking
comparison, (E/F/G) an illustrative three-way decisioning policy with a
configurable cost model and sensitivity analysis, and (I/J) vintage/
stability governance diagnostics. **None of this is a production credit
decisioning system.** No boosted trees/random forest are used. No
approve/decline recommendation is made for any real applicant. No
rejected-applicant outcome is inferred (Section H). Every number below
comes from one real, reproducible run of
`src/credit_risk/run_milestone4_decisioning.py` against the real
2,260,668-row LendingClub file; none is estimated. Full stdout and the
JSON reproducibility artifact
(`data/interim/credit_risk_milestone4_metrics.json`) back every figure
quoted here.

Reproduce with:
```bash
.venv/bin/python -m src.credit_risk.run_milestone4_decisioning
```

**Target framing, restated (hard constraint carried from Milestone 3):**
`terminal_adverse_outcome` is **NOT** a 12-month probability of default.
It estimates the probability of an eventual adverse terminal outcome
among originated loans with sufficient lifecycle resolution, over a
loan-specific horizon (36 or 60 months to maturity). This note is
embedded machine-readably in the JSON artifact as `target_framing_note`,
not only in prose.

## A. Probability-quality review (on the existing frozen split)

Milestone 3 already established (and this milestone does not refit)
Brier scores and a sigmoid-vs-isotonic calibration comparison for both
models on validation:

| | Model A | Model B |
|---|---|---|
| n_train | 42,535 | 42,535 |
| Uncalibrated Brier (validation) | 0.128637 | 0.128466 |
| Sigmoid Brier (validation) | 0.129002 | 0.128497 |
| Isotonic Brier (validation) | 0.128686 | 0.128351 |
| Chosen method (Milestone 3 rule) | sigmoid | sigmoid |

For both models the **uncalibrated** base model already has the lowest
Brier score of the three candidates on validation — neither
recalibration map clears the predeclared
`ISOTONIC_PREFERENCE_BRIER_IMPROVEMENT = 0.0005` margin over sigmoid,
and sigmoid itself does not actually improve on the raw, uncalibrated
probability. Per this milestone's own hard constraint ("keep uncalibrated
probabilities if best — do not force calibration if it doesn't improve
out-of-sample quality"), **every downstream Milestone 4 calculation
(risk bands, expected loss, decisioning, cost model, governance
diagnostics) uses Model A/B's raw, uncalibrated predicted probability**,
not a sigmoid- or isotonic-recalibrated variant.

This milestone adds a second calibration diagnostic — a Cox calibration
regression (`logit(y_prob) ~ intercept + slope·y_true`,
`calibration_slope_intercept` in `calibration.py`) — computed on both
validation and benchmark (Milestone 3 only assessed validation):

| | Model A — validation | Model A — benchmark | Model B — validation | Model B — benchmark |
|---|---|---|---|---|
| n | 53,367 | 134,804 | 53,367 | 134,804 |
| intercept | 0.0138 | -0.9540 | -0.2035 | -0.8667 |
| slope | 0.9494 | 0.4211 | 0.8510 | 0.4981 |

Read honestly: on validation, both models' calibration slopes are close
to 1 (0.95 / 0.85), consistent with the already-good validation Brier
scores above. On **benchmark**, both models' slopes drop sharply (0.42 /
0.50) with a large negative intercept — a real, out-of-time calibration
degradation that the Brier score alone does not surface as starkly
(benchmark Brier, from Milestone 3, is actually slightly *better* than
validation Brier for both models: 0.1267/0.1265 vs. 0.1286/0.1285).
Slope < 1 with a negative intercept means predicted probabilities are
systematically **too spread out relative to benchmark's true log-odds**
— the model is overconfident at the extremes on the 2013 benchmark year
relative to how it was fit on 2007-2011 train. This is flagged here as
an honest finding, not smoothed over, and is picked up again in Section
I's vintage analysis and Section J's governance discussion.

## B. Risk bands (low / medium / high, boundaries fit on train only)

`src/credit_risk/risk_bands.py::fit_risk_band_boundaries` draws
tertile-quantile cut points **from train's predicted probabilities
only** (`prob_train`, 42,535 rows); validation/benchmark rows are only
ever bucketed into these already-fit edges, never used to redraw them.
These are illustrative, transparent bands — **not real LendingClub
underwriting cutoffs**.

| | Model A | Model B |
|---|---|---|
| low/medium boundary | 0.09543 | 0.09436 |
| medium/high boundary | 0.16732 | 0.17087 |
| n_train | 42,535 | 42,535 |

**Model A — validation**:

| risk_band | n | observed adverse-outcome rate | mean predicted probability | calibration gap |
|---|---|---|---|---|
| low | 14,383 | 7.231% | 6.782% | -0.448 pts |
| medium | 22,219 | 14.155% | 12.874% | -1.280 pts |
| high | 16,765 | 26.597% | 24.895% | -1.702 pts |

**Model A — benchmark**:

| risk_band | n | observed adverse-outcome rate | mean predicted probability | calibration gap |
|---|---|---|---|---|
| low | 30,838 | 7.296% | 6.415% | -0.881 pts |
| medium | 52,181 | 12.644% | 12.955% | +0.310 pts |
| high | 51,785 | 23.513% | 25.327% | +1.814 pts |

**Model B — validation**:

| risk_band | n | observed adverse-outcome rate | mean predicted probability | calibration gap |
|---|---|---|---|---|
| low | 14,892 | 7.340% | 6.126% | -1.213 pts |
| medium | 19,205 | 13.559% | 13.080% | -0.479 pts |
| high | 19,270 | 25.672% | 26.255% | +0.583 pts |

**Model B — benchmark**:

| risk_band | n | observed adverse-outcome rate | mean predicted probability | calibration gap |
|---|---|---|---|---|
| low | 32,498 | 6.619% | 5.893% | -0.726 pts |
| medium | 43,997 | 11.921% | 13.102% | +1.180 pts |
| high | 58,309 | 23.372% | 27.223% | +3.851 pts |

Read honestly: both models' bands rank-order the observed adverse-outcome
rate correctly (low < medium < high) on both splits — the bands are
doing their basic job. Calibration gaps are small on validation (all
under 2 points) but widen on benchmark's `high` band (Model A +1.8
points, Model B +3.9 points) — the `high` band is where both models
over-predict risk most on the out-of-time split, consistent with
Section A's slope/intercept finding.

## C. Illustrative LGD × EAD expected-loss framework

`src/credit_risk/expected_loss.py`: `Expected Loss = risk_probability ×
LGD × EAD`. **LGD** (loss-given-default) is a single configurable
scalar, default **0.8**, stored in
`configs/credit_risk_expected_loss_assumptions.json` — it is an
author-chosen illustrative assumption, **not** derived from the real
`recoveries`/`total_rec_prncp` fields, and is not a CECL/Basel/
regulatory LGD estimate. **EAD** (exposure at default) is proxied by
`loan_amnt` (`EAD_FIELD` constant) — the original funded amount at
origination, chosen because these are closed-end installment loans with
no revolving redraw, and `loan_amnt` is consistent with Milestone 3's
exclusion of `funded_amnt`/`funded_amnt_inv` as redundant/collinear
fields. See the companion doc,
`docs/credit_risk_expected_loss_framework.md`, for the full rationale.
Every expected-loss number in this milestone carries this framing; none
is an accounting, CECL, Basel, or regulatory estimate.

## D. Model A vs. Model B expected-loss comparison (benchmark, same LGD/EAD assumptions for both)

| | Model A | Model B |
|---|---|---|
| Average expected loss | $2,065.52 | $2,147.31 |
| EL — low band | $671.21 | $680.42 |
| EL — medium band | $1,360.95 | $1,415.45 |
| EL — high band | $3,605.80 | $3,517.09 |

Model B's average EL is $81.78 higher than Model A's on benchmark — a
small difference relative to either figure. **Ranking consistency**
(Spearman correlation between Model A's and Model B's per-loan benchmark
expected-loss arrays): **ρ = 0.9667** (p ≈ 0.0, n = 134,804) — the two
models agree closely on which loans are economically riskier than which
others, even though Model B uses three additional lender-derived fields.
This is read as evidence about **economic risk-ranking agreement**, not
as a deployment comparison: Model B is not recommended for deployment
over Model A (see Section J and
`docs/credit_risk_model_feature_sets.md`). Interestingly, Model B's
`high`-band EL is slightly *lower* than Model A's ($3,517 vs. $3,606)
despite Model B's higher overall average — a reminder that per-band
aggregates and the overall average can move in different directions
once band membership itself differs slightly between the two models'
scores (Model B's `high` band has more rows, 58,309 vs. 51,785, than
Model A's at the same split).

## E. Illustrative three-way decisioning policy

`src/credit_risk/decisioning.py::classify_decisions` labels each loan
`approve_like` / `manual_review` / `decline_like` using neutral,
deliberately non-operational labels (never "approve"/"decline") to avoid
overstating underwriting authority. Thresholds are **Section B's
risk-band boundaries**, reused directly rather than independently
refit: `review_threshold = low/medium boundary`, `decline_threshold =
medium/high boundary` — both fit on train only. This is an illustrative
policy only, not a real approve/decline system.

## F. Illustrative decision-cost assumptions and base policy (Model A, benchmark)

`src/credit_risk/decision_cost_model.py` distinguishes three numbers
this milestone is careful never to conflate: the **model probability**
(a model output), **expected loss** (`p × LGD × EAD`, forward-looking,
probability-weighted — Section C/D), and the **business decision cost**
computed here (backward-looking, realized-outcome: a loan that actually
had the adverse outcome and was not declined contributes its realized
`LGD × EAD` dollar loss, not a probability-weighted figure).

Cost assumptions (`configs/credit_risk_decisioning_cost_assumptions.json`,
illustrative/synthetic, not real LendingClub review-operations costs):

| assumption | value |
|---|---|
| cost_per_manual_review | $50.00 |
| opportunity_cost_per_good_loan_declined | $150.00 |

Base policy result (Model A's boundaries as thresholds, applied to the
134,804-row benchmark split, `LGD = 0.8`):

| metric | value |
|---|---|
| review_threshold / decline_threshold | 0.09543 / 0.16732 |
| approve_volume / review_volume / decline_volume | 30,838 / 52,181 / 51,785 |
| adverse_outcome_capture | 57.91% |
| good_loan_rejection_rate | 34.81% |
| realized_adverse_outcome_loss | $91,282,500 |
| review_cost | $2,609,050 |
| opportunity_cost | $5,941,350 |
| **estimated_illustrative_total_cost** | **$99,832,900** |

`adverse_outcome_capture` counts only `decline_like` loans as "captured"
— `manual_review` is not assumed to catch or mitigate an adverse outcome
at all, mirroring `src/models/cost_model.py`'s treatment of its decline
action in Module B.

## G. Threshold / assumption sensitivity analysis

`sensitivity_analysis` varies LGD, the decline threshold, and both flat
costs one at a time off the Section F base case. **No row below is
called "optimal"** — this table only reports how each metric moves as
assumptions move; picking a policy requires picking a loss function this
milestone does not adjudicate.

| variation | LGD | decline_threshold | capture | good-loan rejection | review_vol | decline_vol | total cost |
|---|---|---|---|---|---|---|---|
| base | 0.80 | 0.16732 | 57.91% | 34.81% | 52,181 | 51,785 | $99,832,900 |
| lgd_low | 0.60 | 0.16732 | 57.91% | 34.81% | 52,181 | 51,785 | $77,012,275 |
| lgd_high | 0.95 | 0.16732 | 57.91% | 34.81% | 52,181 | 51,785 | $116,948,369 |
| stricter_decline | 0.80 | 0.11732 | 80.53% | 61.42% | 17,153 | 86,813 | $53,625,840 |
| looser_decline | 0.80 | 0.21732 | 39.26% | 19.64% | 73,366 | 30,600 | $144,571,340 |
| costlier_review | 0.80 | 0.16732 | 57.91% | 34.81% | 52,181 | 51,785 | $105,051,000 |
| costlier_opportunity | 0.80 | 0.16732 | 57.91% | 34.81% | 52,181 | 51,785 | $111,715,600 |

Read honestly: LGD moves total cost roughly linearly through the
realized-loss term, as expected (`lgd_low`/`lgd_high` leave every volume
metric unchanged, only the dollar figure). The decline threshold has by
far the largest effect on the *shape* of the tradeoff: `stricter_decline`
roughly triples adverse-outcome capture's cost-efficiency while also
tripling the good-loan rejection rate (more good loans get declined
along with more adverse ones) — and, in this run, produces the *lowest*
total cost of all seven rows, purely because this toy cost model does
not charge the opportunity cost (bigger here, $10.5M vs. base's $5.9M)
nearly as much as the realized-loss term it avoids (down to $42.3M from
$91.3M). That is an artifact of the chosen $50/$150 flat costs relative
to LGD × EAD loan sizes in the thousands of dollars, **not** a claim
that a stricter policy is actually better — a different cost ratio would
move this conclusion, which is exactly Section G's point.

## H. Selection-bias constraint (restated, machine-readable)

This dataset only observes loans LendingClub actually originated.
Rejected applicants have no repayment outcome in this data, so nothing
in Sections B-G can validate what would happen to the *full applicant
population* under a changed approval policy — only to the already-
approved population under a changed post-hoc decision rule applied
retrospectively. Reject inference is explicitly out of scope. This
disclaimer is embedded machine-readably as a top-level
`selection_bias_disclaimer` key in
`data/interim/credit_risk_milestone4_metrics.json`, not only in this
prose.

## I. Governance diagnostics — vintage stability and drift (Model A, descriptive)

**Population Stability Index** (`compute_psi`, train-fit quantile bins,
10 bins): train vs. validation and train vs. benchmark, for both models:

| | PSI (train vs. validation) | category | PSI (train vs. benchmark) | category |
|---|---|---|---|---|
| Model A | 0.0508 | no_significant_change | 0.0644 | no_significant_change |
| Model B | 0.0171 | no_significant_change | 0.0652 | no_significant_change |

Both models land in the heuristic `no_significant_change` band
(`PSI < 0.10`) for both comparisons — per `PSI_HEURISTIC_CAVEAT`, these
0.10/0.25 cut points are an industry rule of thumb, not a calibrated
hypothesis test on this dataset, and a value landing on either side of
them is a prompt to look closer, not a formal significance finding.
Note the tension with Section A: the **score distribution itself**
(what PSI measures) is stable train→validation→benchmark, even though
**calibration quality** (what the Cox slope/intercept measures) degrades
substantially benchmark. These are genuinely different properties — a
model can keep producing a similarly-shaped distribution of scores while
those scores drift away from being accurate probabilities.

**Score distribution by vintage** (Model A, all labeled rows 2007-2016,
applied never-refit — 2014-2016 rows are the already-labeled,
non-evaluation-eligible subset of `excluded_too_recent`, reported
descriptively only per Milestone 3's precedent):

| year | n | mean | p10 | p50 | p90 |
|---|---|---|---|---|---|
| 2007 | 603 | 0.2636 | 0.0552 | 0.2247 | 0.5280 |
| 2008 | 2,393 | 0.1801 | 0.0647 | 0.1525 | 0.3210 |
| 2009 | 5,281 | 0.1254 | 0.0473 | 0.1107 | 0.2178 |
| 2010 | 12,537 | 0.1499 | 0.0520 | 0.1265 | 0.2814 |
| 2011 | 21,721 | 0.1518 | 0.0530 | 0.1278 | 0.2863 |
| 2012 | 53,367 | 0.1501 | 0.0641 | 0.1312 | 0.2674 |
| 2013 | 134,804 | 0.1621 | 0.0666 | 0.1428 | 0.2904 |
| 2014 | 180,416 | 0.1377 | 0.0526 | 0.1264 | 0.2386 |
| 2015 | 283,026 | 0.1196 | 0.0454 | 0.1142 | 0.1994 |
| 2016 | 109,235 | 0.1178 | 0.0412 | 0.1117 | 0.1995 |

Read honestly: the predicted-probability distribution visibly narrows
and its mean/p90 decline in the later, non-evaluation-eligible years
(p90 falls from 0.528 in 2007 to 0.1995 in 2016) — fewer loans are being
scored as high-risk in later vintages by a model that was never refit on
them. This is consistent with, but does not by itself explain, the
calibration-slope degradation in Section A.

**Calibration by vintage** (Model A, same rows):

| year | n | observed rate | mean predicted | calibration gap |
|---|---|---|---|---|
| 2007 | 603 | 26.202% | 26.361% | +0.159 pts |
| 2008 | 2,393 | 20.727% | 18.010% | -2.717 pts |
| 2009 | 5,281 | 13.691% | 12.537% | -1.154 pts |
| 2010 | 12,537 | 14.015% | 14.993% | +0.978 pts |
| 2011 | 21,721 | 15.179% | 15.181% | +0.003 pts |
| 2012 | 53,367 | 16.197% | 15.009% | -1.189 pts |
| 2013 | 134,804 | 15.596% | 16.212% | +0.616 pts |
| 2014 | 180,416 | 15.087% | 13.770% | -1.317 pts |
| 2015 | 283,026 | 14.886% | 11.959% | -2.928 pts |
| 2016 | 109,235 | **17.280%** | 11.776% | **-5.505 pts** |

2016 shows the largest calibration gap in the table (-5.5 points) —
the model under-predicts risk most severely in the most recent,
descriptive-only vintage, even as its predicted-probability distribution
(previous table) has narrowed and shifted down. Per Milestone 3's own
precedent, 2014-2016 are reported **only** as a descriptive cross-check;
they are never used to retrain, re-threshold, or report a headline
metric, and this apparent drift is flagged, not investigated further,
here.

**Adverse-outcome rate by risk band over time**: computed for all
year/band cells that actually occur (30 cells across 2007-2016 for Model
A; absent cells are omitted, never fabricated as zero — full table in
the JSON artifact's `models.<name>.adverse_outcome_rate_by_band_over_time`).
Spot-checking the endpoints: 2007's `high` band (n=380) observed 36.58%
adverse outcomes vs. 2016's `high` band (n=20,860) at 27.05% — consistent
with the overall downward drift already visible in Sections A and I's
other tables, though 2007's n is small enough that it should not be
over-read as a precise estimate.

## J. Model A vs. Model B — governance discussion (why stronger performance ≠ better governance)

Milestone 3 already found Model B's `grade`/`sub_grade`/`int_rate`
lift over Model A to be real but modest (ROC-AUC +0.0027 validation /
+0.0146 benchmark; PR-AUC +0.0001 / +0.0095). This milestone's new
figures tell a consistent story from a different angle: Model A and
Model B's benchmark expected-loss rankings agree almost completely
(Spearman ρ = 0.9667, Section D) — the lender's own grade/pricing signal
changes *who looks riskier than whom* only at the margins, even though
it measurably improves raw discrimination. Three governance
implications follow:

1. **Circularity risk.** `grade`/`sub_grade`/`int_rate` are themselves
   LendingClub's own prior underwriting and pricing decision, produced
   using information likely including some of Model A's own 30 fields
   plus data this project does not have (e.g. full underwriting file,
   possibly a credit-bureau pull beyond FICO). Model B's lift is
   therefore evidence of **signal already embedded in a historical
   human/algorithmic decision**, not newly-discovered independent
   predictive information. Section G of Milestone 3 documented this
   explicitly (`AblationResult.framing_note`); this milestone's near-1.0
   rank correlation is additional, consistent evidence for the same
   point.
2. **Interpretability.** Model A's 30 application-time-safe fields are
   all things the borrower reported or that are independently
   verifiable at application time, with a defensible odds-ratio story
   (Milestone 3, Section G: `purpose_small_business`, several
   `addr_state` dummies). Model B adds three fields whose own internal
   construction is opaque to this project — `grade`/`sub_grade` are
   LendingClub's proprietary output, not raw application data — and
   Milestone 3 already found several of Model B's grade/sub-grade
   coefficients are non-monotonic once Model A's other features are in
   the model (e.g. `grade_B` > `grade_C` ≈ `grade_D`), the expected
   signature of collinearity with a derived field, not a causal credit
   story.
3. **Dependency on LendingClub's own historical system.** A model that
   leans on `grade`/`sub_grade`/`int_rate` inherits whatever biases,
   errors, or policy changes existed in LendingClub's own historical
   underwriting/pricing process over 2007-2011 (train). Model A, built
   only from independently-interpretable application-time fields, does
   not carry that dependency. Combined with (1) and (2), **Model B's
   higher raw ROC-AUC is not evidence of better governance** — it is
   evidence of a small amount of additional information leaking in
   through a feature whose own provenance this project cannot audit.
   This is why Model A, not Model B, is the one whose risk bands/
   decisioning policy are used as this milestone's running illustrative
   example in Sections E-G, and why neither model is recommended for
   deployment.

## K. New reusable modules + tests

Six new modules under `src/credit_risk/`: `expected_loss.py`,
`risk_bands.py`, `decisioning.py`, `decision_cost_model.py`,
`score_stability.py`, plus the orchestration script
`run_milestone4_decisioning.py`; one existing module extended additively
(`calibration.py`, new `calibration_slope_intercept` function, Milestone
3's functions untouched). Two new config files
(`configs/credit_risk_expected_loss_assumptions.json`,
`configs/credit_risk_decisioning_cost_assumptions.json`), each carrying
an explicit `_disclaimer` key.

Six new/extended test files under `tests/credit_risk/`
(`test_expected_loss.py`, `test_risk_bands.py`, `test_decisioning.py`,
`test_decision_cost_model.py`, `test_score_stability.py`, and six new
cases appended to `test_calibration.py`) cover: hand-computed expected-
loss arithmetic and its linear scaling in LGD/EAD; LGD validated to
`[0,1]`, negative EAD and mismatched shapes raise; risk-band boundaries
are fit on train-only quantiles and reused, never refit, on
validation/benchmark, with degenerate/tied-boundary inputs raising and
zero-row bands reported with explicit `None` fields rather than
fabricated zeros; decision thresholds must satisfy
`0 <= review_threshold < decline_threshold <= 1`, else raise, with
boundary-inclusivity checked explicitly; the decision-cost model's three
components (realized `lgd*ead` loss on undeclined-but-adverse loans,
flat review cost, flat opportunity cost) are independently hand-verified
on a 4-loan toy example, and sensitivity analysis never labels any row
"optimal"; PSI rejects empty inputs, too few bins, and a degenerate
all-tied reference distribution, and correctly separates a
near-identical from a fully-shifted distribution against the heuristic
bands; vintage-grouped score distribution/calibration/adverse-outcome-
by-band functions reject mismatched input shapes and correctly omit
(never zero-fill) absent year/band cells; the new Cox calibration-
regression function recovers near-1/0 slope/intercept on a synthetic
well-calibrated sample, correctly flags an overconfident synthetic
sample (slope < 1), and does not raise on near-degenerate/extreme
probability inputs (only warns, per `statsmodels`). Also pinned: both
new config files load successfully and their `_disclaimer` keys are
non-empty.

Full repository suite: **371 passed, 1 skipped**
(`.venv/bin/python -m pytest -q`), confirming **zero regressions**
against the pre-Milestone-4 baseline (318 passed).

## L. Documentation

This document (Sections A-M),
`docs/credit_risk_expected_loss_framework.md` (Section C/D LGD/EAD
rationale in focused detail), `README.md`'s Module C status section,
and `docs/decision_log.md`'s Milestone 4 entry.

## M. Consolidated summary

Both Model A and Model B's **raw, uncalibrated** predicted probabilities
were kept as the probability representation for this milestone, since
neither sigmoid nor isotonic recalibration improves on them on
validation (Section A) — the single biggest honest finding being that
this model's **calibration degrades substantially out-of-time**
(Cox slope 0.95→0.42 for Model A, 0.85→0.50 for Model B, validation→
benchmark) even though its Brier score and PSI-measured score
distribution both look stable, a distinction worth remembering before
trusting any probability from this model on a future vintage. Transparent,
train-only-fit tertile risk bands (Section B) rank-order observed
adverse-outcome rate correctly on both evaluation splits, with the
widest calibration gap appearing in the `high` band on benchmark (+1.8
to +3.9 points). An illustrative LGD=0.8 × EAD=`loan_amnt` expected-loss
framework (Section C) puts Model A's average benchmark EL at $2,065.52
and Model B's at $2,147.31 (Section D) — a small difference, and the two
models' per-loan EL rankings agree closely (Spearman ρ=0.9667),
supporting Milestone 3's existing finding that Model B's lender-grade
lift is real but modest. An illustrative three-way decisioning policy
(approve-like/manual-review/decline-like, Section E) built directly on
Model A's risk-band boundaries produces a base-case adverse-outcome
capture of 57.91% at a 34.81% good-loan rejection rate and an estimated
illustrative total cost of $99.8M on the 134,804-row benchmark split
under $50/review and $150/opportunity-cost assumptions (Section F); a
7-row sensitivity table (Section G) shows total cost and the
capture/rejection tradeoff both move substantially with LGD and the
decline threshold, and **no row is called optimal**. The dataset's
selection-bias constraint — rejected applicants have no outcome, so no
section here validates a changed approval policy against the full
applicant population — is restated in prose and embedded machine-
readably in the JSON artifact (Section H). Vintage/governance
diagnostics (Section I) show a stable PSI-measured score distribution
alongside a real, descriptive calibration drift that peaks at -5.5
points in the 2016 non-evaluation-eligible vintage. Section J argues
that Model B's higher raw discrimination is not evidence of better
governance, given its near-total expected-loss rank agreement with
Model A, its reliance on an opaque, LendingClub-internal
`grade`/`sub_grade` signal, and that signal's own non-monotonic,
collinearity-flavored coefficients already documented in Milestone 3.
Six new, independently tested library modules (**371 passed, 1 skipped**
full suite, zero regressions) extend `src/credit_risk/` with this
milestone's capabilities. No model is recommended for deployment, no
approve/decline decision is made for any real applicant, no rejected-
applicant outcome is inferred, and no 12-month PD claim is made anywhere
in this milestone. **This milestone's deliverable stops here for
review — no further Module C milestone is started.**
