# Milestone 4 — Probability Calibration & Illustrative Fraud Decisioning

Module B (Fraud Detection), Milestone 4. Builds on Milestone 3's
chronological train/validation/holdout split (byte-for-byte identical
boundaries — `src/models/split.py::chronological_split` with its default
fractions) and Milestone 3's class-weighted logistic-regression baseline.
**No boosted-tree, random-forest, or neural-network model is trained
here**, per explicit scope. Holdout is touched exactly twice: once to
*evaluate* (never fit) calibration quality, and once to *evaluate* (never
re-optimize) the single validation-selected decision policy.

All code: `src/models/calibration.py` (sigmoid/isotonic calibrators,
reliability table), `src/models/cost_model.py` (illustrative cost
framework, three-action decisioning, threshold grid, sensitivity
analysis), `src/models/train_calibrated.py` (orchestrator), plus small
backward-compatible extensions to `src/models/schema.py`,
`src/models/preprocessing.py`, and `src/models/train_baseline.py` to
support the time-feature ablation and to let calibrated-probability
arrays reuse Milestone 3's metric logic
(`train_baseline.py::evaluate_probs`). Run with
`.venv/bin/python -m src.models.train_calibrated`; writes
`data/interim/milestone4_metrics.json` (raw numbers this document is
built from) in addition to printing the same report to stdout.

## 1. What this milestone solves

Milestone 3 found that `class_weight="balanced"` produces a
well-*ordered* score (ROC-AUC 0.765 validation / 0.732 holdout) that is
badly *miscalibrated* as a probability (Brier 0.260 validation / 0.316
holdout) — predicted probabilities ran several multiples above observed
fraud rates in every decile bin. Milestone 4 fixes the probability side
without touching the ranking side: it recalibrates the same frozen
Milestone 3 scores using two standard post-hoc methods fit on validation
only, then builds an explicit, labeled-as-hypothetical cost framework on
top of the resulting probabilities so that threshold selection can be
reasoned about in dollar terms instead of only ROC-AUC/precision/recall.

## 2. Why hand-rolled calibrators instead of `CalibratedClassifierCV`

`sklearn.calibration.CalibratedClassifierCV(cv="prefit")` is the
textbook tool for this, but `cv="prefit"` support has shifted across
recent scikit-learn releases and this project pins
`scikit-learn>=1.4,<2`, a range where relying on that exact code path is
fragile to a future upgrade. Instead, `src/models/calibration.py` defines
two minimal, directly testable calibrator classes:

- **`SigmoidCalibrator`** (Platt scaling): a 1-D
  `LogisticRegression` of `y_true` on the raw score, i.e. exactly what
  Platt scaling is — a strictly monotonic transform of the raw score, so
  it cannot change the rank ordering of any two transactions, and
  therefore cannot change ROC-AUC/PR-AUC at all (floating-point noise
  aside).
- **`IsotonicCalibrator`**: wraps
  `sklearn.isotonic.IsotonicRegression(out_of_bounds="clip")`, a
  non-decreasing step function fit on the raw score. Non-decreasing, not
  strictly increasing, so it *can* introduce ties (flat regions) that
  nudge ROC-AUC by a small amount relative to the raw score — expected,
  not a bug (see §3).

Both expose `.fit(raw_scores, y_true)` / `.transform(raw_scores)`, fit
**only** on validation raw probabilities and validation labels via
`fit_calibrator(method, raw_probs_val, y_val)`, and are evaluated by
applying the already-fitted `.transform` to holdout — holdout is never
passed to `.fit`. This is verified by mutating the holdout array *after*
fitting and confirming the calibrator's output on a fixed input doesn't
change (`tests/models/test_calibration.py::test_calibrator_fit_on_validation_does_not_change_when_holdout_is_mutated`).

## 3. Calibration comparison (objective 1–3, requirement A/B)

Both calibrators were fit on validation raw probabilities + validation
labels only, then applied to validation (the fit split) and holdout
(never fit on) for evaluation.

| Method | Split | ROC-AUC | PR-AUC | Brier score |
|---|---|---|---|---|
| Uncalibrated | Validation | 0.7652 | 0.1428 | 0.2599 |
| Uncalibrated | Holdout | 0.7317 | 0.1202 | 0.3157 |
| Sigmoid (Platt) | Validation | 0.7652 | 0.1428 | **0.0359** |
| Sigmoid (Platt) | Holdout | 0.7317 | 0.1202 | **0.0324** |
| Isotonic | Validation | 0.7674 | 0.1389 | **0.0356** |
| Isotonic | Holdout | 0.7312 | 0.1105 | **0.0337** |

**Ranking vs. calibration, confirmed exactly as expected (requirement
B):** sigmoid's ROC-AUC matches the uncalibrated score to the tested
tolerance (`abs=1e-9` — it is a strictly monotonic transform, so it
*must* match). Isotonic's ROC-AUC moves by at most ~0.005 in either
direction (0.7652→0.7674 validation, 0.7317→0.7312 holdout) — the
expected, small effect of tie-induced flat regions, not a real ranking
improvement or regression. **Brier score, in contrast, drops by roughly
7–9x on both splits for both methods.** This is the headline finding of
this milestone: calibration overwhelmingly improves *probability
quality*, while *ranking quality* (ROC-AUC/PR-AUC) is essentially
unchanged, exactly the separation Milestone 4 set out to demonstrate.

**Isotonic was selected** as the best method
(`select_best_calibration_method`, lowest validation Brier: 0.0356 vs.
0.0359 for sigmoid) and is the method used for all decisioning/cost work
in §5–§7 below. The margin between the two methods is small; sigmoid
would have been a reasonable first choice too, and is a strictly simpler,
more conservative transform (no ties, no flat regions).

## 4. Holdout calibration performance — reliability tables

Reliability tables (`src/models/calibration.py::reliability_table`,
quantile-binned) make the Brier-score improvement concrete.
**Uncalibrated, validation**, predicted probability vs. observed fraud
rate by decile — predicted is consistently 10–20x the observed rate:

| Predicted prob. (mean) | Observed fraud rate | n rows |
|---|---|---|
| 0.196 | 0.0111 | 11,879 |
| 0.353 | 0.0156 | 11,879 |
| 0.497 | 0.0273 | 11,879 |
| 0.778 | 0.0841 | 11,879 |
| 0.928 | 0.1512 | 11,879 |

**Isotonic-calibrated, validation** (fit split) — predicted and observed
match almost exactly by construction (isotonic regression minimizes
exactly this gap on its fit data):

| Predicted prob. (mean) | Observed fraud rate | n rows |
|---|---|---|
| 0.0106 | 0.0106 | 17,057 |
| 0.0233 | 0.0233 | 15,690 |
| 0.0534 | 0.0534 | 15,329 |
| 0.0908 | 0.0908 | 8,433 |
| 0.1570 | 0.1570 | 10,937 |

**Isotonic-calibrated, holdout** (never fit on) — predicted and observed
no longer match exactly (as expected — the calibrator's step function
was fit on validation's distribution, not holdout's), but both track the
same broad shape and stay within the same order of magnitude at every
bin, in sharp contrast to the uncalibrated 10–20x gap above:

| Predicted prob. (mean) | Observed fraud rate | n rows |
|---|---|---|
| 0.0104 | 0.0135 | 11,787 |
| 0.0257 | 0.0171 | 11,353 |
| 0.0370 | 0.0220 | 10,159 |
| 0.0647 | 0.0437 | 10,957 |
| 0.1044 | 0.0680 | 12,129 |
| 0.2627 | 0.1254 | 10,512 |

The holdout top bin (predicted 0.263 vs. observed 0.125) is the largest
remaining gap — calibration transfers well but not perfectly to a later
time period, consistent with the same non-stationarity Milestone 3 flagged
for ranking metrics.

## 5. Illustrative fraud-decision cost framework (requirement C)

**These cost figures are HYPOTHETICAL, illustrative portfolio
assumptions chosen by the author for this learning project.** They do
**not** represent Vesta Corporation, the IEEE-CIS competition, any real
bank, any real payments company, or any employer, and no total-cost
number in this document or in `data/interim/milestone4_metrics.json`
should be read as a real financial saving or loss estimate. They exist
only so the decision-cost mechanics below have concrete numbers to
compute with. See `configs/cost_assumptions.json` (which carries the
same disclaimer inline) and `src/models/cost_model.py`.

| Cost input | Value | Meaning |
|---|---|---|
| `cost_per_missed_fraud` | $500 | Illustrative cost of a false negative — fraud approved |
| `cost_per_false_positive_decline` | $25 | Illustrative cost of a false positive — legitimate transaction declined outright |
| `cost_per_manual_review` | $3 | Illustrative cost of routing a transaction to manual review |

All three are config-driven (`load_cost_assumptions`), not hardcoded in
model logic — changing them requires editing one JSON file, not source
code.

**Three-action decisioning** (`classify_actions`) requires two ordered
thresholds, not one: `score < review_threshold` → approve;
`review_threshold <= score < decline_threshold` → review;
`score >= decline_threshold` → decline. Invalid orderings
(`review_threshold >= decline_threshold`, or either outside `[0, 1]`) are
rejected with a `ThresholdError` rather than silently producing a
degenerate two-action policy.

A modeling simplification, stated explicitly rather than hidden: **a
transaction routed to manual review is treated as a perfect (if costly)
backstop against fraud** — `fraud_recall` counts review the same as
decline. This project has no real review-team accuracy data to build a
more realistic partial-catch-rate assumption; see §9 (Limitations).

## 6. Decision actions — threshold/cost tradeoff (requirement D)

Computed with `evaluate_policy_grid` on **validation only**, using
isotonic-calibrated probabilities, over `REVIEW_THRESHOLD_GRID = [0.05,
0.10, 0.15, 0.20, 0.30, 0.40, 0.50]` ×
`DECLINE_THRESHOLD_GRID = [0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]`
(restricted to `review < decline`), sorted ascending by illustrative
total cost. One representative (lowest-cost) row per `review_threshold`
value is shown below; the full 44-row grid is in
`data/interim/milestone4_metrics.json::threshold_cost_grid_validation`.

| Review thr. | Decline thr. | Fraud recall | Precision | False pos. | False neg. | Review vol. | Decline vol. | % affected | Illustrative total cost |
|---|---|---|---|---|---|---|---|---|---|
| 0.05 | 0.50 | 0.640 | 0.112 | 0 | 1,686 | 26,819 | 0 | 22.58% | $923,457 |
| 0.10 | 0.80 | 0.367 | 0.157 | 0 | 2,965 | 10,937 | 0 | 9.21% | $1,515,311 |
| 0.15 | 0.50 | 0.186 | 0.220 | 0 | 3,809 | 3,966 | 0 | 3.34% | $1,916,398 |
| 0.20 | 0.70 | 0.130 | 0.250 | 0 | 4,075 | 2,427 | 0 | 2.04% | $2,044,781 |
| 0.30 | 0.50 | 0.0094 | 0.376 | 0 | 4,638 | 117 | 0 | 0.10% | $2,319,351 |
| 0.40 | 0.50 | 0.0030 | 0.424 | 0 | 4,668 | 33 | 0 | 0.03% | $2,334,099 |
| 0.50 | 0.60 | 0.000 | n/a | 0 | 4,682 | 0 | 0 | 0.00% | $2,341,000 |

**Lowest-cost row under base assumptions: `review_threshold=0.05,
decline_threshold=0.5`** — recall 0.640, 22.58% of validation
transactions flagged (all to review, zero outright declines), cost
$923,457. Under these specific cost inputs, manual review ($3) is so much
cheaper than a declined legitimate transaction ($25) that the grid never
prefers decline over review at any threshold pair tested — `decline_volume`
is 0 in every lowest-cost-ordered row shown above except where review
itself becomes nearly unused. **This is not a recommended production
threshold** — it is one point on an exploration grid under one specific,
hypothetical cost configuration (requirement D: "explore thresholds
rather than hardcoding a final production policy").

This validation-selected policy was evaluated once, and only once, on
holdout (never re-optimized there): review_threshold=0.05,
decline_threshold=0.5 → holdout fraud_recall=0.649, precision=0.081,
review_volume=31,707 (27.44% of holdout), 0 outright declines,
illustrative holdout cost $787,621. Recall transferred well
(0.640→0.649); precision dropped slightly (0.112→0.081), consistent with
holdout's lower fraud prevalence (3.94%→3.41%) diluting the same alert
volume.

## 7. Sensitivity analysis (requirement E)

`build_cost_variations` perturbs one cost input at a time;
`sensitivity_analysis` re-runs the full validation threshold grid under
each variation and reports the new lowest-cost policy. The initial
design (simple ±2x on each of the three inputs) turned out to be too
mild for this dataset and is itself a reportable finding, not a
negative result:

| Variation | `cost_per_missed_fraud` | `cost_per_false_positive_decline` | `cost_per_manual_review` | Selected (review, decline) | Fraud recall | Decline vol. |
|---|---|---|---|---|---|---|
| Base | $500 | $25 | $3 | (0.05, 0.50) | 0.640 | 0 |
| `fraud_loss_2x` | $1,000 | $25 | $3 | (0.05, 0.50) | 0.640 | 0 |
| `fraud_loss_half` | $250 | $25 | $3 | (0.05, 0.50) | 0.640 | 0 |
| `false_positive_cost_2x` | $500 | $50 | $3 | (0.05, 0.50) | 0.640 | 0 |
| `false_positive_cost_half` | $500 | $12.50 | $3 | (0.05, 0.50) | 0.640 | 0 |
| `review_cost_2x` | $500 | $25 | $6 | (0.05, 0.50) | 0.640 | 0 |
| `review_cost_half` | $500 | $25 | $1.50 | (0.05, 0.50) | 0.640 | 0 |

**At base costs, manual review ($3) is so far below the decline cost
($25) that ±2x moves on any single input never cross the break-even
point where decline becomes preferable to review, or where acting at all
stops being worth it — the argmin-cost policy is identical across all six
variations, only the reported dollar total moves.** That is a genuine
property of this specific cost configuration (review is cheap relative
to both alternatives), not a bug in the sensitivity mechanism. Two more
extreme, targeted variations were added specifically to demonstrate
requirement E's point — that the *selected threshold region* does move
once a cost assumption crosses the relevant economic break-even point:

| Variation | `cost_per_missed_fraud` | `cost_per_false_positive_decline` | `cost_per_manual_review` | Selected (review, decline) | Fraud recall | Decline vol. | Illustrative cost |
|---|---|---|---|---|---|---|---|
| `review_cost_exceeds_decline_cost` | $500 | $25 | **$60** | **(0.10, 0.30)** | **0.367** | **117** | $2,133,525 |
| `fraud_loss_near_zero` | **$5** | $25 | $3 | **(0.50, 0.90)** | **0.000** | 0 | $23,410 |

- **Raising manual-review cost above the decline cost** ($3→$60, more
  than 2x the $25 decline cost) flips the policy: review is no longer
  unconditionally the cheaper way to act on a suspicious transaction, so
  the argmin-cost policy both becomes more conservative (review threshold
  rises 0.05→0.10) *and* starts using outright decline (`decline_volume`
  0→117) — fraud recall drops from 0.640 to 0.367 as a direct consequence.
- **Dropping missed-fraud cost to near-zero** ($500→$5) collapses the
  policy entirely to all-approve (review_threshold=0.50,
  decline_threshold=0.90, recall=0.000) — once a missed fraud is cheap
  enough, acting on any transaction costs more than it saves under these
  assumptions.

**This is the core requirement-E finding: the "best" threshold pair is
not a fixed property of the model or its ROC-AUC — it is a function of
assumed unit costs, and moving those costs far enough (not just ±2x, but
past the specific break-even points that happen to exist in this cost
configuration) changes both which actions get used and how much fraud
gets caught.**

## 8. Time-feature ablation (requirement F)

Compares the full feature set against `NUMERIC_FEATURES_NO_TIME_DRIFT`
(`day_index`/`week_index` removed — the two linear time-trend proxies
Milestone 3 flagged as carrying extrapolation risk), fit and evaluated on
identical train/validation/holdout data.

| Variant | Split | ROC-AUC | PR-AUC | Brier score |
|---|---|---|---|---|
| With `day_index`/`week_index` | Validation | 0.7652 | 0.1428 | 0.2599 |
| With `day_index`/`week_index` | Holdout | 0.7317 | 0.1202 | 0.3157 |
| Without `day_index`/`week_index` | Validation | 0.7636 | **0.1710** | 0.2301 |
| Without `day_index`/`week_index` | Holdout | **0.7351** | **0.1429** | 0.2552 |

**Removing the two time-drift features did not hurt holdout
performance — it slightly improved it on every metric reported here**:
holdout ROC-AUC rose (0.7317→0.7351) and holdout PR-AUC rose
substantially (0.1202→0.1429, a ~19% relative increase on the
minority-class-sensitive metric). Validation ROC-AUC dropped
negligibly (0.7652→0.7636) while validation PR-AUC *also* rose
(0.1428→0.1710).

**Why this is not surprising, and why the features are still a
deployment risk regardless of this result:** `day_index`/`week_index`
are monotonically increasing counters, not bounded behavioral signals.
Milestone 3's coefficient analysis found a positive linear coefficient on
both (fraud odds rose across the observed 182-day window), which is a
real pattern *within* that window but is, by construction, an
extrapolation assumption the moment any transaction is scored outside
it — there's no evidence the linear trend continues, accelerates, resets,
or reverses past day 182, and this project's data doesn't extend far
enough to check. This ablation result (removing them doesn't cost
performance — it marginally helps) weakens the case for keeping them at
all, on top of the extrapolation-risk argument Milestone 3 already made.
Because the margins here (holdout ROC-AUC +0.0034, PR-AUC +0.0227) are
small relative to a single train/validation/holdout split, this is
treated as suggestive, not conclusive — see §9.

## 9. Limitations

- **Manual review is modeled as a perfect fraud backstop** (§5) — no real
  accuracy rate for a review team exists in this project, so every
  reviewed transaction counts as "caught" for `fraud_recall` purposes.
  This likely overstates the real value of routing to review relative to
  outright decline, and any real deployment would need an empirical
  review-team catch rate to replace this simplifying assumption.
- **All cost figures are hypothetical and configurable, not measured**
  (§5, §7) — every dollar number in this document is illustrative of
  *mechanics*, not a claim about real fraud-loss, chargeback, or
  review-operations economics for any company.
- **The sensitivity analysis's two most informative variations
  (`review_cost_exceeds_decline_cost`, `fraud_loss_near_zero`) are
  deliberately extreme**, chosen specifically because the milder ±2x
  variations all landed on the same policy at base costs — this says the
  argmin-cost policy is locally flat near the current assumptions for
  small perturbations, and only moves once a variation crosses a specific
  break-even point. A different base cost configuration could easily have
  the opposite property (sensitive even to small moves); this result is
  specific to the $500/$25/$3 configuration chosen here.
- **The time-feature ablation (§8) is based on one train/validation/holdout
  split** — the holdout improvement from removing `day_index`/`week_index`
  (ROC-AUC +0.0034, PR-AUC +0.0227) is directionally consistent with the
  extrapolation-risk concern but is not large enough, from a single split,
  to rule out split-specific noise as a contributing factor.
- **Isotonic calibration's small ROC-AUC shift (±0.005) is expected and
  is not evidence of a ranking improvement** (§3) — it is a known
  artifact of isotonic regression's tie-inducing flat regions, not a
  reason to prefer isotonic over sigmoid on ranking grounds.
- **Calibration quality degrades somewhat from validation to holdout**
  (§4) — the isotonic calibrator's top-bin gap on holdout (predicted
  0.263 vs. observed 0.125) shows calibration, like ranking, is subject
  to the same non-stationarity Milestone 3 documented; it is a large
  improvement over uncalibrated, not a perfect fix.
- **No production threshold is chosen here** (§6) — the threshold/cost
  grid is exploration, not a recommendation, per explicit Milestone 4
  scope ("explore thresholds rather than hardcoding a final production
  policy").
- **Carried over, unresolved from Milestones 1–3**: `C*`/`D*`/`M*`/`V*`
  feature-group leakage status remains open; `card2`/`card3`/`card5`/
  `addr2` still carry an unverified ordinal/interval assumption.
- **Holdout has now been touched twice total across Milestones 3–4**
  (once per milestone's single evaluation pass) — any future milestone
  that revisits modeling choices and re-checks this same holdout slice
  would need to treat it as contaminated and use a fresh, never-touched
  slice instead.

## 10. Engineering & tests (requirement G)

`tests/models/test_calibration.py` (11 tests) — uncalibrated method is an
identity transform; sigmoid/isotonic outputs are valid probabilities in
`[0, 1]`; both materially reduce Brier score on their own fit
distribution; a calibrator's output on a fixed input is unaffected by
mutating a *different* array after fitting (no retroactive leakage);
unknown method names are rejected; the calibration-methods registry has
exactly the three expected keys; reliability-table bin counts sum to the
full input size; reliability-table rates are valid and bin edges are
ordered; both calibrator classes are directly usable outside the
`fit_calibrator`/`apply_calibrator` wrapper functions.

`tests/models/test_cost_model.py` (10 tests) — three-action classification
produces the correct bucket per threshold; inverted, equal, or
out-of-range thresholds raise `ThresholdError`; `evaluate_decision_policy`
matches an exact hand-computed 5-row example (recall, precision, false
positives/negatives, review/decline volume, total cost all independently
verified by hand); zero-fraud/all-approve inputs produce zero cost;
`threshold_pair_grid` only returns ordered `review < decline` pairs;
`evaluate_policy_grid` is sorted ascending by cost; raising fraud-loss
cost by 100x never *decreases* the chosen policy's fraud recall relative
to the base policy; `load_cost_assumptions` reads real positive values
from the committed config file.

`tests/models/test_schema.py` (2 new tests, 8 total) — the ablation
numeric-feature set is exactly `NUMERIC_FEATURES` minus
`{day_index, week_index}`, with matching set sizes.

`tests/models/test_train_calibrated.py` (6 tests, including a
real-data-gated smoke test) — `calibration_comparison` returns all three
methods on both splits with the expected metric/reliability-table keys;
mutating holdout raw probabilities after the fact leaves every
validation-side number byte-for-byte unchanged (the leakage guarantee
this milestone is built around); `select_best_calibration_method` reads
only the requested split's metric; `time_feature_ablation` returns the
expected structure with metrics in valid ranges; `build_cost_variations`
produces the expected 8 labeled variations preserving unvaried cost
fields; and, when `data/processed/baseline_features_v1.parquet` is
present, a full run against real data confirms sigmoid preserves
validation ROC-AUC exactly, isotonic preserves it within a small
tolerance, both calibrated methods beat uncalibrated Brier score, the
cost grid is non-empty and cost-sorted, and both ablation variants exceed
0.5 ROC-AUC on holdout.

**Full suite: 86 passed, 1 skipped** (same intentional
missing-raw-file error-path skip carried from Milestones 1–3).

## 11. Runtime

Fitting two additional full pipelines (the time-feature ablation's
with/without variants, on top of Milestone 3's baseline pipeline already
fit once for calibration), computing calibration curves, and evaluating
a 44-pair threshold grid × 8 sensitivity variations all complete in well
under a minute on the same machine used for Milestones 1–3. No new
dependencies were added — everything used (`sklearn.isotonic`,
`sklearn.linear_model.LogisticRegression`,
`sklearn.calibration.calibration_curve`, `dataclasses.replace`) was
already available via `scikit-learn`/stdlib.

## 12. Recommended next milestone

Two natural directions, not mutually exclusive:

1. **A non-linear model (gradient-boosted trees), explicitly deferred
   from Milestones 3–4**, evaluated on the *same* chronological split and
   held to the same calibration + cost-decisioning framework built here —
   this would let later milestones ask "does a more complex model change
   the ranking metrics, the calibration behavior, or the chosen cost-based
   threshold region" using apples-to-apples machinery.
2. **Resolve the `C*`/`D*`/`M*`/`V*` leakage question** carried unresolved
   since Milestone 1 — the single largest open risk in this project,
   independent of which model architecture is used on top of it.

Per this milestone's explicit instruction, work stops here for review;
no next milestone has been started.
