# Milestone 4 — Probability Calibration & Illustrative Fraud Decisioning

Module B (Fraud Detection), Milestone 4, **as corrected on 2026-10-02**.
Builds on Milestone 3's chronological train/validation/temporal-benchmark
split (byte-for-byte identical boundaries —
`src/models/split.py::chronological_split` with its default fractions)
and Milestone 3's class-weighted logistic-regression baseline. **No
boosted-tree, random-forest, or neural-network model is trained here**,
per explicit scope.

## 0. Methodology correction (read this first)

The first version of this milestone had two methodology problems, both
fixed here. **The numbers in this document are the corrected,
post-fix numbers** — see `docs/decision_log.md`'s 2026-10-02 entry for
the full before/after comparison.

**Correction 1 — in-sample calibration evaluation.** The original run fit
each calibrator on the validation period and then reported that
calibrator's Brier score / reliability table on the *same* validation
period. This is optimistically biased, and was especially severe for
isotonic regression (a non-parametric step function that can nearly
perfectly fit the distribution it was trained on) — the original
"validation" reliability table for isotonic had `mean_predicted_prob`
exactly equal to `observed_fraud_rate` in every bin, the signature of
evaluating a fit on its own training data. **Fix**: Milestone 3's
validation period is now further sub-split, chronologically, into:
- **`cal_fit`** (earlier half) — calibrators are `.fit()` here, and only
  here.
- **`selection`** (later half) — calibration quality (Brier/reliability)
  is evaluated here, and the cost-based threshold/cost grid + sensitivity
  analysis are also selected here. Both uses are legitimate because
  `selection` was never passed to any `.fit()` call.

The original train/validation/holdout day boundaries are **untouched** —
only the already-chronological validation period is internally
subdivided, via `src/models/split.py::chronological_split_two_way`.

**Correction 2 — unlimited, perfect manual review.** The original cost
model assumed manual review had unlimited capacity and caught 100% of
routed fraud with zero review-side errors. `src/models/cost_model.py` now
has three independently configurable parameters —
`review_capacity_frac`, `review_fraud_catch_rate`,
`review_false_positive_rate` — all defaulting to the old
unlimited/perfect behavior for backward compatibility, but now set to
illustrative non-default values (§5) so the realistic behavior is
actually exercised, not just theoretically available.

**Terminology correction.** The final chronological segment, called
"holdout" through Milestone 3 and the original Milestone 4 run, is
renamed **temporal benchmark** everywhere in this document, in
`README.md`, and in code/tests. **It has now been inspected multiple
times during iterative development** — Milestone 3's baseline metrics,
the original Milestone 4 calibration/policy evaluation, and this
correction's re-evaluation — and **must not be presented as a pristine,
never-touched final unbiased test set.** It remains useful as a fixed,
later-in-time check that choices made earlier (calibration method,
decision thresholds) still hold up at a later point in the calendar, but
not as an unbiased estimate of future production performance.

All code: `src/models/calibration.py` (sigmoid/isotonic calibrators,
reliability table), `src/models/cost_model.py` (illustrative cost
framework, three-action decisioning, threshold grid, sensitivity
analysis, review-capacity/catch-rate/error-rate model),
`src/models/split.py` (`chronological_split_two_way`),
`src/models/train_calibrated.py` (orchestrator). Run with
`.venv/bin/python -m src.models.train_calibrated`; writes
`data/interim/milestone4_metrics.json` (raw numbers this document is
built from) in addition to printing the same report to stdout.

## 1. Corrected chronological split structure

```
train (day 1–101)         val (day 102–141)             temporal_benchmark (day 142–182)
                     ┌─────────────┴─────────────┐
               cal_fit (day 102–121)     selection (day 122–141)
```

| Segment | Rows | Fraud count | Fraud rate | Role |
|---|---|---|---|---|
| `train` | 356,216 | 12,039 | 3.380% | Fits the Milestone 3 baseline pipeline only |
| `cal_fit` | 61,343 | 2,682 | 4.372% | Calibrators (sigmoid, isotonic) are `.fit()` here, and only here |
| `selection` | 57,447 | 2,000 | 3.481% | Calibration evaluation (Brier/reliability) + threshold/cost grid selection — never fit on |
| `temporal_benchmark` | 115,534 | 3,942 | 3.412% | Touched twice: once to evaluate calibration out-of-sample, once to evaluate (not re-optimize) the selection-chosen policy |

`train`/`val`/`temporal_benchmark` boundaries are byte-for-byte identical
to Milestone 3's split. `cal_fit`/`selection` are a 50/50 chronological
sub-split of `val` only (`CALIBRATION_FIT_FRAC_OF_VALIDATION = 0.5`),
snapped to whole calendar-day boundaries by
`chronological_split_two_way`, the same boundary-snapping rule the
original 3-way split uses.

## 2. Why hand-rolled calibrators instead of `CalibratedClassifierCV`

`sklearn.calibration.CalibratedClassifierCV(cv="prefit")` is the
textbook tool for this, but `cv="prefit"` support has shifted across
recent scikit-learn releases and this project pins
`scikit-learn>=1.4,<2`, a range where relying on that exact code path is
fragile to a future upgrade. Instead, `src/models/calibration.py` defines
two minimal, directly testable calibrator classes:

- **`SigmoidCalibrator`** (Platt scaling): a 1-D `LogisticRegression` of
  `y_true` on the raw score — a strictly monotonic transform, so it
  cannot change the rank ordering of any two transactions, and therefore
  cannot change ROC-AUC/PR-AUC at all (floating-point noise aside).
- **`IsotonicCalibrator`**: wraps
  `sklearn.isotonic.IsotonicRegression(out_of_bounds="clip")`, a
  non-decreasing step function. Non-decreasing, not strictly increasing,
  so it *can* introduce ties (flat regions) that nudge ROC-AUC by a small
  amount relative to the raw score — expected, not a bug.

Both expose `.fit(raw_scores, y_true)` / `.transform(raw_scores)`, fit
**only** on `cal_fit` raw probabilities and labels, and are evaluated by
applying the already-fitted `.transform` to `selection` and
`temporal_benchmark` — neither is ever passed to `.fit`. Verified by
mutating `temporal_benchmark`'s raw probabilities *after* fitting and
confirming `selection`-side numbers don't change
(`tests/models/test_train_calibrated.py::test_calibration_comparison_selection_results_unaffected_by_temporal_benchmark_mutation`).

## 3. Calibration comparison — out-of-sample only (corrected)

Both calibrators were fit on `cal_fit` only, then applied to `selection`
and `temporal_benchmark` — **neither split was used to fit either
calibrator**, unlike the original (uncorrected) run, which reported
validation-split numbers for a calibrator fit on that same validation
split.

| Method | Split | ROC-AUC | PR-AUC | Brier score |
|---|---|---|---|---|
| Uncalibrated | selection | 0.7454 | 0.1273 | 0.2609 |
| Uncalibrated | temporal_benchmark | 0.7317 | 0.1202 | 0.3157 |
| Sigmoid (Platt) | selection | 0.7454 | 0.1273 | **0.0325** |
| Sigmoid (Platt) | temporal_benchmark | 0.7317 | 0.1202 | **0.0330** |
| Isotonic | selection | 0.7455 | 0.1127 | 0.0328 |
| Isotonic | temporal_benchmark | 0.7297 | 0.0929 | 0.0352 |

**Ranking vs. calibration, confirmed exactly as expected:** sigmoid's
ROC-AUC matches the uncalibrated score to the tested tolerance
(`abs=1e-9`). Isotonic's ROC-AUC moves by at most ~0.002 on `selection`
but drops by ~0.002 and PR-AUC drops substantially (0.1273→0.1127 on
`selection`, 0.1202→0.0929 on `temporal_benchmark`) — a real, if modest,
ranking cost from isotonic's tie-inducing flat regions on *this*
out-of-sample data, not just noise.

**Brier score, both methods, drop by roughly 8x versus uncalibrated on
both splits** — calibration overwhelmingly improves probability quality
regardless of which method is used. **But once evaluated strictly
out-of-sample, isotonic no longer wins on Brier score**: sigmoid's
0.0325 beats isotonic's 0.0328 on `selection`, and sigmoid's 0.0330 beats
isotonic's 0.0352 on `temporal_benchmark` — the opposite of the original
(uncorrected) run's conclusion (isotonic 0.0356 vs. sigmoid 0.0359,
evaluated in-sample on validation).

### Selected method: **sigmoid**

`select_best_calibration_method` computes the relative Brier improvement
isotonic would need to show on `selection` to be preferred:
`(sigmoid_brier − isotonic_brier) / sigmoid_brier = (0.032470 −
0.032819) / 0.032470 ≈ −1.08%` — **negative**, meaning isotonic is
*worse*, not better, out-of-sample. Since this falls well short of the
+3% relative-improvement bar required to switch to isotonic, **sigmoid
is selected**, and is also the simpler, more conservative transform (no
ties, no flat regions, exact ROC-AUC preservation). This is the clearest
possible illustration of why Correction 1 mattered: the original
in-sample evaluation picked the *wrong* calibrator for this dataset.

## 4. Reliability tables (sigmoid, out-of-sample)

`selection` (never fit on):

| Predicted prob. (mean) | Observed fraud rate | n rows |
|---|---|---|
| 0.0081 | 0.0125 | 5,745 |
| 0.0112 | 0.0124 | 5,745 |
| 0.0136 | 0.0138 | 5,744 |
| 0.0161 | 0.0136 | 5,745 |
| 0.0191 | 0.0124 | 5,745 |
| 0.0231 | 0.0221 | 5,744 |
| 0.0294 | 0.0261 | 5,745 |
| 0.0433 | 0.0373 | 5,744 |
| 0.0928 | 0.0693 | 5,745 |
| 0.1731 | 0.1288 | 5,745 |

`temporal_benchmark` (never fit on; this is calibration's *second*
out-of-sample check, further forward in time than `selection`):

| Predicted prob. (mean) | Observed fraud rate | n rows |
|---|---|---|
| 0.0086 | 0.0135 | 11,554 |
| 0.0126 | 0.0148 | 11,553 |
| 0.0160 | 0.0126 | 11,553 |
| 0.0196 | 0.0132 | 11,554 |
| 0.0242 | 0.0173 | 11,553 |
| 0.0303 | 0.0177 | 11,553 |
| 0.0396 | 0.0217 | 11,554 |
| 0.0610 | 0.0430 | 11,553 |
| 0.1302 | 0.0679 | 11,553 |
| 0.1970 | 0.1195 | 11,554 |

Both tables track the same broad shape (predicted probability rises
monotonically with observed fraud rate), and the gap in the top bin
(predicted ~0.17–0.20 vs. observed ~0.12–0.13) is real but far smaller
than the 10–20x gap the uncalibrated score showed (§3) — consistent
with Milestone 3's finding that this data is non-stationary across time,
so calibration quality degrades somewhat but doesn't collapse the
farther forward in time it's checked.

## 5. Illustrative fraud-decision cost framework, with realistic review (requirement C)

**These cost figures are HYPOTHETICAL, illustrative portfolio
assumptions chosen by the author for this learning project.** They do
**not** represent Vesta Corporation, the IEEE-CIS competition, any real
bank, any real payments company, or any employer. See
`configs/cost_assumptions.json` (same disclaimer inline) and
`src/models/cost_model.py`.

| Cost input | Value | Meaning |
|---|---|---|
| `cost_per_missed_fraud` | $500 | Illustrative cost of a false negative — fraud approved |
| `cost_per_false_positive_decline` | $25 | Illustrative cost of a false positive — legitimate transaction declined outright |
| `cost_per_manual_review` | $3 | Illustrative cost of routing a transaction to manual review |
| `review_capacity_frac` | 5% | Illustrative review-team capacity, as a fraction of transactions evaluated |
| `review_fraud_catch_rate` | 80% | Illustrative probability a *reviewed* fraud case is actually caught (not a perfect backstop) |
| `review_false_positive_rate` | 5% | Illustrative probability a *reviewed* legitimate transaction is incorrectly flagged |

All six are config-driven (`load_cost_assumptions`), not hardcoded in
model logic.

**Three-action decisioning** (`classify_actions`) requires two ordered
thresholds: `score < review_threshold` → approve; `review_threshold <=
score < decline_threshold` → review; `score >= decline_threshold` →
decline. Invalid orderings are rejected with `ThresholdError`.

**Capacity-constrained review, with explicit prioritization (requirement
8).** When the number of transactions that land in the review band
exceeds `review_capacity_frac × n`, the review-band candidates are
ranked by calibrated probability **descending**, and only the top
`review_capacity_limit = floor(review_capacity_frac × n)` are actually
routed to review. The rest — the *lower*-risk review-band candidates —
**fall back to approve, not decline**: declining a review-band
transaction outright would silently convert a human-judgment case into
an automatic block, which is stricter than what the threshold pair
itself specifies. This is implemented deterministically (expected
values, not simulation) in `evaluate_decision_policy`
(`src/models/cost_model.py`), and demonstrated concretely in §6 below.

**Imperfect review, computed as expected value.** Of transactions
actually routed to review, `review_fraud_catch_rate` of the fraud cases
among them are counted as caught (the rest count as missed, same as an
approved fraud), and `review_false_positive_rate` of the legitimate
transactions among them are counted as false positives (same cost as an
outright decline false positive) — both as expected values, so results
stay deterministic and hand-computable.

## 6. Decision actions — threshold/cost tradeoff on `selection` (requirement D)

Computed with `evaluate_policy_grid` on **`selection` only**
(out-of-sample from calibration fitting), using sigmoid-calibrated
probabilities, over `REVIEW_THRESHOLD_GRID = [0.05, 0.10, 0.15, 0.20,
0.30, 0.40, 0.50]` × `DECLINE_THRESHOLD_GRID = [0.30, 0.40, 0.50, 0.60,
0.70, 0.80, 0.90]` (restricted to `review < decline`, 43 valid pairs),
sorted ascending by illustrative total cost. The full grid is in
`data/interim/milestone4_metrics.json::threshold_cost_grid_selection`.

**Lowest-cost row: `review_threshold=0.05, decline_threshold=0.30`**:

| Metric | Value |
|---|---|
| Fraud recall | 0.1804 |
| Precision | 0.1570 |
| Review volume (actually reviewed) | 2,872 |
| Review capacity limit | 2,872 (5% of 57,447) |
| **Desired** review volume (before capacity) | 12,652 (22.02% of `selection`) |
| **Overflow → approve** | 9,780 (77.3% of desired review volume) |
| Decline volume | 0 |
| Expected false positives | 121.05 |
| Expected false negatives | 1,639.2 |
| Illustrative total cost | $831,242.25 |

**Capacity-exhaustion demonstration (requirement 8, concretely, not just
theoretically):** at this policy, 22.02% of `selection` transactions fall
in the review band, but capacity only allows 5% to actually be reviewed.
The **2,872 highest-probability** candidates are kept; the other 9,780
(77.3% of the desired volume) fall back to **approve**, not decline. This
is why fraud recall (0.1804) is far below what it would be if review
capacity were unlimited — tightening or loosening capacity directly
trades off against recall (quantified further in §7).

**One incidental, genuine finding from the correction:** under
sigmoid-calibrated probabilities, **no transaction in `selection` reaches
a calibrated probability of 0.30** (the lowest value in
`DECLINE_THRESHOLD_GRID`) — `decline_volume` is exactly 0 in **every one
of the 43 pairs** in the grid. All cost-minimizing behavior here routes
through (capacity-limited) review, never outright decline, at these
illustrative costs and this calibrated-score distribution. This is a
property of this specific run, not a modeling bug — see §7 for what
makes decline start getting used.

**This selection-chosen policy was evaluated once, and only once, on the
temporal benchmark** (never re-optimized there):

| Metric | `selection` | `temporal_benchmark` |
|---|---|---|
| n rows | 57,447 | 115,534 |
| Fraud recall | 0.1804 | 0.1705 |
| Precision | 0.1570 | 0.1454 |
| Review capacity limit | 2,872 (5%) | 5,776 (5%) |
| Desired review volume | 12,652 (22.02%) | 32,758 (28.36%) |
| Overflow → approve | 9,780 | 26,982 |
| Decline volume | 0 | 0 |
| Illustrative total cost | $831,242.25 | $1,658,498.00 |

Recall transferred reasonably (0.1804→0.1705); the capacity-overflow
pattern is, if anything, *more* pronounced on the temporal benchmark
(28.36% desired vs. 5% capacity) than on `selection`, since the temporal
benchmark's fraud rate and score distribution shift slightly later in
time (§1, §4). As stated in §0, this is the temporal benchmark's
**third** inspection across Milestones 3–4 and is not a pristine check.

## 7. Sensitivity analysis on `selection` (requirement E, corrected review parameters)

`build_cost_variations` perturbs one input at a time — including, now,
all three review-realism parameters — and `sensitivity_analysis` re-runs
the full `selection` threshold grid under each, reporting the new lowest-
cost policy.

| Variation | Changed input | Selected (review, decline) | Fraud recall | Review vol. | Decline vol. | Illustrative cost |
|---|---|---|---|---|---|---|
| Base | — | (0.05, 0.30) | 0.1804 | 2,872 | 0 | $831,242.25 |
| `fraud_loss_2x` | `cost_per_missed_fraud`=$1,000 | (0.05, 0.30) | 0.1804 | 2,872 | 0 | $1,650,842.25 |
| `fraud_loss_half` | `cost_per_missed_fraud`=$250 | (0.05, 0.30) | 0.1804 | 2,872 | 0 | $421,442.25 |
| `false_positive_cost_2x` | `cost_per_false_positive_decline`=$50 | (0.05, 0.30) | 0.1804 | 2,872 | 0 | $834,268.50 |
| `false_positive_cost_half` | `cost_per_false_positive_decline`=$12.50 | (0.05, 0.30) | 0.1804 | 2,872 | 0 | $829,729.13 |
| `review_cost_2x` | `cost_per_manual_review`=$6 | (0.05, 0.30) | 0.1804 | 2,872 | 0 | $839,858.25 |
| `review_cost_half` | `cost_per_manual_review`=$1.50 | (0.05, 0.30) | 0.1804 | 2,872 | 0 | $826,934.25 |
| `review_cost_exceeds_decline_cost` | `cost_per_manual_review`=$60 | **(0.20, 0.30)** | **0.0552** | 488 | 0 | $974,517.50 |
| `fraud_loss_near_zero` | `cost_per_missed_fraud`=$5 | **(0.50, 0.90)** | **0.0000** | 0 | 0 | $10,000.00 |
| `review_capacity_tightened` | `review_capacity_frac`=1% | (0.05, 0.30) | **0.0616** | 574 (cap) | 0 | $940,647.00 |
| `review_catch_rate_low` | `review_fraud_catch_rate`=40% | (0.05, 0.30) | **0.0902** | 2,872 | 0 | $921,442.25 |
| `review_false_positive_rate_high` | `review_false_positive_rate`=30% | (0.05, 0.30) | 0.1804 | 2,872 | 0 | $846,373.50 |

**±2x variations on the three original cost inputs never change the
selected policy** — manual review ($3) stays far cheaper than either
alternative across these ranges, so only the reported dollar total
moves, same finding as the original run. Two deliberately extreme
variations still cross the relevant break-even points:
`review_cost_exceeds_decline_cost` ($60, above the $25 decline cost)
flips the policy toward outright decline for the first time and drops
recall to 0.0552; `fraud_loss_near_zero` ($5) collapses the policy to
all-approve.

**The three review-realism variations are the headline of this
correction (requirement 7/9):**
- **`review_capacity_tightened`** (5%→1% capacity): recall drops from
  0.1804 to 0.0616 — tightening capacity alone, with the same threshold
  pair, cuts recall by two-thirds, because a much larger share of the
  desired review volume now overflows to approve.
- **`review_catch_rate_low`** (80%→40% catch rate): recall drops from
  0.1804 to 0.0902 — almost exactly halved, since this parameter scales
  `reviewed_fraud_expected_caught` linearly and capacity/thresholds are
  unchanged.
- **`review_false_positive_rate_high`** (5%→30% review error rate):
  recall is unaffected (0.1804, as expected — this parameter only touches
  false positives among *legitimate* reviewed transactions), but expected
  false positives jump from 121.05 to 726.30 (6x, matching the 6x rate
  increase) and cost rises from $831,242 to $846,374.

**Core finding, updated for the correction:** the "best" threshold pair
is a function of assumed unit costs *and* of how realistically manual
review is modeled — a policy that looks like it achieves 0.1804 recall
under unlimited, perfect review can silently achieve far less once
capacity and imperfect catch rate are modeled explicitly, even with the
threshold pair unchanged.

## 8. Time-feature ablation (requirement F)

Unaffected by the calibration/review corrections above (no calibrator is
fit here; both pipelines are fit on `train` only and evaluated on
`val`/`temporal_benchmark`). Compares the full feature set against
`NUMERIC_FEATURES_NO_TIME_DRIFT` (`day_index`/`week_index` removed).

| Variant | Split | ROC-AUC | PR-AUC | Brier score |
|---|---|---|---|---|
| With `day_index`/`week_index` | Validation | 0.7652 | 0.1428 | 0.2599 |
| With `day_index`/`week_index` | Temporal benchmark | 0.7317 | 0.1202 | 0.3157 |
| Without `day_index`/`week_index` | Validation | 0.7636 | **0.1710** | 0.2301 |
| Without `day_index`/`week_index` | Temporal benchmark | **0.7351** | **0.1429** | 0.2552 |

Same conclusion as the original run: removing the two linear time-drift
proxies did not hurt temporal-benchmark performance — it slightly
improved it on every metric reported here. See the original reasoning
(extrapolation risk from a monotonically-increasing feature, treated as
suggestive from one split, not conclusive) — unchanged by this
correction.

## 9. Limitations

- **Review capacity, catch rate, and error rate remain illustrative and
  configurable, not measured** from any real review operation (§5, §7) —
  the correction replaces an unrealistic *assumption* (unlimited, perfect
  review) with a more realistic *mechanism*, not with empirical numbers.
- **All cost figures are hypothetical** (§5, §7) — every dollar number in
  this document is illustrative of mechanics, not a claim about real
  fraud-loss, chargeback, or review-operations economics for any company.
- **The `selection` split (57,447 rows) is smaller than Milestone 3's
  full validation period (118,790 rows)** — a direct, necessary cost of
  Correction 1: the validation period had to be split in half so
  calibration fitting and evaluation/selection use disjoint data. Metrics
  computed on `selection` therefore have somewhat more sampling variance
  than the original (uncorrected) validation-period numbers did.
- **The `review_cost_exceeds_decline_cost` and `fraud_loss_near_zero`
  sensitivity variations are deliberately extreme**, chosen because the
  milder ±2x variations all land on the same policy at base costs — the
  argmin-cost policy is locally flat near current assumptions and only
  moves once a variation crosses a specific break-even point. A different
  base cost configuration could have the opposite property.
- **The time-feature ablation (§8) is based on one train/validation/
  temporal-benchmark split** — directionally consistent with the
  extrapolation-risk concern but not large enough, from a single split,
  to rule out split-specific noise.
- **Calibration quality degrades somewhat from `cal_fit`'s fit
  distribution to `selection` to the temporal benchmark** (§4) — expected
  given the same non-stationarity Milestone 3 documented for ranking
  metrics; still a large improvement over uncalibrated at every point
  checked.
- **No production threshold is chosen here** (§6) — the threshold/cost
  grid is exploration, not a recommendation.
- **Carried over, unresolved from Milestones 1–3**: `C*`/`D*`/`M*`/`V*`
  feature-group leakage status remains open; `card2`/`card3`/`card5`/
  `addr2` still carry an unverified ordinal/interval assumption.
- **The temporal benchmark has now been touched three times total**
  across Milestones 3–4 (Milestone 3's baseline metrics, the original
  Milestone 4 run, and this correction) — any future milestone that
  revisits modeling choices and re-checks this same slice should treat it
  as increasingly contaminated and consider a fresh, never-touched slice
  instead.

## 10. Engineering & tests (requirement G)

`tests/models/test_split.py` (11 tests, 4 new for
`chronological_split_two_way`) — the validation-period sub-split covers
every row exactly once, is strictly chronological, never splits a
calendar day, rejects out-of-range `first_frac`, and its report matches
the actual rows.

`tests/models/test_calibration.py` (11 tests) — unchanged from the
original run; calibrator correctness is independent of which split is
passed in.

`tests/models/test_cost_model.py` (16 tests, 6 new for the review-realism
correction) — the three new `CostAssumptions` fields default to the
exact pre-correction (unlimited/perfect) numeric behavior; out-of-range
values raise `CostAssumptionError`; capacity overflow falls back to
approve, never decline; capacity prioritizes the highest-risk candidates;
an imperfect catch rate reduces expected recall proportionally; a
nonzero review error rate adds the expected false positives it implies.

`tests/models/test_train_calibrated.py` (8 tests, including a
real-data-gated smoke test) — `calibration_comparison`'s corrected
6-argument signature returns all three methods on both out-of-sample
splits (`selection`, `temporal_benchmark`); mutating temporal-benchmark
raw probabilities after fitting leaves every `selection`-side number
byte-for-byte unchanged (the leakage guarantee this correction is built
around); `select_best_calibration_method` reads only the requested
split's metric and demonstrably prefers sigmoid when tied and switches to
isotonic only on a clear (>3%) improvement; `time_feature_ablation`
returns the expected structure; `build_cost_variations` produces the
expected 10 labeled variations (including the three new review-realism
ones) preserving unvaried cost fields; and, against real data, a full run
confirms sigmoid preserves `selection` ROC-AUC exactly, isotonic preserves
it within a small tolerance, both calibrated methods beat uncalibrated
Brier score on `selection`, the cost grid is non-empty and cost-sorted,
and both ablation variants exceed 0.5 ROC-AUC on the temporal benchmark.

**Full suite: 98 passed, 1 skipped** (same intentional missing-raw-file
error-path skip carried from Milestones 1–3; net +12 tests versus the
original Milestone 4 run's 86 passed).

## 11. Runtime

Unchanged in character from the original run — fitting the baseline
pipeline, two ablation-variant pipelines, two calibrators, computing
reliability tables, and evaluating a 43-pair threshold grid × 11
sensitivity variations (base + 10 labeled) all complete in well under a
minute on the same machine used for Milestones 1–3. No new dependencies.

## 12. Recommended next milestone

Unchanged from the original run:

1. **A non-linear model (gradient-boosted trees), explicitly deferred
   from Milestones 3–4**, evaluated on the same chronological split and
   held to the same calibration + cost-decisioning framework built here.
2. **Resolve the `C*`/`D*`/`M*`/`V*` leakage question** carried unresolved
   since Milestone 1.

Per this correction's explicit instruction, work stops here for review;
no Milestone 5 work has been started.
