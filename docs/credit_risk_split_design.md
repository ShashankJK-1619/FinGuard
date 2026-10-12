# Module C, Milestone 2, Section G — Chronological Split Design

## Why this doc exists

Milestone 1 flagged chronological-split feasibility as a design concern
but noted one dataset-specific wrinkle for the implementing milestone:
cutoffs must account for the resolution lag whichever target imposes, or
a naive "most recent window" slice would be overwhelmingly censored.
This doc records the actual boundary choices, made only after inspecting
the real censoring-by-origination-year table for each target
(`docs/moduleC_milestone2_real_data_validation.md` Section E), not
picked arbitrarily.

## Governance correction: there are now TWO split designs, not one

A pre-commit review corrected Milestone 2's target governance (see
`docs/decision_log.md`'s "Milestone 2 target-governance correction"
entry and `configs/credit_risk_target_governance.json`, the
machine-readable source of truth): `terminal_adverse_outcome` is the
**PRIMARY** modeling target for Milestone 3, and
`default_12m_last_pymnt_proxy` is retained only as a **SECONDARY**,
sensitivity-analysis-only target.

Because the two targets resolve on fundamentally different schedules —
a fixed 12-month window for the proxy target vs. each loan's own 36- or
60-month maturity for the terminal target — **the original single split
could not simply be reused for the new primary target.** Reusing the
proxy split's boundaries (train through 2014, validation 2015-2016,
benchmark 2017) against the terminal target would have put a **100%
censored** year (2017) into the benchmark bucket and 33-75% censored
years (2015-2016) into validation — see the empirical table below. A
dedicated, earlier-boundaried split was designed instead. Both splits
are preserved as separate, explicitly-named configs, both loadable via
`src/credit_risk/chronological_split.py::load_split_design`:

| | config file | target | enforced by |
|---|---|---|---|
| **Primary** (default) | `configs/credit_risk_split_design_primary_terminal.json` | `terminal_adverse_outcome` | `chronological_split.py`'s `DEFAULT_SPLIT_DESIGN_FILE` |
| **Secondary** | `configs/credit_risk_split_design_secondary_proxy.json` | `default_12m_last_pymnt_proxy` | `chronological_split.py`'s `SECONDARY_PROXY_SPLIT_DESIGN_FILE` |

No change was needed to `SplitDesign`/`build_chronological_split`'s
*code* to support this — `outcome_window_months` already generically
means "how long after the benchmark cutoff must elapse before every
benchmark-bucket row is guaranteed resolved," so setting it to `60` (the
real empirical maximum of LendingClub's `term` column — confirmed
distinct values `[36, 60]`) rather than `12` makes the existing
`benchmark_end + outcome_window_months <= data_as_of_date` validation
rule correctly guarantee full resolution for the variable-horizon
primary target too.

---

## Primary split — `terminal_adverse_outcome`

### The empirical basis

Censoring % by origination year, under `terminal_adverse_outcome`
(resolved only once a loan passes its own scheduled maturity date):

| origination year | total rows | pct censored |
|---|---|---|
| 2007 | 603 | 0.00% |
| 2008 | 2,393 | 0.00% |
| 2009 | 5,281 | 0.00% |
| 2010 | 12,537 | 0.00% |
| 2011 | 21,721 | 0.00% |
| 2012 | 53,367 | 0.00% |
| 2013 | 134,814 | 0.0074% (10 rows) |
| 2014 | 235,629 | 23.43% |
| 2015 | 421,095 | 32.79% |
| 2016 | 434,407 | 74.85% |
| 2017 | 443,579 | 100.00% |
| 2018 | 495,242 | 100.00% |

This is why the proxy split's boundaries (benchmark = 2017) were
unusable for this target: 2017 is **100% censored** under
`terminal_adverse_outcome` (no 2017-originated loan, 36- or 60-month
term, has reached its own maturity as of the inferred
`data_as_of_date`, 2019-04-01), and 2015-2016 are already 33-75%
censored. Only origination years through 2013 are (essentially) fully
resolved.

### The boundaries chosen

`configs/credit_risk_split_design_primary_terminal.json`:

| bucket | start | end |
|---|---|---|
| train | 2007-06-01 | 2011-12-31 |
| validation | 2012-01-01 | 2012-12-31 |
| benchmark | 2013-01-01 | 2013-12-31 |
| excluded_too_recent | 2014-01-01 | (open-ended) |

Same three design principles as the secondary split (see below), applied
with `outcome_window_months=60`:

1. **The entire 2014-onward tail is excluded wholesale, not partially
   included** — same survivorship-bias rationale as the secondary
   split's 2018 exclusion.
2. **`benchmark` (2013) is the newest year that is (near-)fully resolved
   under the 60-month horizon**, held out as a temporal-generalization
   check, never used for fitting or threshold selection.
3. **`SplitDesign.__post_init__`'s existing validation rule** —
   `benchmark_end + outcome_window_months <= data_as_of_date` — is
   satisfied here as `2013-12-31 + 60 months = 2018-12-31 <=
   2019-04-01`. No new validation code was needed; generalizing
   `outcome_window_months` from 12 to 60 was sufficient.

### Resulting counts (real data, `terminal_adverse_outcome` target)

| split_bucket | total_rows | censored_excluded | labeled_rows | default_count | default_prevalence_pct | origination range |
|---|---|---|---|---|---|---|
| train | 42,535 | 0 | 42,535 | 6,431 | 15.119% | 2007-06 – 2011-12 |
| validation | 53,367 | 0 | 53,367 | 8,644 | 16.197% | 2012-01 – 2012-12 |
| benchmark | 134,814 | 10 | 134,804 | 21,024 | 15.596% | 2013-01 – 2013-12 |
| excluded_too_recent | 2,029,952 | 1,457,275 | 572,677 | 88,228 | 15.406% (of labeled) | 2014-01 – 2018-12 |

All 2,260,668 rows are accounted for across the four buckets. The 10
censored rows in `benchmark` (loans that, despite an origination month
in 2013, have a computed maturity date fractionally past
`data_as_of_date` — e.g. late-December originations with the full
60-month term) are excluded from `benchmark`'s labeled count per the
hard no-censored-rows-in-labeled-buckets guarantee (below); they do not
violate it.

**Hard guarantee, checked programmatically**: `build_chronological_split`
raises `SplitDesignError` if any censored row is ever found inside the
*labeled* subset of `train`, `validation`, or `benchmark` — verified
empirically above (0 rows in each bucket's labeled subset are
censored).

**Sample-size tradeoff, stated plainly**: this primary split is much
smaller (train+validation+benchmark = 230,716 rows) than the secondary
split's (1,765,426 rows), because the terminal target demands each
loan reach its own, often multi-year, maturity before it is labeled at
all. This is a direct, expected consequence of correctly designating a
variable-horizon target as primary, not an error — any future milestone
fitting a model on this split should expect and plan around the smaller
labeled sample, not be surprised by it.

---

## Secondary split — `default_12m_last_pymnt_proxy` (sensitivity analysis only)

Preserved **unchanged** from the original Milestone 2 draft — only its
file name/location moved, from the single `credit_risk_split_design.json`
to `configs/credit_risk_split_design_secondary_proxy.json`, so it can
coexist explicitly alongside the new primary split.

### The empirical basis

Censoring % by origination year, for `default_12m_last_pymnt_proxy`:

| origination year | pct censored |
|---|---|
| 2007-2017 | 0.0% |
| 2018 | 68.9% |

Every origination year through 2017 is **fully resolved** under the
inferred `data_as_of_date` (2019-04-01): a loan originated by
2017-12-31, plus its 12-month observation window, resolves by
2018-12-31 at the latest — well before the snapshot. Only the 2018
vintage has insufficient elapsed follow-up.

### The boundaries chosen

`configs/credit_risk_split_design_secondary_proxy.json`:

| bucket | start | end |
|---|---|---|
| train | 2007-06-01 | 2014-12-31 |
| validation | 2015-01-01 | 2016-12-31 |
| benchmark | 2017-01-01 | 2017-12-31 |
| excluded_too_recent | 2018-01-01 | (open-ended) |

Three deliberate choices:

1. **The entire 2018 vintage is excluded, not partially included.** A
   loan-level filter that kept only the fastest-to-resolve early-2018
   loans would introduce a subtle survivorship distortion — loans that
   happen to default or pay off quickly are systematically different
   from loans still open. The project rule "do not touch later periods
   merely to optimize performance" rules out cherry-picking a resolved
   subset of the newest data just to squeeze in more recent rows.
2. **`benchmark` (2017) is the newest fully-resolved year, held out as
   a temporal-generalization check**, analogous in spirit to Module B's
   `temporal_benchmark` (Milestones 3-5) — a slice never used for
   fitting or threshold selection, only for a forward-looking read.
3. **`SplitDesign.__post_init__` enforces `benchmark_end +
   outcome_window_months <= data_as_of_date` as a hard validation
   rule**, not just a convention followed by this one config —
   `test_rejects_benchmark_window_not_yet_resolved` exercises this by
   constructing a design where `benchmark_end` is moved close enough to
   `data_as_of_date` that the +12-month window would not have elapsed,
   and asserts `SplitDesignError` is raised.

### Resulting counts (real data, `default_12m_last_pymnt_proxy` target)

| split_bucket | total_rows | censored_excluded | labeled_rows | default_count | default_prevalence_pct | origination range |
|---|---|---|---|---|---|---|
| train | 466,345 | 0 | 466,345 | 24,147 | 5.178% | 2007-06 – 2014-12 |
| validation | 855,502 | 0 | 855,502 | 55,673 | 6.508% | 2015-01 – 2016-12 |
| benchmark | 443,579 | 0 | 443,579 | 27,862 | 6.281% | 2017-01 – 2017-12 |
| excluded_too_recent | 495,242 | 341,325 | 153,917 | 8,852 | 5.751% (of labeled) | 2018-01 – 2018-12 |

All 2,260,668 rows are accounted for across the four buckets. `excluded_
too_recent` is reported for transparency — its 153,917 already-resolved
rows are a useful empirical cross-check, but by design this bucket is
never one of the three evaluation-eligible splits.

**Hard guarantee, checked programmatically, not just asserted in
prose**: `build_chronological_split` raises `SplitDesignError` if any
censored row is ever found inside the labeled subset of `train`,
`validation`, or `benchmark` — verified empirically above (0 censored
excluded in all three). This directly implements the project's
"no unresolved/censored loans in labeled evaluation splits" rule.

---

## What these splits are for, and what they are not

- The **primary** split (`terminal_adverse_outcome`) is the one
  Milestone 3 modeling must use by default — it is what
  `load_split_design()`'s no-argument default now returns.
- The **secondary** split (`default_12m_last_pymnt_proxy`) must only be
  used for sensitivity analysis alongside the primary target, never
  promoted back to primary status, and its documented one-directional
  bias (`last_pymnt_d` is NOT a default-event timestamp — see
  `docs/moduleC_milestone2_real_data_validation.md` Section D/J and
  `configs/credit_risk_target_governance.json`) must be preserved
  wherever it is used.
- Mild prevalence drift across buckets, in both splits, is reported as
  an observed fact, not adjusted for — no resampling, reweighting, or
  boundary-shifting was done to flatten it. Any future milestone fitting
  a model on either split should expect and account for this drift, not
  be surprised by it.
- No model has been fit against either split yet — this document
  defines the splits only, per this milestone's explicit scope boundary.
