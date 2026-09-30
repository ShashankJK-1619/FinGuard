# TransactionDT Structure — Milestone 1 Section H

No model or split created here. This documents what TransactionDT structurally is, to inform a time-aware split design in a later milestone.

- train_transaction TransactionDT range: 86,400 to 15,811,131 (182.0 days if seconds) across 590,540 rows
- test_transaction TransactionDT range: 18,403,224 to 34,214,345 across 506,691 rows

- Values consistent with an absolute Unix timestamp (>= 1,000,000,000): False
- test_transaction's minimum TransactionDT >= train_transaction's maximum: True

## Interpretation

TransactionDT values are far too small to be Unix timestamps (which would be in the billions for any recent date). This is consistent with the competition's documentation that TransactionDT is a **relative time delta in seconds from an undisclosed reference point**, not a wall-clock timestamp. We cannot recover the actual calendar date of any transaction from this field alone.

test_transaction's TransactionDT values start after train_transaction's maximum value — the provided train/test split is itself time-based, not random. This is a strong signal that fraud patterns are expected to drift over time, and that evaluating on a random split of train data would overstate how the model performs on genuinely future, unseen traffic.

## Why random splitting is risky for fraud modeling

Fraud behavior is non-stationary: new fraud rings, new attack patterns, and merchant/product mix all shift over time. A random split lets the model "see" transactions from the same time window it will be tested on, including potentially correlated fraud rings active only in that window. That inflates offline metrics relative to real-world performance, where the model only ever sees the future. A time-based split (train on earlier TransactionDT, validate/test on strictly later TransactionDT) is the defensible design — to be implemented in a later milestone, not here.
