# Join Validation — Milestone 1 Section D

Validates the TransactionID relationship between transaction and identity tables. No modeling performed.

## `train_transaction` ⨝ `train_identity`

- Transaction rows: 590,540
- Matched to identity (INNER JOIN): 144,233 (24.4239%)
- Identity rows with duplicate TransactionID: 0
- Row count after LEFT JOIN transaction->identity: 590,540
- LEFT JOIN changes transaction row count: False
- Many-to-many relationship detected: False

Interpretation: the relationship is one-to-zero-or-one (each transaction has at most one identity row). Not every transaction has identity data — that is expected per the competition's documented design (identity data is only available for a subset of transactions), not a data quality bug.

## `test_transaction` ⨝ `test_identity`

- Transaction rows: 506,691
- Matched to identity (INNER JOIN): 141,907 (28.0066%)
- Identity rows with duplicate TransactionID: 0
- Row count after LEFT JOIN transaction->identity: 506,691
- LEFT JOIN changes transaction row count: False
- Many-to-many relationship detected: False

Interpretation: the relationship is one-to-zero-or-one (each transaction has at most one identity row). Not every transaction has identity data — that is expected per the competition's documented design (identity data is only available for a subset of transactions), not a data quality bug.
