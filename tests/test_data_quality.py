from src.data.config import load_paths
from src.data.loaders import get_connection, register_raw_views
from tests.conftest import requires_raw_data


@requires_raw_data
def test_transaction_ids_are_unique_in_train_and_test():
    con = get_connection()
    register_raw_views(con, load_paths())
    for table in ("train_transaction", "test_transaction"):
        total = con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        distinct = con.execute(f"SELECT COUNT(DISTINCT TransactionID) FROM {table}").fetchone()[0]
        assert total == distinct, f"{table} has duplicate TransactionID values"


@requires_raw_data
def test_isfraud_is_binary_in_train():
    con = get_connection()
    register_raw_views(con, load_paths())
    values = con.execute("SELECT DISTINCT isFraud FROM train_transaction").fetchdf()["isFraud"].tolist()
    assert set(values) <= {0, 1}


@requires_raw_data
def test_isfraud_not_present_in_test():
    con = get_connection()
    register_raw_views(con, load_paths())
    cols = con.execute("SELECT * FROM test_transaction LIMIT 0").fetchdf().columns
    assert "isFraud" not in cols


@requires_raw_data
def test_identity_transactionid_is_subset_of_transaction_transactionid():
    con = get_connection()
    register_raw_views(con, load_paths())
    orphans = con.execute(
        "SELECT COUNT(*) FROM train_identity i "
        "LEFT JOIN train_transaction t ON i.TransactionID = t.TransactionID "
        "WHERE t.TransactionID IS NULL"
    ).fetchone()[0]
    assert orphans == 0, "train_identity has TransactionID values absent from train_transaction"


@requires_raw_data
def test_transaction_amount_is_non_negative():
    con = get_connection()
    register_raw_views(con, load_paths())
    for table in ("train_transaction", "test_transaction"):
        negative = con.execute(f"SELECT COUNT(*) FROM {table} WHERE TransactionAmt < 0").fetchone()[0]
        assert negative == 0, f"{table} has negative TransactionAmt values"


@requires_raw_data
def test_loading_is_reproducible_across_two_connections():
    paths = load_paths()
    con_a = get_connection()
    register_raw_views(con_a, paths)
    con_b = get_connection()
    register_raw_views(con_b, paths)

    row_count_a = con_a.execute("SELECT COUNT(*) FROM train_transaction").fetchone()[0]
    row_count_b = con_b.execute("SELECT COUNT(*) FROM train_transaction").fetchone()[0]
    assert row_count_a == row_count_b

    schema_a = con_a.execute("DESCRIBE train_transaction").fetchdf()["column_type"].tolist()
    schema_b = con_b.execute("DESCRIBE train_transaction").fetchdf()["column_type"].tolist()
    assert schema_a == schema_b
