import pytest

from src.data.config import load_paths
from src.data.loaders import get_connection, raw_files_present, register_raw_views


def test_raw_files_present_reports_missing_files_without_crashing():
    status = raw_files_present()
    assert set(status.keys()) == {"train_transaction", "train_identity", "test_transaction", "test_identity"}
    assert all(isinstance(v, bool) for v in status.values())


def test_register_raw_views_raises_clear_error_when_file_missing():
    paths = load_paths()
    status = raw_files_present(paths)
    if all(status.values()):
        pytest.skip("Raw data present; missing-file error path not exercised here.")
    con = get_connection()
    with pytest.raises(FileNotFoundError, match="data/raw"):
        register_raw_views(con, paths)
