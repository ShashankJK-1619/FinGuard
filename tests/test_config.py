from src.data.config import load_paths


def test_paths_resolve_under_project_root():
    paths = load_paths()
    assert paths.raw_dir.name == "raw"
    assert paths.interim_dir.name == "interim"
    assert paths.processed_dir.name == "processed"
    assert paths.docs_dir.exists()  # docs/ is committed, unlike data/


def test_raw_files_config_has_expected_keys():
    paths = load_paths()
    expected = {"train_transaction", "train_identity", "test_transaction", "test_identity"}
    assert set(paths.raw_files.keys()) == expected
