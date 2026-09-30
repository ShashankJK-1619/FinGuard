import pytest

from src.data.config import load_paths
from src.data.loaders import raw_files_present

paths = load_paths()
_present = raw_files_present(paths)

requires_raw_data = pytest.mark.skipif(
    not all(_present.values()),
    reason=(
        "IEEE-CIS raw CSVs not present in data/raw/. Download from "
        "https://www.kaggle.com/competitions/ieee-fraud-detection/data "
        "to run this test."
    ),
)

_baseline_parquet_path = paths.processed_dir / "baseline_features_v1.parquet"

requires_baseline_features = pytest.mark.skipif(
    not _baseline_parquet_path.exists(),
    reason=(
        f"{_baseline_parquet_path} not present. Run "
        "`.venv/bin/python -m src.features.build_baseline_dataset` (Milestone 2) first."
    ),
)
