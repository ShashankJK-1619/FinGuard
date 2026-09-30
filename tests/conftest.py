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
