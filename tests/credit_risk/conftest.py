import pytest

from src.credit_risk.raw_loader import raw_file_present
from src.credit_risk.raw_source import load_raw_source_metadata

_metadata = load_raw_source_metadata()
_present = raw_file_present(_metadata)

requires_credit_risk_raw_data = pytest.mark.skipif(
    not _present,
    reason=(
        f"Real LendingClub raw CSV not present at {_metadata.local_path}. "
        f"Download from {_metadata.source_url} ({_metadata.mirror_provider}) "
        "to run this test -- see docs/moduleC_milestone2_real_data_validation.md Section A."
    ),
)
