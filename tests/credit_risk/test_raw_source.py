import json

import pytest

from src.credit_risk.raw_source import (
    RawSourceMetadata,
    RawSourceMetadataError,
    load_raw_source_metadata,
    verify_local_file_integrity,
)
from tests.credit_risk.conftest import requires_credit_risk_raw_data

VALID_KWARGS = dict(
    dataset_name="Test Dataset",
    source_url="https://example.com/data.csv",
    mirror_provider="Example Mirror",
    local_path="data/raw/data.csv",
    file_size_bytes=100,
    sha256_checksum="a" * 64,
    license_statement="No license shown.",
    license_statement_is_from_original_publisher=False,
    access_date="2026-10-12",
)


def test_load_real_metadata_file_is_valid():
    metadata = load_raw_source_metadata()
    assert metadata.source_url.startswith("https://")
    assert len(metadata.sha256_checksum) == 64
    assert metadata.license_statement_is_from_original_publisher is False


def test_rejects_non_url_source():
    with pytest.raises(RawSourceMetadataError):
        RawSourceMetadata(**{**VALID_KWARGS, "source_url": "not-a-url"})


def test_rejects_bad_checksum_length():
    with pytest.raises(RawSourceMetadataError):
        RawSourceMetadata(**{**VALID_KWARGS, "sha256_checksum": "abc123"})


def test_rejects_zero_file_size():
    with pytest.raises(RawSourceMetadataError):
        RawSourceMetadata(**{**VALID_KWARGS, "file_size_bytes": 0})


def test_rejects_claiming_publisher_license_without_review():
    with pytest.raises(RawSourceMetadataError):
        RawSourceMetadata(**{**VALID_KWARGS, "license_statement_is_from_original_publisher": True})


def test_rejects_empty_license_statement():
    with pytest.raises(RawSourceMetadataError):
        RawSourceMetadata(**{**VALID_KWARGS, "license_statement": "   "})


@requires_credit_risk_raw_data
def test_verify_local_file_integrity_matches_recorded_checksum():
    metadata = load_raw_source_metadata()
    ok, message = verify_local_file_integrity(metadata)
    assert ok, message


def test_verify_local_file_integrity_reports_missing_file(tmp_path):
    metadata = RawSourceMetadata(**{**VALID_KWARGS, "local_path": str(tmp_path / "does_not_exist.csv")})
    ok, message = verify_local_file_integrity(metadata, project_root=tmp_path.parent)
    assert not ok
    assert "not found" in message.lower()
