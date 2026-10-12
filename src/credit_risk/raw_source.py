"""Module C, Milestone 2: provenance/source-metadata record for the ACTUAL
downloaded raw LendingClub file.

This mirrors the project's config-driven, loudly-validated pattern
(`src/payments/config.py`, `src/credit_risk/config.py`): the facts about
*where this file came from* live in
`configs/credit_risk_raw_source_metadata.json`, authored once, at download
time, by a human reading that mirror's own page -- never inferred,
guessed, or copied from a different mirror's listing. This module only
loads and validates that record; it does not re-derive or re-check any of
its factual fields against the live internet (no network access here).

Critical discipline this module exists to enforce: `license_statement`
describes what *this specific mirror* shows, not a license grant from the
original publisher (LendingClub). `license_statement_is_from_original_publisher`
must always be explicit (`True`/`False`), never left implicit, so that no
later code or doc can silently conflate a mirror's absence-of-license with
an actual open-license grant from LendingClub itself.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RAW_SOURCE_METADATA_FILE = (
    PROJECT_ROOT / "configs" / "credit_risk_raw_source_metadata.json"
)


class RawSourceMetadataError(ValueError):
    pass


@dataclass(frozen=True)
class RawSourceMetadata:
    dataset_name: str
    source_url: str
    mirror_provider: str
    local_path: str
    file_size_bytes: int
    sha256_checksum: str
    license_statement: str
    license_statement_is_from_original_publisher: bool
    access_date: str

    def __post_init__(self) -> None:
        if not self.source_url.startswith("http"):
            raise RawSourceMetadataError(
                f"source_url must be a fully-qualified URL; got {self.source_url!r}"
            )
        if self.file_size_bytes <= 0:
            raise RawSourceMetadataError(
                f"file_size_bytes must be > 0; got {self.file_size_bytes}"
            )
        if len(self.sha256_checksum) != 64 or not all(
            c in "0123456789abcdef" for c in self.sha256_checksum.lower()
        ):
            raise RawSourceMetadataError(
                f"sha256_checksum must be a 64-character hex string; got "
                f"{self.sha256_checksum!r}"
            )
        if self.license_statement_is_from_original_publisher:
            # This repo has never encountered a mirror that shows a
            # publisher-attributed license -- if one ever does, the
            # provenance narrative in docs must be written by a human for
            # that specific case, not silently accepted by this
            # dataclass's default behavior.
            raise RawSourceMetadataError(
                "license_statement_is_from_original_publisher=True is not "
                "supported by this loader without an accompanying human-"
                "reviewed doc update -- see docs/moduleC_milestone2_"
                "real_data_validation.md before setting this."
            )
        if not self.license_statement.strip():
            raise RawSourceMetadataError("license_statement must not be empty")


def load_raw_source_metadata(
    config_file: Path = DEFAULT_RAW_SOURCE_METADATA_FILE,
) -> RawSourceMetadata:
    with open(config_file) as f:
        cfg = json.load(f)
    return RawSourceMetadata(
        dataset_name=cfg["dataset_name"],
        source_url=cfg["source_url"],
        mirror_provider=cfg["mirror_provider"],
        local_path=cfg["local_path"],
        file_size_bytes=cfg["file_size_bytes"],
        sha256_checksum=cfg["sha256_checksum"],
        license_statement=cfg["license_statement"],
        license_statement_is_from_original_publisher=cfg[
            "license_statement_is_from_original_publisher"
        ],
        access_date=cfg["access_date"],
    )


def verify_local_file_integrity(
    metadata: RawSourceMetadata, project_root: Path = PROJECT_ROOT
) -> tuple[bool, str]:
    """Recomputes the SHA256 of the local file and compares to the recorded
    checksum. Returns (ok, message). Does not raise -- callers (e.g. a
    verification script, or a skip-marker in conftest) decide what to do
    with a mismatch or a missing file."""
    path = project_root / metadata.local_path
    if not path.exists():
        return False, f"Local file not found at {path}"
    actual_size = path.stat().st_size
    if actual_size != metadata.file_size_bytes:
        return False, (
            f"Size mismatch: recorded {metadata.file_size_bytes}, "
            f"actual {actual_size}"
        )
    sha256 = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024 * 8), b""):
            sha256.update(chunk)
    actual_checksum = sha256.hexdigest()
    if actual_checksum != metadata.sha256_checksum:
        return False, (
            f"SHA256 mismatch: recorded {metadata.sha256_checksum}, "
            f"actual {actual_checksum}"
        )
    return True, "OK: size and SHA256 checksum match recorded metadata"
