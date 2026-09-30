"""Configuration-driven path resolution.

All file locations come from configs/paths.json so that no script hardcodes
a path. Milestone 1 scope: local filesystem only, no remote storage.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PATHS_CONFIG_FILE = PROJECT_ROOT / "configs" / "paths.json"


@dataclass(frozen=True)
class Paths:
    raw_dir: Path
    interim_dir: Path
    processed_dir: Path
    docs_dir: Path
    raw_files: dict[str, str]

    def raw_file(self, key: str) -> Path:
        return self.raw_dir / self.raw_files[key]


def load_paths(config_file: Path = PATHS_CONFIG_FILE) -> Paths:
    with open(config_file) as f:
        cfg = json.load(f)
    return Paths(
        raw_dir=PROJECT_ROOT / cfg["raw_dir"],
        interim_dir=PROJECT_ROOT / cfg["interim_dir"],
        processed_dir=PROJECT_ROOT / cfg["processed_dir"],
        docs_dir=PROJECT_ROOT / cfg["docs_dir"],
        raw_files=cfg["raw_files"],
    )
