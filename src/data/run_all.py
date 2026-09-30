"""Run every Milestone 1 inspection/validation script in sequence.

Run: python -m src.data.run_all
"""
from __future__ import annotations

from src.data import analyze_time, inspect_dataset, leakage_checks, validate_join


def main() -> int:
    for module in (inspect_dataset, validate_join, analyze_time, leakage_checks):
        rc = module.main()
        if rc != 0:
            return rc
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
