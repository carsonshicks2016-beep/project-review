#!/usr/bin/env python3
"""Repair cross-edition Fable auto-ladder state without guessing identity."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from supra.fable_editions import repair_manifest_auto_blocks  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Move a leaked Fable manifest auto block only when its inflight "
            "checkpoint safely proves the destination edition."
        )
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=ROOT,
        help="checkpoint/project directory (default: repository root)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="report provable moves without changing or backing up files",
    )
    parser.add_argument(
        "--timestamp",
        help="optional deterministic backup directory stamp",
    )
    args = parser.parse_args()

    report = repair_manifest_auto_blocks(
        args.root,
        dry_run=args.dry_run,
        timestamp=args.timestamp,
    )
    print(json.dumps(report, indent=2))
    return 0 if not report["warnings"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
