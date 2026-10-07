#!/usr/bin/env python3
"""Run each prepared verification case once under the standard resource caps."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("matrix_root", type=Path)
    parser.add_argument("--executable", type=Path, default=ROOT / "build" / "hawkmoth_hover")
    parser.add_argument("--ranks", type=int, default=1)
    parser.add_argument("--max-rss-gb", type=float, default=8.0)
    parser.add_argument("--min-free-disk-gb", type=float, default=40.0)
    args = parser.parse_args()
    if not 1 <= args.ranks <= 8:
        parser.error("ranks must be between 1 and 8")
    matrix_path = args.matrix_root / "prepared_matrix.csv"
    if not matrix_path.is_file():
        parser.error(f"missing prepared matrix: {matrix_path}")
    results = []
    with matrix_path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    for row in rows:
        run_dir = Path(row["run_dir"])
        record_path = run_dir / "resource_record.json"
        if record_path.exists():
            results.append({"run_id": row["run_id"], "status": "SKIPPED_ALREADY_RECORDED"})
            continue
        command = [sys.executable, str(ROOT / "tools" / "run_case.py"), str(run_dir),
                   "--executable", str(args.executable), "--ranks", str(args.ranks),
                   "--max-rss-gb", str(args.max_rss_gb), "--min-free-disk-gb", str(args.min_free_disk_gb)]
        completed = subprocess.run(command, check=False, capture_output=True, text=True)
        status = "COMPLETED" if completed.returncode == 0 else "FAILED"
        results.append({"run_id": row["run_id"], "status": status,
                        "returncode": completed.returncode,
                        "runner_output": completed.stdout.strip(),
                        "runner_error": completed.stderr.strip()})
    output = {"matrix_root": str(args.matrix_root), "cases": results,
              "completed": sum(case["status"] == "COMPLETED" for case in results),
              "failed": sum(case["status"] == "FAILED" for case in results)}
    (args.matrix_root / "matrix_execution.json").write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output, indent=2))
    raise SystemExit(1 if output["failed"] else 0)


if __name__ == "__main__":
    main()
