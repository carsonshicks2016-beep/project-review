#!/usr/bin/env python3
"""Run every checked-in faithful-v2 foundation validator as one gate."""
from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
VALIDATORS = (
    "tools/validate_faithful_evidence.py",
    "tools/validate_faithful_runtime_spec.py",
    "tools/validate_faithful_v2_core.py",
    "tools/validate_faithful_record_core.py",
    "tools/validate_faithful_policy.py",
    "tools/validate_faithful_av.py",
    "tools/validate_faithful_program.py",
)


def main() -> None:
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(ROOT) + (
        os.pathsep + environment["PYTHONPATH"]
        if environment.get("PYTHONPATH") else ""
    )
    failures: list[str] = []
    for relative in VALIDATORS:
        path = ROOT / relative
        print(f"\n=== {relative} ===", flush=True)
        if not path.is_file():
            print(f"missing validator: {path}", file=sys.stderr, flush=True)
            failures.append(relative)
            continue
        result = subprocess.run(
            [sys.executable, str(path)],
            cwd=ROOT,
            env=environment,
            check=False,
        )
        if result.returncode != 0:
            failures.append(f"{relative} (exit {result.returncode})")
    if failures:
        raise SystemExit("faithful-v2 aggregate validation failed: " + ", ".join(failures))
    print("\nfaithful-v2 aggregate validation: PASS", flush=True)


if __name__ == "__main__":
    main()
