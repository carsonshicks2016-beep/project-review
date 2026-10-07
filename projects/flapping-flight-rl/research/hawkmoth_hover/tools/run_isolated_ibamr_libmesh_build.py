#!/usr/bin/env python3
"""Build IBAMR 0.19.0 with libMesh in an isolated prefix with a disk guard."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import signal
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def disk_gib(path: Path) -> float:
    return shutil.disk_usage(path).free / (1024 ** 3)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--autoibamr-root", type=Path, required=True)
    parser.add_argument("--prefix", type=Path, required=True)
    parser.add_argument("--record-dir", type=Path, required=True)
    parser.add_argument("--minimum-free-gib", type=float, default=40.0)
    parser.add_argument("--jobs", type=int, default=2)
    args = parser.parse_args()

    auto_root = args.autoibamr_root.resolve()
    prefix = args.prefix.resolve()
    record_dir = args.record_dir.resolve()
    script = auto_root / "autoibamr.sh"
    recipe = auto_root / "IBAMR-toolchain/packages/libmesh.package"
    if not script.is_file() or not recipe.is_file():
        parser.error("autoibamr script or pinned libMesh package recipe is missing")
    if record_dir.exists() and any(record_dir.iterdir()):
        parser.error(f"refusing to overwrite nonempty record directory: {record_dir}")
    free_before = disk_gib(record_dir.parent if record_dir.parent.exists() else Path.home())
    if free_before < args.minimum_free_gib:
        parser.error(f"only {free_before:.2f} GiB free; minimum is {args.minimum_free_gib:.2f} GiB")
    if prefix.exists() and any(prefix.iterdir()):
        parser.error(f"refusing to reuse nonempty build prefix: {prefix}")

    record_dir.mkdir(parents=True, exist_ok=True)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    start = datetime.now(timezone.utc)
    stdout_path = record_dir / "build_stdout.log"
    stderr_path = record_dir / "build_stderr.log"
    result_path = record_dir / "build_result.json"
    command = [str(script), f"--prefix={prefix}", f"--jobs={args.jobs}", "--yes"]
    metadata = {
        "study_id": record_dir.name,
        "started_utc": start.isoformat(),
        "status": "RUNNING",
        "command": command,
        "working_directory": str(auto_root),
        "isolated_prefix": str(prefix),
        "minimum_free_disk_gib": args.minimum_free_gib,
        "free_disk_gib_before": free_before,
        "jobs": args.jobs,
        "autoibamr_script_sha256": sha256(script),
        "libmesh_recipe_sha256": sha256(recipe),
        "libmesh_version": "1.7.8",
        "libmesh_checksum": "ceb05c868fe1270ede4f91a1f445313ef795ce6df3257e7d82cd6f2502753918",
        "existing_install_preserved": True,
        "hawkmoth_solver_run": False,
        "stdout_file": stdout_path.name,
        "stderr_file": stderr_path.name,
    }
    result_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    peak_free = free_before
    reason = None
    returncode = None
    with stdout_path.open("wb") as stdout, stderr_path.open("wb") as stderr:
        proc = subprocess.Popen(command, cwd=auto_root, stdout=stdout, stderr=stderr,
                                 start_new_session=True, env=os.environ.copy())
        while proc.poll() is None:
            time.sleep(10)
            free_now = disk_gib(record_dir)
            peak_free = min(peak_free, free_now)
            if free_now < args.minimum_free_gib:
                reason = f"disk reserve guard triggered at {free_now:.2f} GiB free"
                os.killpg(proc.pid, signal.SIGTERM)
                try:
                    proc.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    os.killpg(proc.pid, signal.SIGKILL)
                    proc.wait()
                break
        returncode = proc.wait()
    end = datetime.now(timezone.utc)
    metadata.update({
        "finished_utc": end.isoformat(),
        "elapsed_seconds": (end - start).total_seconds(),
        "free_disk_gib_after": disk_gib(record_dir),
        "minimum_observed_free_gib": peak_free,
        "returncode": returncode,
        "stop_reason": reason,
        "status": "STOPPED_DISK_GUARD" if reason else ("BUILD_SUCCEEDED" if returncode == 0 else "BUILD_FAILED"),
        "stdout_sha256": sha256(stdout_path),
        "stderr_sha256": sha256(stderr_path),
    })
    result_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    print(json.dumps(metadata, indent=2, sort_keys=True))
    return 0 if returncode == 0 and reason is None else 1


if __name__ == "__main__":
    raise SystemExit(main())
