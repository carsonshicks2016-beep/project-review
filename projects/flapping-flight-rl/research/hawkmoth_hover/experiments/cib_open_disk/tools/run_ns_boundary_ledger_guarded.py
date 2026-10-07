#!/usr/bin/env python3
"""Run one analytic Navier-Stokes boundary-ledger control under local limits."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import signal
import subprocess
import time
from datetime import datetime, timezone


HERE = Path(__file__).resolve()
REPO_ROOT = HERE.parents[5]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def process_tree_rss_kib(root_pid: int) -> int:
    result = subprocess.run(["ps", "-A", "-o", "pid=,ppid=,rss="], check=True, capture_output=True, text=True)
    parents: dict[int, int] = {}
    rss: dict[int, int] = {}
    for line in result.stdout.splitlines():
        fields = line.split()
        if len(fields) != 3:
            continue
        pid, ppid, size = map(int, fields)
        parents[pid] = ppid
        rss[pid] = size
    members = {root_pid}
    changed = True
    while changed:
        changed = False
        for pid, ppid in parents.items():
            if ppid in members and pid not in members:
                members.add(pid)
                changed = True
    return sum(rss.get(pid, 0) for pid in members)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--executable", type=Path, required=True)
    parser.add_argument("--ibamr-prefix", type=Path, required=True)
    parser.add_argument("--ranks", type=int, default=1)
    parser.add_argument("--rss-stop-gib", type=float, default=7.5)
    parser.add_argument("--min-free-disk-gib", type=float, default=40.0)
    parser.add_argument("--sample-seconds", type=float, default=0.2)
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()
    executable = args.executable.resolve()
    prefix = args.ibamr_prefix.resolve()
    input_path = run_dir / "input3d"
    if not input_path.is_file():
        parser.error(f"missing input deck: {input_path}")
    if not executable.is_file() or not os.access(executable, os.X_OK):
        parser.error(f"executable is missing or not runnable: {executable}")
    if not 1 <= args.ranks <= 8:
        parser.error("local MPI rank count must be from one to eight")
    if args.rss_stop_gib <= 0.0 or args.min_free_disk_gib <= 0.0 or args.sample_seconds < 0.2:
        parser.error("resource limits must be positive and sampling interval at least 0.2 seconds")
    if shutil.disk_usage(run_dir).free < args.min_free_disk_gib * 1024**3:
        parser.error("free disk is already below the required reserve")
    for name in ("solver.stdout.log", "solver.stderr.log", "resource_record.json"):
        if (run_dir / name).exists():
            parser.error(f"refusing to overwrite existing run artifact {name}")

    source_files = [
        HERE,
        HERE.with_name("prepare_ns_boundary_ledger.py"),
        HERE.parents[1] / "src/ns_boundary_ledger.cpp",
        HERE.parents[1] / "src/outer_momentum_flux.hpp",
        HERE.parents[1] / "CMakeLists.txt",
        input_path,
    ]
    build_cache = executable.parent / "CMakeCache.txt"
    identity = {
        "git_revision": subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, check=True,
                                        capture_output=True, text=True).stdout.strip(),
        "git_worktree_dirty_at_launch": bool(subprocess.run(["git", "status", "--porcelain"], cwd=REPO_ROOT,
                                                              check=True, capture_output=True, text=True).stdout.strip()),
        "source_and_input_sha256": {str(path.relative_to(REPO_ROOT)): sha256(path) for path in source_files},
        "executable_path": str(executable),
        "executable_sha256": sha256(executable),
        "build_cache_path": str(build_cache) if build_cache.is_file() else None,
        "build_cache_sha256": sha256(build_cache) if build_cache.is_file() else None,
        "ibamr_prefix": str(prefix),
        "platform": platform.platform(),
        "mpi_version": subprocess.run(["mpirun", "--version"], check=True, capture_output=True,
                                       text=True).stdout.splitlines()[0],
    }
    command = ["mpirun", "--oversubscribe", "-n", str(args.ranks), str(executable), "input3d"]
    started_utc = datetime.now(timezone.utc).isoformat()
    start = time.monotonic()
    disk_before = shutil.disk_usage(run_dir).free
    run_bytes_before = sum(path.stat().st_size for path in run_dir.rglob("*") if path.is_file())
    peak_rss_kib = 0
    stop_reason = None
    with (run_dir / "solver.stdout.log").open("w") as stdout, (run_dir / "solver.stderr.log").open("w") as stderr:
        process = subprocess.Popen(command, cwd=run_dir, stdout=stdout, stderr=stderr, start_new_session=True)
        while process.poll() is None:
            peak_rss_kib = max(peak_rss_kib, process_tree_rss_kib(process.pid))
            if peak_rss_kib >= args.rss_stop_gib * 1024**2:
                stop_reason = f"process-tree RSS reached conservative stop threshold {args.rss_stop_gib:.2f} GiB"
            elif shutil.disk_usage(run_dir).free < args.min_free_disk_gib * 1024**3:
                stop_reason = f"free disk fell below {args.min_free_disk_gib:.1f} GiB reserve"
            if stop_reason:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
                break
            time.sleep(args.sample_seconds)
        exit_code = process.returncode

    status = "STOPPED_RESOURCE_CAP" if stop_reason else ("COMPLETED" if exit_code == 0 else "FAILED")
    record = {
        "status": status,
        "stop_reason": stop_reason,
        "started_utc": started_utc,
        "elapsed_seconds": time.monotonic() - start,
        "command": command,
        "working_directory": str(run_dir),
        "exit_code": exit_code,
        "mpi_ranks": args.ranks,
        "rss_stop_threshold_gib": args.rss_stop_gib,
        "sample_interval_seconds": args.sample_seconds,
        "peak_process_tree_rss_gib_sampled": peak_rss_kib / 1024**2,
        "free_disk_bytes_before": disk_before,
        "free_disk_bytes_after": shutil.disk_usage(run_dir).free,
        "run_artifact_bytes_before": run_bytes_before,
        "run_artifact_bytes_after": sum(path.stat().st_size for path in run_dir.rglob("*") if path.is_file()),
        "identity": identity,
    }
    (run_dir / "resource_record.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record, indent=2))
    raise SystemExit(0 if status == "COMPLETED" else 1)


if __name__ == "__main__":
    main()
