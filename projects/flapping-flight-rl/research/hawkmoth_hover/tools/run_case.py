#!/usr/bin/env python3
"""Run a prepared IBAMR case with MPI, memory, and free-disk guardrails."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import time


def process_tree_rss_kb(root_pid: int) -> int:
    result = subprocess.run(
        ["ps", "-A", "-o", "pid=", "-o", "ppid=", "-o", "rss="],
        check=True,
        capture_output=True,
        text=True,
    )
    parents: dict[int, int] = {}
    rss: dict[int, int] = {}
    for line in result.stdout.splitlines():
        parts = line.split()
        if len(parts) != 3:
            continue
        pid, ppid, size = map(int, parts)
        parents[pid] = ppid
        rss[pid] = size
    members = {root_pid}
    updated = True
    while updated:
        updated = False
        for pid, ppid in parents.items():
            if ppid in members and pid not in members:
                members.add(pid)
                updated = True
    return sum(rss.get(pid, 0) for pid in members)


def directory_size(path: Path) -> int:
    total = 0
    for item in path.rglob("*"):
        if item.is_file():
            try:
                total += item.stat().st_size
            except FileNotFoundError:
                pass
    return total


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--executable", type=Path, required=True)
    parser.add_argument("--ranks", type=int, default=1)
    parser.add_argument("--max-rss-gb", type=float, default=8.0)
    parser.add_argument("--rss-stop-headroom-gb", type=float, default=0.5,
                        help="stop this far below the hard RSS ceiling to cover sampling overshoot")
    parser.add_argument("--min-free-disk-gb", type=float, default=40.0)
    parser.add_argument("--sample-seconds", type=float, default=0.2)
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    executable = args.executable.resolve()
    if not (run_dir / "input3d").is_file():
        parser.error(f"missing {run_dir / 'input3d'}; prepare the case first")
    if not executable.is_file() or not os.access(executable, os.X_OK):
        parser.error(f"solver executable is missing or not executable: {executable}")
    if not 1 <= args.ranks <= 8:
        parser.error("local run rank count must be between 1 and 8")
    if args.max_rss_gb <= 0 or args.rss_stop_headroom_gb < 0 or \
            args.rss_stop_headroom_gb >= args.max_rss_gb or args.sample_seconds < 0.2:
        parser.error("resource caps must be positive and sampling interval at least 0.2 s")
    rss_stop_threshold_gb = args.max_rss_gb - args.rss_stop_headroom_gb
    if shutil.disk_usage(run_dir).free < args.min_free_disk_gb * 1024**3:
        parser.error("free disk is already below the configured safety reserve")

    command = ["mpirun", "--oversubscribe", "-n", str(args.ranks), str(executable), "input3d"]
    repo_root = Path(__file__).resolve().parents[3]
    environment_file = Path.home() / "Applications" / "ibamr-research" / "install" / "share" / "hawkmoth-hover" / "build-environment.txt"
    identity = {
        "input3d_sha256": sha256_file(run_dir / "input3d"),
        "executable_sha256_at_launch": sha256_file(executable),
        "git_revision": subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo_root,
                                       check=True, capture_output=True, text=True).stdout.strip(),
        "git_worktree_dirty_at_launch": bool(subprocess.run(["git", "status", "--porcelain"], cwd=repo_root,
                                                            check=True, capture_output=True, text=True).stdout.strip()),
        "application_source_sha256": {
            str(path.relative_to(repo_root)): sha256_file(path)
            for path in (repo_root / "research" / "hawkmoth_hover" / "src" / "main.cpp",
                         repo_root / "research" / "hawkmoth_hover" / "CMakeLists.txt",
                         Path(__file__).resolve())
        },
        "build_environment_file": str(environment_file) if environment_file.is_file() else None,
        "build_environment_sha256": sha256_file(environment_file) if environment_file.is_file() else None,
    }
    started = time.monotonic()
    before_bytes = directory_size(run_dir)
    peak_rss = 0
    stop_reason = None
    with (run_dir / "solver.stdout.log").open("w") as stdout, (run_dir / "solver.stderr.log").open("w") as stderr:
        proc = subprocess.Popen(command, cwd=run_dir, stdout=stdout, stderr=stderr, start_new_session=True)
        while proc.poll() is None:
            rss = process_tree_rss_kb(proc.pid)
            peak_rss = max(peak_rss, rss)
            free = shutil.disk_usage(run_dir).free
            if rss >= rss_stop_threshold_gb * 1024**2:
                stop_reason = (f"sum of solver process-tree RSS reached the conservative stop threshold "
                               f"of {rss_stop_threshold_gb:.2f} GiB below the {args.max_rss_gb:.2f} GiB hard ceiling")
            elif free < args.min_free_disk_gb * 1024**3:
                stop_reason = f"free disk fell below {args.min_free_disk_gb:.1f} GiB reserve"
            if stop_reason:
                os.killpg(proc.pid, signal.SIGTERM)
                try:
                    proc.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    os.killpg(proc.pid, signal.SIGKILL)
                    proc.wait()
                break
            time.sleep(args.sample_seconds)
        exit_code = proc.returncode

    status = "STOPPED_RESOURCE_CAP" if stop_reason else ("COMPLETED" if exit_code == 0 else "FAILED")
    result = {
        "status": status,
        "stop_reason": stop_reason,
        "command": command,
        "cwd": str(run_dir),
        "exit_code": exit_code,
        "elapsed_seconds": time.monotonic() - started,
        "mpi_ranks": args.ranks,
        "resource_sample_interval_seconds": args.sample_seconds,
        "rss_hard_ceiling_gb": args.max_rss_gb,
        "rss_stop_threshold_gb": rss_stop_threshold_gb,
        "peak_process_tree_rss_gb": peak_rss / 1024**2,
        "disk_bytes_before": before_bytes,
        "disk_bytes_after": directory_size(run_dir),
        "free_disk_bytes_after": shutil.disk_usage(run_dir).free,
        "run_identity": identity,
    }
    (run_dir / "resource_record.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if status == "COMPLETED" else 1)


if __name__ == "__main__":
    main()
