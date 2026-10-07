#!/usr/bin/env python3
"""Run selected IBAMR attest regressions with memory and disk stop guards."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def free_gib(path: Path) -> float:
    return shutil.disk_usage(path).free / (1024 ** 3)


def process_tree_rss_kib(root_pid: int) -> int:
    result = subprocess.run(["ps", "-axo", "pid=,ppid=,rss="], check=True,
                            text=True, capture_output=True)
    parent: dict[int, int] = {}
    rss: dict[int, int] = {}
    for line in result.stdout.splitlines():
        fields = line.split()
        if len(fields) == 3:
            pid, ppid, kib = map(int, fields)
            parent[pid] = ppid
            rss[pid] = kib
    descendants = {root_pid}
    changed = True
    while changed:
        changed = False
        for pid, ppid in parent.items():
            if ppid in descendants and pid not in descendants:
                descendants.add(pid)
                changed = True
    return sum(rss.get(pid, 0) for pid in descendants)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-root", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--include-regex", default=r"IIM/flow_past_sphere\.dg")
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--timeout-s", type=int, default=600)
    parser.add_argument("--max-rss-gib", type=float, default=8.0)
    parser.add_argument("--min-free-disk-gib", type=float, default=40.0)
    args = parser.parse_args()

    build = args.build_root.resolve()
    source = args.source_root.resolve()
    output = args.output_dir.resolve()
    attest = source / "attest"
    test_dir = build / "tests"
    pattern = re.compile(args.include_regex)
    inputs = sorted(path for path in test_dir.rglob("*.input") if pattern.search(str(path.relative_to(test_dir))))
    if not inputs:
        parser.error(f"no regression inputs match {args.include_regex!r} under {test_dir}")
    files: list[Path] = []
    for input_path in inputs:
        output_path = input_path.with_suffix(".output")
        executable = input_path.with_name(input_path.name.split(".", 1)[0])
        if not output_path.is_file() or not executable.is_file():
            parser.error(f"missing regression output or executable for {input_path}")
        files.extend((input_path, output_path, executable))
    if not attest.is_file():
        parser.error(f"attest runner not found: {attest}")
    if output.exists() and any(output.iterdir()):
        parser.error(f"refusing to overwrite nonempty output directory: {output}")
    if free_gib(output.parent if output.parent.exists() else Path.home()) < args.min_free_disk_gib:
        parser.error("free-disk preflight is below the declared minimum")

    output.mkdir(parents=True, exist_ok=True)
    stdout_path = output / "attest_stdout.log"
    stderr_path = output / "attest_stderr.log"
    record_path = output / "regression_record.json"
    command = [sys.executable, str(attest), f"-j{args.jobs}", "--verbose",
               "--test-timeout", str(args.timeout_s), "-R", args.include_regex]
    start = datetime.now(timezone.utc)
    before = free_gib(output)
    record = {
        "study_id": output.name,
        "status": "RUNNING",
        "started_utc": start.isoformat(),
        "command": command,
        "working_directory": str(build),
        "include_regex": args.include_regex,
        "jobs": args.jobs,
        "max_mpi_ranks_in_selected_inputs": max(
            [int(m.group(1)) for path in inputs if (m := re.search(r"mpirun=(\d+)", path.name))] or [1]
        ),
        "timeout_per_case_seconds": args.timeout_s,
        "rss_stop_gib": args.max_rss_gib,
        "minimum_free_disk_gib": args.min_free_disk_gib,
        "free_disk_gib_before": before,
        "inputs": [{"path": str(path), "sha256": sha256(path)} for path in files],
        "attest_sha256": sha256(attest),
        "stderr_file": stderr_path.name,
        "stdout_file": stdout_path.name,
    }
    record_path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")

    peak_rss = 0
    minimum_free = before
    stop_reason = None
    with stdout_path.open("wb") as stdout, stderr_path.open("wb") as stderr:
        process = subprocess.Popen(command, cwd=build, stdout=stdout, stderr=stderr,
                                   start_new_session=True, env=os.environ.copy())
        while process.poll() is None:
            time.sleep(1)
            rss_now = process_tree_rss_kib(process.pid)
            peak_rss = max(peak_rss, rss_now)
            free_now = free_gib(output)
            minimum_free = min(minimum_free, free_now)
            if rss_now > args.max_rss_gib * (1024 ** 2):
                stop_reason = f"RSS stop guard: {rss_now / (1024 ** 2):.3f} GiB"
            elif free_now < args.min_free_disk_gib:
                stop_reason = f"disk stop guard: {free_now:.3f} GiB free"
            if stop_reason:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
                break
        returncode = process.wait()

    end = datetime.now(timezone.utc)
    record.update({
        "finished_utc": end.isoformat(),
        "elapsed_seconds": (end - start).total_seconds(),
        "peak_process_tree_rss_gib": peak_rss / (1024 ** 2),
        "minimum_observed_free_disk_gib": minimum_free,
        "free_disk_gib_after": free_gib(output),
        "returncode": returncode,
        "stop_reason": stop_reason,
        "stdout_sha256": sha256(stdout_path),
        "stderr_sha256": sha256(stderr_path),
        "status": "STOPPED_RESOURCE_GUARD" if stop_reason else ("PASSED" if returncode == 0 else "FAILED"),
    })
    record_path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    print(json.dumps(record, indent=2, sort_keys=True))
    return 0 if returncode == 0 and stop_reason is None else 1


if __name__ == "__main__":
    raise SystemExit(main())
