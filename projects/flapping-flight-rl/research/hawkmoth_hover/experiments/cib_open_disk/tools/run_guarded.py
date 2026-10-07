#!/usr/bin/env python3
"""Run a recorded CIB pilot case under the project resource guardrails."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import platform
import re
import shutil
import signal
import subprocess
import time
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def free_gib(path: Path) -> float:
    return shutil.disk_usage(path).free / 1024**3


def tree_rss_kib(pid: int) -> int:
    result = subprocess.run(["ps", "-axo", "pid=,ppid=,rss="], text=True, capture_output=True, check=True)
    parents: dict[int, int] = {}
    rss: dict[int, int] = {}
    for line in result.stdout.splitlines():
        fields = line.split()
        if len(fields) == 3:
            child, parent, size = map(int, fields)
            parents[child] = parent
            rss[child] = size
    tree = {pid}
    updated = True
    while updated:
        updated = False
        for child, parent in parents.items():
            if parent in tree and child not in tree:
                tree.add(child)
                updated = True
    return sum(rss.get(item, 0) for item in tree)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--case-dir", type=Path, required=True)
    parser.add_argument(
        "--petsc-options",
        type=Path,
        help="optional explicit PETSc options file; copied into the run as petsc_options.dat",
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--mpi-ranks", type=int, default=1)
    parser.add_argument("--rss-stop-gib", type=float, default=7.5)
    parser.add_argument("--disk-reserve-gib", type=float, default=40.0)
    args = parser.parse_args()

    binary = args.binary.resolve()
    input_file = args.input.resolve()
    case_dir = args.case_dir.resolve()
    run_dir = args.run_dir.resolve()
    if not binary.is_file() or not input_file.is_file():
        parser.error("binary and input must exist")
    if not case_dir.is_dir() or not run_dir.is_dir() or any(run_dir.iterdir()):
        parser.error("case directory must exist and run directory must be empty")
    if not 1 <= args.mpi_ranks <= 8:
        parser.error("MPI ranks must be between one and eight")
    if args.rss_stop_gib >= 8.0:
        parser.error("use a conservative stop below the 8 GiB hard ceiling")
    if free_gib(run_dir) < args.disk_reserve_gib:
        parser.error("pre-run free-disk reserve is below the required threshold")

    local_binary = run_dir / "cib_open_disk"
    local_input = run_dir / "cib_open_disk.input"
    shutil.copy2(binary, local_binary)
    shutil.copy2(input_file, local_input)
    copied_case_files: dict[str, str] = {}
    for name in ("disk_3d.vertex",):
        source = case_dir / name
        if not source.is_file():
            parser.error(f"missing required case file: {source}")
        destination = run_dir / name
        shutil.copy2(source, destination)
        copied_case_files[name] = sha256(destination)
    petsc_source = (args.petsc_options or (case_dir / "petsc_options.dat")).resolve()
    if not petsc_source.is_file():
        parser.error(f"missing PETSc options file: {petsc_source}")
    petsc_destination = run_dir / "petsc_options.dat"
    shutil.copy2(petsc_source, petsc_destination)
    copied_case_files["petsc_options.dat"] = sha256(petsc_destination)

    command = [str(local_binary), str(local_input)]
    if args.mpi_ranks > 1:
        command = ["mpirun", "-n", str(args.mpi_ranks), *command]
    stdout_path = run_dir / "stdout.log"
    stderr_path = run_dir / "stderr.log"
    record_path = run_dir / "run_record.json"
    started = dt.datetime.now(dt.timezone.utc)
    free_before = free_gib(run_dir)
    record = {
        "study": "cib_open_disk_method_pilot",
        "case": run_dir.name,
        "status": "RUNNING",
        "started_utc": started.isoformat(),
        "command": command,
        "working_directory": str(run_dir),
        "mpi_ranks": args.mpi_ranks,
        "rss_stop_gib": args.rss_stop_gib,
        "disk_reserve_gib": args.disk_reserve_gib,
        "free_disk_gib_before": free_before,
        "binary_sha256": sha256(local_binary),
        "input_sha256": sha256(local_input),
        "case_file_sha256": copied_case_files,
        "host_platform": platform.platform(),
        "python_version": platform.python_version(),
        "stdout_file": stdout_path.name,
        "stderr_file": stderr_path.name,
    }
    record_path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    peak_rss = 0
    minimum_free = free_before
    stop_reason = None
    with stdout_path.open("wb") as stdout, stderr_path.open("wb") as stderr:
        process = subprocess.Popen(command, cwd=run_dir, stdout=stdout, stderr=stderr,
                                   start_new_session=True, env=os.environ.copy())
        while process.poll() is None:
            time.sleep(1.0)
            rss_now = tree_rss_kib(process.pid)
            free_now = free_gib(run_dir)
            peak_rss = max(peak_rss, rss_now)
            minimum_free = min(minimum_free, free_now)
            if rss_now > args.rss_stop_gib * 1024**2:
                stop_reason = f"conservative RSS stop reached ({rss_now / 1024**2:.3f} GiB)"
            elif free_now < args.disk_reserve_gib:
                stop_reason = f"disk reserve breached ({free_now:.3f} GiB free)"
            if stop_reason:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
                break
        return_code = process.wait()

    solver_log_path = run_dir / "CIB3d.log"
    solver_text = solver_log_path.read_text(encoding="utf-8", errors="replace") if solver_log_path.exists() else ""
    solver_diverged = re.search(r"CIBStaggeredStokesSolver:\s*diverged:", solver_text) is not None
    solver_converged = re.search(r"CIBStaggeredStokesSolver:\s*converged:", solver_text) is not None
    if solver_diverged:
        solver_outcome = "DIVERGED"
    elif solver_converged:
        solver_outcome = "CONVERGED"
    else:
        solver_outcome = "UNKNOWN_OR_MISSING_LOG"

    ended = dt.datetime.now(dt.timezone.utc)
    if stop_reason:
        final_status = "STOPPED_RESOURCE_GUARD"
    elif return_code != 0:
        final_status = "FAILED_RUNTIME"
    elif solver_outcome == "DIVERGED":
        final_status = "FAILED_SOLVER_NONCONVERGENCE"
    elif solver_outcome != "CONVERGED":
        final_status = "FAILED_SOLVER_STATUS_UNKNOWN"
    else:
        final_status = "COMPLETED"
    record.update({
        "finished_utc": ended.isoformat(),
        "elapsed_seconds": (ended - started).total_seconds(),
        "peak_process_tree_rss_gib": peak_rss / 1024**2,
        "minimum_observed_free_disk_gib": minimum_free,
        "free_disk_gib_after": free_gib(run_dir),
        "return_code": return_code,
        "stop_reason": stop_reason,
        "solver_log_file": solver_log_path.name,
        "solver_log_sha256": sha256(solver_log_path) if solver_log_path.exists() else None,
        "solver_outcome": solver_outcome,
        "stdout_sha256": sha256(stdout_path),
        "stderr_sha256": sha256(stderr_path),
        "status": final_status,
    })
    record_path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (run_dir / "resource_record.json").write_text(json.dumps({
        "status": record["status"],
        "solver_outcome": solver_outcome,
        "elapsed_seconds": record["elapsed_seconds"],
        "peak_process_tree_rss_gib": record["peak_process_tree_rss_gib"],
        "minimum_observed_free_disk_gib": minimum_free,
        "free_disk_gib_before": free_before,
        "free_disk_gib_after": record["free_disk_gib_after"],
        "mpi_ranks": args.mpi_ranks,
        "stop_reason": stop_reason,
    }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (run_dir / "run_manifest.json").write_text(json.dumps({
        "study": "cib_open_disk_method_pilot",
        "case": run_dir.name,
        "interpretation": "method feasibility diagnostic only; not Stage A validation",
        "ibamr_version": "0.19.0",
        "solver_outcome": solver_outcome,
        "run_record": record_path.name,
    }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(record, indent=2, sort_keys=True))
    return 0 if final_status == "COMPLETED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
