#!/usr/bin/env python3
"""Retire the mis-stamped Porsche 919 Fable lineage without losing evidence.

The archive is deliberately self-contained: it includes every checkpoint and
run artifact, the exact working-tree source snapshot, a human-readable diff,
the launch command, dependency inventories, and SHA-256 checksums.  It never
modifies a checkpoint.  With ``--remove-stale-lock`` it removes the legacy
training lock only after proving the recorded PID is not alive and after the
archive has been completed successfully.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import tarfile
import time
from typing import Any, Iterable


SCHEMA = "legacy-919-retirement-v1"
LABEL = "legacy_misstamped_noncertifiable"
PREFIX = "fable5_919_ring"
FINGERPRINT_SOURCES = (
    "supra/fable5.py", "supra/ppo.py", "supra/ppo_env.py", "supra/config.py",
    "supra/physics.py", "supra/sensors.py", "supra/track.py",
    "supra/data/tracks/nordschleife.json",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _run(root: Path, *command: str, check: bool = True) -> str:
    result = subprocess.run(
        command,
        cwd=root,
        check=check,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    return result.stdout


def _pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _read_lock(lock_path: Path) -> dict[str, Any]:
    if not lock_path.exists():
        return {}
    try:
        value = json.loads(lock_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"refusing to archive: unreadable lock {lock_path}: {exc}")
    if not isinstance(value, dict):
        raise SystemExit(f"refusing to archive: lock is not a JSON object: {lock_path}")
    return value


def _read_locked_stream(stream, lock_path: Path) -> dict[str, Any]:
    stream.seek(0)
    raw = stream.read().strip()
    if not raw:
        return {}
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise SystemExit(f"refusing to archive: unreadable lock {lock_path}: {exc}")
    if not isinstance(value, dict):
        raise SystemExit(f"refusing to archive: lock is not a JSON object: {lock_path}")
    return value


def _source_fingerprint(root: Path) -> str:
    digest = hashlib.sha256()
    for name in FINGERPRINT_SOURCES:
        path = root / name
        if not path.is_file():
            raise SystemExit(f"missing checkpoint-semantic source: {path}")
        digest.update(name.encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _load_checkpoint_metadata(path: Path) -> dict[str, Any]:
    # Dashboard/archive discovery must not execute pickle payloads before a
    # checkpoint can be identified. The constrained loader supports the inert
    # NumPy containers in historical project files while keeping PyTorch's
    # weights-only unpickler enabled.
    project_root = path.resolve().parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    try:
        from supra.checkpoint_io import load_torch_checkpoint_safe
        payload = load_torch_checkpoint_safe(path)
    except (ImportError, ValueError, RuntimeError) as exc:
        raise SystemExit(f"could not safely verify legacy checkpoint {path}: {exc}") from exc
    car = payload.get("car")
    if car != "porsche_919evo":
        raise SystemExit(
            f"refusing to label archive as the 919 lineage: checkpoint car={car!r}"
        )
    return {
        "car": car,
        "stage": payload.get("fable_stage"),
        "updates": payload.get("updates"),
        "checkpoint_time_unix": payload.get("checkpoint_time_unix"),
        "legacy_drivetrain_stamp": payload.get("fable_drivetrain_version"),
        "legacy_code_fingerprint": payload.get("fable_code_fingerprint"),
        "legacy_envelope_scale": payload.get("fable_envelope_scale"),
        "metric": payload.get("metric"),
        "observation_layout": payload.get("obs_layout"),
        "observation_schema": payload.get("norm_schema"),
    }


def _safe_copy(path: Path, root: Path, destination: Path) -> None:
    relative = path.resolve().relative_to(root.resolve())
    target = destination / "run_artifacts" / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(path, target)


def _source_paths(root: Path) -> list[Path]:
    raw = subprocess.run(
        ["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard"],
        cwd=root,
        check=True,
        stdout=subprocess.PIPE,
    ).stdout
    paths: list[Path] = []
    for name in raw.split(b"\0"):
        if not name:
            continue
        path = root / os.fsdecode(name)
        if path.is_file() or path.is_symlink():
            paths.append(path)
    return sorted(paths)


def _write_source_snapshot(root: Path, destination: Path) -> int:
    paths = _source_paths(root)
    archive = destination / "source" / "working_tree.tar.gz"
    archive.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, "w:gz", format=tarfile.PAX_FORMAT) as tar:
        for path in paths:
            tar.add(path, arcname=path.relative_to(root), recursive=False)
    return len(paths)


def _copy_run_evidence(root: Path, destination: Path) -> list[str]:
    candidates: set[Path] = set(root.glob(f"{PREFIX}*.pt"))
    for name in (
        ".fable5_training.lock",
        "fable5_ring_pipeline.json",
        "fable5_ring_eval_latest.json",
        "fable5_pit_log.txt",
        "fable5_supervisor.logpath",
    ):
        path = root / name
        if path.is_file():
            candidates.add(path)

    runtime_run = root / "runtime" / "fable5" / PREFIX
    runtime_events = root / "runtime" / "fable5" / f"{PREFIX}_foundation"
    for directory in (runtime_run, runtime_events):
        if directory.is_dir():
            candidates.update(path for path in directory.rglob("*") if path.is_file())

    log_pointer = root / "fable5_supervisor.logpath"
    if log_pointer.is_file():
        pointed = Path(log_pointer.read_text(encoding="utf-8").strip())
        if pointed.is_file() and root.resolve() in pointed.resolve().parents:
            candidates.add(pointed)

    for path in sorted(candidates):
        _safe_copy(path, root, destination)
    return [str(path.relative_to(root)) for path in sorted(candidates)]


def _write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value, encoding="utf-8")


def _inventory(destination: Path) -> Iterable[Path]:
    for path in sorted(destination.rglob("*")):
        if path.is_file() and path.name != "SHA256SUMS":
            yield path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".", help="project root")
    parser.add_argument(
        "--out",
        default="runtime/fable5/archives",
        help="archive parent directory (relative paths are resolved under root)",
    )
    parser.add_argument(
        "--remove-stale-lock",
        action="store_true",
        help="remove .fable5_training.lock after a successful archive",
    )
    args = parser.parse_args()

    root = Path(args.root).resolve()
    out_parent = Path(args.out)
    if not out_parent.is_absolute():
        out_parent = root / out_parent

    final_checkpoint = root / f"{PREFIX}_foundation.pt"
    if not final_checkpoint.is_file():
        raise SystemExit(f"missing final checkpoint: {final_checkpoint}")

    lock_path = root / ".fable5_training.lock"
    # Hold the same advisory lock used by run.py across the complete snapshot
    # and optional unlink. This closes the start/archive/remove TOCTOU window.
    lock_handle = lock_path.open("a+", encoding="utf-8")
    try:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        lock_handle.close()
        raise SystemExit("refusing to archive: Fable training lock is live") from exc
    lock = _read_locked_stream(lock_handle, lock_path)
    recorded_pid = int(lock.get("pid", 0) or 0)
    if _pid_alive(recorded_pid):
        raise SystemExit(
            f"refusing to archive or remove lock: recorded PID {recorded_pid} is alive"
        )

    checkpoint_metadata = _load_checkpoint_metadata(final_checkpoint)
    source_fingerprint = _source_fingerprint(root)
    checkpoint_fingerprint = checkpoint_metadata.get("legacy_code_fingerprint")
    source_matches_checkpoint = source_fingerprint == checkpoint_fingerprint
    stamp = time.strftime("%Y%m%d-%H%M%S")
    destination = out_parent / f"{LABEL}-{stamp}"
    destination.mkdir(parents=True, exist_ok=False)

    evidence_files = _copy_run_evidence(root, destination)
    source_count = _write_source_snapshot(root, destination)
    _write_text(destination / "source" / "git_status.txt", _run(root, "git", "status", "--porcelain=v2", "--branch"))
    _write_text(destination / "source" / "working_tree.diff", _run(root, "git", "diff", "--no-ext-diff", "--submodule=diff", "HEAD"))
    _write_text(destination / "source" / "untracked_files.txt", _run(root, "git", "ls-files", "--others", "--exclude-standard"))
    _write_text(destination / "dependencies" / "pip_freeze.txt", _run(root, sys.executable, "-m", "pip", "freeze"))
    _write_text(destination / "dependencies" / "python_version.txt", sys.version + "\n")
    _write_text(
        destination / "dependencies" / "platform.json",
        json.dumps(
            {
                "platform": platform.platform(),
                "machine": platform.machine(),
                "python_executable": sys.executable,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
    )
    for name in ("requirements.txt", "package.json", "package-lock.json"):
        path = root / name
        if path.is_file():
            target = destination / "dependencies" / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)

    head = _run(root, "git", "rev-parse", "HEAD").strip()
    branch = _run(root, "git", "branch", "--show-current").strip()
    manifest = {
        "schema": SCHEMA,
        "archive_id": destination.name,
        "label": LABEL,
        "created_unix": time.time(),
        "certifiable": False,
        "resume_into_faithful_v2_permitted": False,
        "promotion_permitted": False,
        "legacy_evaluation_import_permitted": False,
        "legacy_hof_import_permitted": False,
        "legacy_sector_heat_import_permitted": False,
        "legacy_optimizer_import_permitted": False,
        "reasons": [
            "Porsche checkpoint carries a Mazda drivetrain identity stamp",
            "legacy observation, action, normalization, optimizer and evaluation schemas are not faithful-v2",
            "licensed tyre, aero, suspension, control, telemetry and 2018 survey evidence was not frozen",
            "legacy training reference was not an authoritative optimal-control feasibility proof",
        ],
        "checkpoint": checkpoint_metadata,
        "source_snapshot_code_fingerprint": source_fingerprint,
        "source_snapshot_matches_checkpoint": source_matches_checkpoint,
        "historical_replay_source_reproducible": source_matches_checkpoint,
        "launch_lock": lock,
        "recorded_pid_alive_at_archive": False,
        "git": {"head": head, "branch": branch, "dirty": True},
        "source_snapshot_file_count": source_count,
        "evidence_files": evidence_files,
    }
    manifest_path = destination / "ARCHIVE_MANIFEST.json"
    _write_text(manifest_path, json.dumps(manifest, indent=2, sort_keys=True) + "\n")

    checksum_lines = []
    for path in _inventory(destination):
        checksum_lines.append(f"{_sha256(path)}  {path.relative_to(destination)}")
    _write_text(destination / "SHA256SUMS", "\n".join(checksum_lines) + "\n")

    if args.remove_stale_lock and lock_path.exists():
        # Re-check the still-locked inode before mutation; non-cooperating
        # writers are detected even though compliant trainers cannot enter.
        latest_lock = _read_locked_stream(lock_handle, lock_path)
        latest_pid = int(latest_lock.get("pid", 0) or 0)
        if latest_pid != recorded_pid or _pid_alive(latest_pid):
            raise SystemExit(
                "archive completed, but lock changed or its PID became live; lock was preserved"
            )
        lock_path.unlink()

    fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)
    lock_handle.close()
    print(destination)


if __name__ == "__main__":
    main()
