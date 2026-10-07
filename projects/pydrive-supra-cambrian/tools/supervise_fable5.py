#!/usr/bin/env python3
"""Durable watchdog for unattended Fable Five training.

Owns a child trainer, a matching caffeinate assertion, a persistent stdout log,
checkpoint-freshness monitoring, disk preflight, and bounded crash restarts.
The exact child command follows ``--``.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
from pathlib import Path
import queue
import shutil
import signal
import subprocess
import sys
import threading
import time


ROOT = Path(__file__).resolve().parent.parent


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    with tmp.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
        fh.write("\n")
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


def checkpoint_files(name: str) -> list[Path]:
    out = list(ROOT.glob(f"{name}_*.pt"))
    direct = ROOT / f"{name}.pt"
    if direct.exists():
        out.append(direct)
    return [p for p in out if ".tmp." not in p.name]


def newest_checkpoint(name: str) -> Path | None:
    files = checkpoint_files(name)
    return max(files, key=lambda p: p.stat().st_mtime) if files else None


def free_gb() -> float:
    return shutil.disk_usage(ROOT).free / (1024 ** 3)


def signal_group(proc: subprocess.Popen | None, sig: int) -> None:
    """Signal the trainer and every multiprocessing worker it spawned."""
    if proc is None or proc.poll() is not None:
        return
    try:
        os.killpg(proc.pid, sig)
    except (ProcessLookupError, PermissionError, OSError):
        try:
            proc.send_signal(sig)
        except ProcessLookupError:
            pass


def signal_parent(proc: subprocess.Popen | None, sig: int) -> None:
    """Give PPO alone the first chance to checkpoint before worker cleanup."""
    if proc is None or proc.poll() is not None:
        return
    try:
        proc.send_signal(sig)
    except ProcessLookupError:
        pass


def parse_args(argv: list[str]) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--name", required=True, help="run prefix, without .pt")
    ap.add_argument("--max-restarts", type=int, default=12)
    ap.add_argument("--stall-minutes", type=float, default=30.0)
    ap.add_argument("--min-free-gb", type=float, default=10.0)
    ap.add_argument("--stop-grace-seconds", type=float, default=20.0)
    ap.add_argument("--logpath-file", default=str(ROOT / "fable5_supervisor.logpath"))
    ap.add_argument("--rerun-completed", action="store_true",
                    help="run even when the same command already completed")
    ap.add_argument("command", nargs=argparse.REMAINDER)
    args = ap.parse_args(argv)
    if args.command and args.command[0] == "--":
        args.command = args.command[1:]
    if not args.command:
        ap.error("pass the trainer command after --")
    args.name = Path(args.name).name.removesuffix(".pt")
    return args


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    run_dir = ROOT / "runtime" / "fable5" / args.name
    run_dir.mkdir(parents=True, exist_ok=True)
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = run_dir / f"supervisor_{stamp}.log"
    state_path = run_dir / "supervisor_state.json"
    try:
        prior_state = json.loads(state_path.read_text())
    except Exception:
        prior_state = {}
    if (not args.rerun_completed and prior_state.get("status") == "complete"
            and prior_state.get("command") == args.command):
        print(f"Fable run {args.name!r} already completed this exact command; "
              "use --rerun-completed or a new --name to run it again", flush=True)
        return 0
    logpath_file = Path(args.logpath_file)
    if not logpath_file.is_absolute():
        logpath_file = ROOT / logpath_file
    logpath_file.parent.mkdir(parents=True, exist_ok=True)
    logpath_file.write_text(str(log_path) + "\n")

    stop_requested = False
    stop_deadline: float | None = None
    child: subprocess.Popen | None = None

    def on_signal(signum, frame):
        nonlocal stop_requested, stop_deadline
        stop_requested = True
        stop_deadline = time.time() + max(0.1, args.stop_grace_seconds)
        signal_parent(child, signal.SIGINT)

    for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        signal.signal(sig, on_signal)

    restarts = 0
    disk_block_reported = False
    with log_path.open("a", encoding="utf-8", buffering=1) as durable:
        console_open = True

        def console_write(message: str) -> None:
            """A closed dashboard pipe must never take down the watchdog."""
            nonlocal console_open
            if not console_open:
                return
            try:
                sys.stdout.write(message)
                sys.stdout.flush()
            except (BrokenPipeError, OSError):
                console_open = False

        def emit(message: str) -> None:
            line = f"[{dt.datetime.now().isoformat(timespec='seconds')}] {message}"
            console_write(line + "\n")
            durable.write(line + "\n")

        while not stop_requested:
            free = free_gb()
            if free < args.min_free_gb:
                if not disk_block_reported:
                    emit(f"PAUSED low disk: {free:.1f} GiB free < "
                         f"{args.min_free_gb:.1f}; rechecking automatically")
                    disk_block_reported = True
                atomic_json(state_path, {"status": "blocked_low_disk", "free_gb": free,
                                         "restarts": restarts, "command": args.command})
                deadline = time.time() + 300.0
                while time.time() < deadline and not stop_requested:
                    time.sleep(min(1.0, deadline - time.time()))
                continue
            if disk_block_reported:
                emit(f"disk recovered: {free:.1f} GiB free; resuming")
                disk_block_reported = False

            emit(f"launch attempt {restarts + 1}: {' '.join(args.command)}")
            env = os.environ.copy()
            env["PYTHONUNBUFFERED"] = "1"
            try:
                child = subprocess.Popen(args.command, cwd=ROOT, env=env,
                                         stdout=subprocess.PIPE,
                                         stderr=subprocess.STDOUT, text=True,
                                         errors="replace", bufsize=1,
                                         start_new_session=True)
                caffeinate = subprocess.Popen(
                    ["/usr/bin/caffeinate", "-dimsu", "-w", str(child.pid)],
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            except (OSError, ValueError) as exc:
                if child is not None and child.poll() is None:
                    child.terminate()
                emit(f"FATAL launch failed: {type(exc).__name__}: {exc}")
                atomic_json(state_path, {"status": "launch_failed",
                                         "error": str(exc),
                                         "restarts": restarts,
                                         "command": args.command})
                # Intentional terminal state: a LaunchAgent should not spin on
                # a malformed command. The state file carries the failure.
                return 0
            q: queue.Queue[str | None] = queue.Queue()

            def reader():
                try:
                    assert child is not None and child.stdout is not None
                    for line in child.stdout:
                        q.put(line)
                finally:
                    q.put(None)

            threading.Thread(target=reader, daemon=True).start()
            launch_time = time.time()
            last_state = 0.0
            saw_completion = False
            stalled = False
            disk_pause = False
            reader_done = False
            child_exit_seen: float | None = None

            while child.poll() is None or not reader_done:
                try:
                    item = q.get(timeout=1.0)
                    if item is None:
                        reader_done = True
                    else:
                        console_write(item)
                        durable.write(item)
                        if "AUTO ladder done" in item or "Done. Watch it" in item:
                            saw_completion = True
                except queue.Empty:
                    pass

                now = time.time()
                if child.poll() is not None and child_exit_seen is None:
                    child_exit_seen = now
                if (child_exit_seen is not None and not reader_done
                        and now - child_exit_seen > 2.0):
                    # A crashed parent can leave workers holding the inherited
                    # stdout pipe open. Reap the whole session so restart cannot
                    # hang forever waiting for EOF.
                    try:
                        os.killpg(child.pid, signal.SIGKILL)
                    except (ProcessLookupError, PermissionError, OSError):
                        pass

                if (stop_requested and child.poll() is None
                        and stop_deadline is not None and now >= stop_deadline):
                    emit("stop grace expired; forcing trainer process-group cleanup")
                    signal_group(child, signal.SIGKILL)

                cp = newest_checkpoint(args.name)
                # Old checkpoints predate this launch. Give the child a full
                # startup/eval window instead of declaring a stall immediately.
                heartbeat = max(launch_time, cp.stat().st_mtime) if cp else launch_time
                age = now - heartbeat
                if (not stop_requested
                        and age > max(120.0, args.stall_minutes * 60.0)):
                    stalled = True
                    emit(f"STALL: checkpoint heartbeat unchanged for {age/60:.1f} min; "
                         "requesting clean checkpoint/restart")
                    signal_parent(child, signal.SIGINT)
                    deadline = time.time() + max(0.1, args.stop_grace_seconds)
                    while child.poll() is None and time.time() < deadline:
                        time.sleep(0.25)
                    if child.poll() is None:
                        signal_group(child, signal.SIGKILL)

                free_now = free_gb()
                if (not stop_requested and not disk_pause
                        and free_now < args.min_free_gb):
                    disk_pause = True
                    emit(f"PAUSED low disk during training: {free_now:.1f} GiB "
                         f"free < {args.min_free_gb:.1f}; checkpointing")
                    signal_parent(child, signal.SIGINT)
                    deadline = time.time() + max(0.1, args.stop_grace_seconds)
                    while child.poll() is None and time.time() < deadline:
                        time.sleep(0.25)
                    if child.poll() is None:
                        signal_group(child, signal.SIGKILL)

                if now - last_state >= 15.0:
                    atomic_json(state_path, {
                        "status": "stopping" if stop_requested else "running",
                        "supervisor_pid": os.getpid(), "trainer_pid": child.pid,
                        "run": args.name, "command": args.command,
                        "restarts": restarts, "started_unix": launch_time,
                        "checkpoint": str(cp) if cp else None,
                        "checkpoint_age_seconds": age, "free_gb": free_gb(),
                        "log": str(log_path),
                    })
                    last_state = now

            code = child.wait()
            try:
                caffeinate.wait(timeout=3.0)
            except subprocess.TimeoutExpired:
                caffeinate.terminate()
            emit(f"trainer exit={code}, completion={saw_completion}, stalled={stalled}")

            if stop_requested:
                atomic_json(state_path, {"status": "stopped", "exit_code": code,
                                         "restarts": restarts, "log": str(log_path)})
                return 0
            if disk_pause:
                atomic_json(state_path, {"status": "blocked_low_disk",
                                         "free_gb": free_gb(),
                                         "restarts": restarts,
                                         "log": str(log_path)})
                continue
            if code == 0 and saw_completion:
                atomic_json(state_path, {"status": "complete", "exit_code": code,
                                         "restarts": restarts, "log": str(log_path),
                                         "command": args.command})
                return 0
            if code == 73:
                emit("FATAL duplicate Fable owner detected; not retrying")
                atomic_json(state_path, {"status": "blocked_duplicate",
                                         "exit_code": code, "restarts": restarts,
                                         "log": str(log_path)})
                return 0

            restarts += 1
            if restarts > args.max_restarts:
                emit(f"FATAL restart budget exhausted ({args.max_restarts})")
                atomic_json(state_path, {"status": "restart_budget_exhausted",
                                         "exit_code": code, "restarts": restarts,
                                         "log": str(log_path)})
                return 0
            delay = min(300.0, 5.0 * (2 ** min(restarts - 1, 6)))
            emit(f"unexpected stop; crash-resume in {delay:.0f}s "
                 f"({restarts}/{args.max_restarts})")
            atomic_json(state_path, {"status": "backoff", "exit_code": code,
                                     "restarts": restarts,
                                     "retry_at_unix": time.time() + delay,
                                     "supervisor_pid": os.getpid(),
                                     "log": str(log_path)})
            deadline = time.time() + delay
            while time.time() < deadline and not stop_requested:
                time.sleep(min(1.0, deadline - time.time()))

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
