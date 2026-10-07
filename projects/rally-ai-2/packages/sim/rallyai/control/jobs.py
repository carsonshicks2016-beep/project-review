"""Server-owned training jobs. Disk is the source of truth.

The control room shells out to ``python -m rallyai.train.cli`` (trainer CLI).
If spawn fails, the run directory and ``state.json`` still record the error so
the API remains honest.

Trainer CLI contract::

    python -m rallyai.train.cli \\
      --run-id <id> \\
      --run-dir <path> \\
      --stage <stage id or path> \\
      --workers <int> \\
      --timesteps <int> \\
      --tier <int> \\
      --metrics <path to metrics.jsonl> \\
      --abort <path to abort sentinel>

The trainer must append metrics lines matching ``metrics.schema.json`` and exit
promptly when the abort file appears (after writing a checkpoint / ``run_end``).
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

RunState = Literal["starting", "running", "stopping", "stopped", "error"]

TRAIN_MODULE = "rallyai.train.cli"


def _repo_root() -> Path:
    # control/jobs.py -> control -> rallyai -> sim -> packages -> repo
    return Path(__file__).resolve().parents[4]


def default_runs_dir() -> Path:
    return _repo_root() / "packages" / "sim" / "runs"


@dataclass
class RunRequest:
    stage: str = "proving_ground"
    workers: int = 1
    timesteps: int = 50_000
    tier: int = 0


class JobStore:
    """Filesystem-backed run registry. Memory only holds live Popen handles."""

    def __init__(self, runs_dir: Path | None = None) -> None:
        self.runs_dir = Path(runs_dir) if runs_dir is not None else default_runs_dir()
        self.runs_dir.mkdir(parents=True, exist_ok=True)
        self._procs: dict[str, subprocess.Popen[str]] = {}
        self._lock = threading.RLock()

    def run_dir(self, run_id: str) -> Path:
        return self.runs_dir / run_id

    def metrics_path(self, run_id: str) -> Path:
        return self.run_dir(run_id) / "metrics.jsonl"

    def state_path(self, run_id: str) -> Path:
        return self.run_dir(run_id) / "state.json"

    def config_path(self, run_id: str) -> Path:
        return self.run_dir(run_id) / "config.json"

    def abort_path(self, run_id: str) -> Path:
        return self.run_dir(run_id) / "abort"

    def pid_path(self, run_id: str) -> Path:
        return self.run_dir(run_id) / "pid"

    def _write_json(self, path: Path, data: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
        tmp.replace(path)

    def _read_json(self, path: Path) -> dict[str, Any] | None:
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text())
        except (json.JSONDecodeError, OSError):
            return None

    def _write_state(self, run_id: str, **fields: Any) -> dict[str, Any]:
        path = self.state_path(run_id)
        state = self._read_json(path) or {"run_id": run_id}
        state.update(fields)
        state["run_id"] = run_id
        state["updated_at"] = time.time()
        self._write_json(path, state)
        return state

    def _reconcile(self, run_id: str) -> dict[str, Any] | None:
        """Refresh state from the live process / pid file. Disk wins."""
        state = self._read_json(self.state_path(run_id))
        if state is None:
            return None
        status = state.get("state")
        if status not in ("starting", "running", "stopping"):
            return state

        with self._lock:
            proc = self._procs.get(run_id)

        alive = False
        exit_code: int | None = None
        if proc is not None:
            code = proc.poll()
            if code is None:
                alive = True
            else:
                exit_code = code
                with self._lock:
                    self._procs.pop(run_id, None)
        else:
            pid = state.get("pid")
            if isinstance(pid, int) and pid > 0:
                try:
                    os.kill(pid, 0)
                    alive = True
                except OSError:
                    alive = False

        if alive:
            if status == "starting":
                return self._write_state(run_id, state="running")
            return state

        if status == "stopping":
            return self._write_state(
                run_id,
                state="stopped",
                ended_at=time.time(),
                exit_code=exit_code,
                pid=None,
            )
        if status in ("starting", "running"):
            err = state.get("error") or (
                f"trainer exited with code {exit_code}"
                if exit_code not in (None, 0)
                else "trainer exited unexpectedly"
            )
            return self._write_state(
                run_id,
                state="error" if exit_code not in (None, 0) else "stopped",
                ended_at=time.time(),
                exit_code=exit_code,
                error=err if exit_code not in (None, 0) else state.get("error"),
                pid=None,
            )
        return state

    def start(self, req: RunRequest) -> dict[str, Any]:
        run_id = time.strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:8]
        rd = self.run_dir(run_id)
        rd.mkdir(parents=True, exist_ok=False)

        config = {
            "run_id": run_id,
            "stage": req.stage,
            "workers": int(req.workers),
            "timesteps": int(req.timesteps),
            "tier": int(req.tier),
            "train_module": TRAIN_MODULE,
        }
        self._write_json(self.config_path(run_id), config)

        metrics = self.metrics_path(run_id)
        metrics.touch()
        abort = self.abort_path(run_id)
        if abort.exists():
            abort.unlink()

        self._write_state(
            run_id,
            state="starting",
            created_at=time.time(),
            started_at=time.time(),
            config=config,
            metrics_path=str(metrics),
            error=None,
            exit_code=None,
            ended_at=None,
            pid=None,
        )

        cmd = [
            sys.executable,
            "-m",
            TRAIN_MODULE,
            "--run-id",
            run_id,
            "--run-dir",
            str(rd),
            "--stage",
            req.stage,
            "--workers",
            str(int(req.workers)),
            "--timesteps",
            str(int(req.timesteps)),
            "--tier",
            str(int(req.tier)),
            "--metrics",
            str(metrics),
            "--abort",
            str(abort),
        ]
        log_path = rd / "trainer.log"
        try:
            log_f = log_path.open("w")
            proc = subprocess.Popen(
                cmd,
                cwd=str(_repo_root() / "packages" / "sim"),
                stdout=log_f,
                stderr=subprocess.STDOUT,
                text=True,
                env={**os.environ, "PYTHONUNBUFFERED": "1"},
                start_new_session=True,
            )
        except OSError as exc:
            return self._write_state(
                run_id,
                state="error",
                error=f"failed to spawn trainer: {exc}",
                ended_at=time.time(),
            )

        self.pid_path(run_id).write_text(str(proc.pid) + "\n")
        with self._lock:
            self._procs[run_id] = proc

        time.sleep(0.15)
        code = proc.poll()
        if code is not None:
            with self._lock:
                self._procs.pop(run_id, None)
            log_tail = ""
            try:
                log_tail = log_path.read_text()[-2000:]
            except OSError:
                pass
            err = (
                f"trainer module `{TRAIN_MODULE}` failed to start "
                f"(exit {code}). Is the trainer CLI (`rallyai.train.cli`) installed?\n"
                f"{log_tail}".strip()
            )
            return self._write_state(
                run_id,
                state="error",
                error=err,
                exit_code=code,
                ended_at=time.time(),
                pid=None,
            )

        state = self._write_state(run_id, state="running", pid=proc.pid)
        threading.Thread(
            target=self._wait_proc,
            args=(run_id, proc, log_f),
            daemon=True,
            name=f"wait-{run_id}",
        ).start()
        return state

    def _wait_proc(
        self,
        run_id: str,
        proc: subprocess.Popen[str],
        log_f: Any,
    ) -> None:
        code = proc.wait()
        try:
            log_f.close()
        except Exception:
            pass
        with self._lock:
            self._procs.pop(run_id, None)
        state = self._read_json(self.state_path(run_id)) or {}
        current = state.get("state")
        if current == "stopping":
            self._write_state(
                run_id,
                state="stopped",
                exit_code=code,
                ended_at=time.time(),
                pid=None,
            )
        elif current in ("running", "starting"):
            if code == 0:
                self._write_state(
                    run_id,
                    state="stopped",
                    exit_code=0,
                    ended_at=time.time(),
                    pid=None,
                )
            else:
                log_tail = ""
                try:
                    log_tail = (self.run_dir(run_id) / "trainer.log").read_text()[-2000:]
                except OSError:
                    pass
                self._write_state(
                    run_id,
                    state="error",
                    exit_code=code,
                    error=(log_tail or f"trainer exited {code}")[:2000],
                    ended_at=time.time(),
                    pid=None,
                )

    def stop(self, run_id: str) -> dict[str, Any]:
        state = self._reconcile(run_id)
        if state is None:
            raise KeyError(run_id)
        if state.get("state") not in ("starting", "running", "stopping"):
            return state

        state = self._write_state(run_id, state="stopping")
        abort = self.abort_path(run_id)
        abort.write_text("1\n")

        def _escalate() -> None:
            time.sleep(8.0)
            with self._lock:
                proc = self._procs.get(run_id)
            if proc is not None and proc.poll() is None:
                try:
                    os.killpg(proc.pid, signal.SIGTERM)
                except OSError:
                    proc.send_signal(signal.SIGTERM)
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    try:
                        os.killpg(proc.pid, signal.SIGKILL)
                    except OSError:
                        proc.kill()

        threading.Thread(target=_escalate, daemon=True, name=f"stop-{run_id}").start()
        return state

    def get(self, run_id: str) -> dict[str, Any] | None:
        return self._reconcile(run_id)

    def list_runs(self) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        if not self.runs_dir.exists():
            return out
        for child in sorted(self.runs_dir.iterdir(), reverse=True):
            if not child.is_dir():
                continue
            if not (child / "state.json").exists():
                continue
            detail = self._reconcile(child.name)
            if detail is not None:
                out.append(
                    {
                        "run_id": detail.get("run_id", child.name),
                        "state": detail.get("state"),
                        "created_at": detail.get("created_at"),
                        "started_at": detail.get("started_at"),
                        "ended_at": detail.get("ended_at"),
                        "config": detail.get("config"),
                        "error": detail.get("error"),
                    }
                )
        return out

    def read_metrics(
        self,
        run_id: str,
        since: int = 0,
        limit: int | None = None,
    ) -> dict[str, Any]:
        """Replay metrics JSONL from disk. ``since`` is a 0-based line cursor."""
        if self._read_json(self.state_path(run_id)) is None:
            raise KeyError(run_id)
        path = self.metrics_path(run_id)
        lines: list[dict[str, Any]] = []
        next_cursor = max(0, int(since))
        if not path.exists():
            return {"run_id": run_id, "since": since, "next": next_cursor, "lines": []}

        with path.open("r") as f:
            for idx, raw in enumerate(f):
                if idx < since:
                    continue
                if limit is not None and len(lines) >= limit:
                    break
                text = raw.strip()
                if not text:
                    next_cursor = idx + 1
                    continue
                try:
                    lines.append(json.loads(text))
                except json.JSONDecodeError:
                    break
                next_cursor = idx + 1
        return {
            "run_id": run_id,
            "since": int(since),
            "next": next_cursor,
            "lines": lines,
        }
