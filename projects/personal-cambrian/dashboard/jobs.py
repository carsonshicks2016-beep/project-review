"""Job orchestration: run actions as cancellable, streaming background jobs.

Each job runs in a daemon thread. The runner gets a `JobContext` to emit log lines,
metric records (for live charts), and artifacts, and to check a cooperative stop flag
(checked inside the engine's per-iteration callbacks, so a QD/PPO run can be stopped
mid-flight). Events accumulate on the job; the HTTP layer polls `events_since` -- no
WebSocket thread-bridge, which keeps it robust.
"""
from __future__ import annotations

import threading
import time
import traceback
import uuid


class JobCancelled(Exception):
    pass


class JobContext:
    def __init__(self, job: "Job"):
        self.job = job

    @property
    def should_stop(self) -> bool:
        return self.job.cancel_requested

    def check_stop(self):
        if self.job.cancel_requested:
            raise JobCancelled()

    def log(self, msg):
        self.job.add_event({"kind": "log", "msg": str(msg)})

    def metric(self, **kv):
        self.job.metrics.append(kv)
        self.job.add_event({"kind": "metric", "data": kv})

    def artifact(self, path: str):
        if path not in self.job.artifacts:
            self.job.artifacts.append(path)
        self.job.add_event({"kind": "artifact", "path": path})

    def status(self, msg: str):
        self.job.stage = msg
        self.job.add_event({"kind": "status", "msg": msg})


class Job:
    def __init__(self, action, params: dict):
        self.id = uuid.uuid4().hex[:12]
        self.action = action
        self.params = params
        self.state = "queued"
        self.stage = ""
        self.created = time.time()
        self.started = None
        self.finished = None
        self.events = []
        self.metrics = []
        self.artifacts = []
        self.result = None
        self.error = None
        self.cancel_requested = False
        self._lock = threading.Lock()

    def add_event(self, e: dict):
        with self._lock:
            e["i"] = len(self.events)
            e["t"] = time.time()
            self.events.append(e)

    def events_since(self, n: int) -> list:
        with self._lock:
            return list(self.events[n:])

    @property
    def n_events(self) -> int:
        with self._lock:
            return len(self.events)

    def summary(self) -> dict:
        return {"id": self.id, "action": self.action.id, "label": self.action.label,
                "category": self.action.category, "kind": self.action.kind,
                "state": self.state, "stage": self.stage,
                "created": self.created, "started": self.started,
                "finished": self.finished, "n_metrics": len(self.metrics),
                "n_artifacts": len(self.artifacts), "error": self.error}

    def detail(self) -> dict:
        d = self.summary()
        d.update({"params": self.params, "result": self.result,
                  "metrics": self.metrics, "artifacts": self.artifacts,
                  "streams": self.action.streams})
        return d


class JobManager:
    def __init__(self):
        self.jobs: dict = {}
        self._lock = threading.Lock()

    def submit(self, action, params: dict) -> Job:
        job = Job(action, params)
        with self._lock:
            self.jobs[job.id] = job
        threading.Thread(target=self._run, args=(job,), daemon=True).start()
        return job

    def _run(self, job: Job):
        job.state = "running"
        job.started = time.time()
        ctx = JobContext(job)
        try:
            res = job.action.runner(job.params, ctx)
            job.result = res if isinstance(res, dict) else {"value": res}
            for a in (job.result.get("artifacts") or []):
                if a not in job.artifacts:
                    job.artifacts.append(a)
            job.state = "done"
        except JobCancelled:
            job.state = "cancelled"
            ctx.log("-- cancelled --")
        except Exception as e:                       # noqa: BLE001
            job.state = "error"
            job.error = f"{type(e).__name__}: {e}"
            ctx.log(traceback.format_exc()[-2000:])
        job.finished = time.time()

    def cancel(self, job_id: str) -> bool:
        job = self.jobs.get(job_id)
        if job and job.state in ("queued", "running"):
            job.cancel_requested = True
            return True
        return False

    def get(self, job_id: str):
        return self.jobs.get(job_id)

    def list(self) -> list:
        with self._lock:
            return sorted((j.summary() for j in self.jobs.values()),
                          key=lambda s: s["created"], reverse=True)

    def active_count(self) -> int:
        return sum(1 for j in self.jobs.values() if j.state in ("queued", "running"))


MANAGER = JobManager()
