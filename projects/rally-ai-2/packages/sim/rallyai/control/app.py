"""FastAPI control room — jobs, live agent stream, live human drive.

The server owns training jobs, live agent sessions, and human-drive sessions.
Browser tabs are disposable; ``runs/<id>/`` on disk is the truth for training
(``state.json`` + ``metrics.jsonl``). A live stream (agent or human) is a
replay that has not finished being written.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from rallyai.control import dashboard as dash
from rallyai.control.drive import mount_drive
from rallyai.control.jobs import JobStore, RunRequest
from rallyai.control.live import mount_live
from rallyai.control.stream import mount_metrics_stream, polling_hint


class StartRunBody(BaseModel):
    stage: str = Field(default="proving_ground", min_length=1)
    workers: int = Field(default=1, ge=1, le=64)
    timesteps: int = Field(default=50_000, ge=1, le=500_000_000)
    tier: int = Field(default=0, ge=0, le=32)


def create_app(
    runs_dir: Path | None = None,
    baselines_dir: Path | None = None,
    replay_dirs: list[Path] | None = None,
) -> FastAPI:
    app = FastAPI(
        title="RallyAI Control Room",
        description=(
            "Server-owned training jobs, live agent streams, and human drive. "
            "Metrics live in JSONL on disk; live frames use the replay contract."
        ),
        version=dash.CONTROL_VERSION,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    store = JobStore(runs_dir=runs_dir)
    app.state.jobs = store
    configured_replays = (
        [Path(p) for p in replay_dirs]
        if replay_dirs is not None
        else dash.default_replay_dirs(store.runs_dir)
    )
    app.state.replay_dirs = configured_replays

    @app.get("/api/health")
    def health() -> dict[str, Any]:
        return {
            "ok": True,
            "status": "ok",
            "version": dash.CONTROL_VERSION,
            "service": "rallyai-control",
            "runs_dir": str(store.runs_dir),
            "replay_dirs": [str(p) for p in configured_replays],
            "metrics_polling": polling_hint(),
            "live": {
                "start": "POST /api/live",
                "stream": "WS /api/live/{id}/stream",
                "frames": "GET /api/live/{id}/frames?since=N",
                "default_driver": "rallyai.env.pilot.ReferencePilot (no checkpoint)",
            },
            "drive": {
                "start": "POST /api/drive",
                "stream": "WS /api/drive/{id}/stream",
                "frames": "GET /api/drive/{id}/frames?since=N",
                "input": "POST /api/drive/{id}/input or WS type=input",
                "controls": "WASD/arrows + Space handbrake @ 30 Hz",
                "replay_dir": "docs/baselines/human/",
            },
            "dashboard": {
                "summary": "GET /api/dashboard/summary",
                "checkpoints": "GET /api/runs/{id}/checkpoints",
                "evals": "GET /api/runs/{id}/evals",
                "replays": "GET /api/replays",
            },
        }

    @app.get("/api/dashboard/summary")
    def dashboard_summary() -> dict[str, Any]:
        return dash.dashboard_summary(store)

    @app.get("/api/runs/{run_id}/checkpoints")
    def get_checkpoints(run_id: str) -> dict[str, Any]:
        if not dash.run_exists(store, run_id):
            raise HTTPException(status_code=404, detail=f"unknown run: {run_id}")
        return {"run_id": run_id, "checkpoints": dash.list_checkpoints(store, run_id)}

    @app.get("/api/runs/{run_id}/evals")
    def get_evals(run_id: str) -> dict[str, Any]:
        if not dash.run_exists(store, run_id):
            raise HTTPException(status_code=404, detail=f"unknown run: {run_id}")
        return {"run_id": run_id, "evals": dash.list_evals(store, run_id)}

    @app.get("/api/replays")
    def get_replays() -> dict[str, Any]:
        return {
            "dirs": [str(p) for p in configured_replays],
            "replays": dash.list_replays(configured_replays),
        }

    @app.post("/api/runs", status_code=201)
    def start_run(body: StartRunBody) -> dict[str, Any]:
        return store.start(
            RunRequest(
                stage=body.stage,
                workers=body.workers,
                timesteps=body.timesteps,
                tier=body.tier,
            )
        )

    @app.get("/api/runs")
    def list_runs() -> dict[str, Any]:
        return {"runs": store.list_runs()}

    @app.get("/api/runs/{run_id}")
    def get_run(run_id: str) -> dict[str, Any]:
        detail = store.get(run_id)
        if detail is None:
            raise HTTPException(status_code=404, detail=f"unknown run: {run_id}")
        return detail

    @app.post("/api/runs/{run_id}/stop")
    def stop_run(run_id: str) -> dict[str, Any]:
        try:
            return store.stop(run_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=f"unknown run: {run_id}") from exc

    @app.get("/api/runs/{run_id}/metrics")
    def get_metrics(
        run_id: str,
        since: int = Query(default=0, ge=0, description="0-based line cursor"),
        limit: int | None = Query(default=None, ge=1, le=10_000),
    ) -> dict[str, Any]:
        try:
            return store.read_metrics(run_id, since=since, limit=limit)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=f"unknown run: {run_id}") from exc

    @app.get("/api/runs/{run_id}/decisions")
    def get_decisions(run_id: str) -> dict[str, Any]:
        if not dash.run_exists(store, run_id):
            raise HTTPException(status_code=404, detail=f"unknown run: {run_id}")
        dec_file = store.get_dir(run_id) / "decisions.jsonl"
        decisions = []
        if dec_file.exists():
            import json
            try:
                with open(dec_file, "r") as f:
                    for line in f:
                        if line.strip():
                            decisions.append(json.loads(line))
            except Exception:
                pass
        return {"run_id": run_id, "decisions": decisions}

    @app.get("/api/runs/{run_id}/replays")
    def get_run_replays(run_id: str) -> dict[str, Any]:
        if not dash.run_exists(store, run_id):
            raise HTTPException(status_code=404, detail=f"unknown run: {run_id}")
        run_dir = store.get_dir(run_id)
        replays = []
        if run_dir.exists():
            for p in run_dir.glob("replay_*.json"):
                replays.append({"name": p.name, "size": p.stat().st_size})
        return {"run_id": run_id, "replays": replays}

    @app.post("/api/runs/{run_id}/evaluate")
    def evaluate_run(run_id: str) -> dict[str, Any]:
        if not dash.run_exists(store, run_id):
            raise HTTPException(status_code=404, detail=f"unknown run: {run_id}")
        import subprocess
        import sys
        cmd = [
            sys.executable,
            "-m",
            "rallyai.train.eval_cli",
            "--run-id",
            run_id,
            "--out-dir",
            str(store.runs_dir)
        ]
        subprocess.Popen(cmd)
        return {"run_id": run_id, "status": "evaluation started"}

    mount_metrics_stream(app, store)
    mount_live(app)
    mount_drive(app, baselines_dir=baselines_dir)
    return app
