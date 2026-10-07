"""FastAPI control room: jobs, live agent stream, live human drive.

Launch::

    cd packages/sim
    .venv/bin/python -m rallyai.control --host 127.0.0.1 --port 8765

Or via the console script::

    .venv/bin/rallyai-control --port 8765

Endpoints::

    POST /api/runs
    GET  /api/runs
    GET  /api/runs/{id}
    POST /api/runs/{id}/stop
    GET  /api/runs/{id}/metrics?since=N
    WS   /api/runs/{id}/stream?since=N
    GET  /api/runs/{id}/checkpoints
    GET  /api/runs/{id}/evals

    GET  /api/dashboard/summary
    GET  /api/replays
    GET  /api/health

    POST /api/live                         start live agent (pilot or checkpoint)
    GET  /api/live
    GET  /api/live/{id}
    POST /api/live/{id}/stop
    GET  /api/live/{id}/frames?since=N
    WS   /api/live/{id}/stream?since=N

    POST /api/drive                        start human-drive session
    GET  /api/drive
    GET  /api/drive/{id}
    POST /api/drive/{id}/stop
    POST /api/drive/{id}/input            buffer [steer,throttle,brake,handbrake]
    GET  /api/drive/{id}/frames?since=N
    WS   /api/drive/{id}/stream?since=N   duplex: inputs in, replay frames out

Python owns truth. Metrics live in ``runs/<id>/metrics.jsonl`` on disk. A live
stream (agent or human) is a replay that has not finished being written — frames
match ``packages/shared/schemas/replay.schema.json``. Observation debug for TRAIN
G2 rides as ``sense`` on the live envelope, not on the frame itself.
"""

from .app import create_app

__all__ = ["create_app"]
