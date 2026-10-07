# RallyAI sim package

Python owns the world and the brain. The viewer only renders.

## Setup

```bash
cd packages/sim
uv venv --python 3.14 .venv
uv pip install --python .venv -e ".[dev]"
```

## Control room (Phase F)

Server-owned training jobs, live agent streams, and live human drive:

```bash
cd packages/sim
.venv/bin/python -m rallyai.control --host 127.0.0.1 --port 8765
# or: .venv/bin/rallyai-control --port 8765
```

| Method | Path | Role |
|--------|------|------|
| POST | `/api/runs` | start trainer subprocess |
| GET | `/api/runs` | list runs |
| GET | `/api/runs/{id}` | run detail |
| POST | `/api/runs/{id}/stop` | graceful stop (abort sentinel) |
| GET | `/api/runs/{id}/metrics?since=N` | replay metrics JSONL (polling fallback) |
| WS | `/api/runs/{id}/stream?since=N` | tail metrics JSONL |
| POST | `/api/live` | start live agent (reference pilot if no checkpoint) |
| WS | `/api/live/{id}/stream?since=N` | replay frames @ 30 Hz + `sense` sidecar |
| GET | `/api/live/{id}/frames?since=N` | polling fallback for live frames |
| POST | `/api/drive` | start human-drive session |
| WS | `/api/drive/{id}/stream?since=N` | duplex: inputs in, frames out |
| POST | `/api/drive/{id}/input` | buffer `[steer,throttle,brake,handbrake]` |

Metrics truth is `runs/<id>/metrics.jsonl` on disk. Live frames use the same
schema as `packages/shared/schemas/replay.schema.json` — a live stream is an
unfinished replay. Observation debug for TRAIN rides as `sense` on the packet
envelope, not on the frame.

## Test

```bash
cd packages/sim && .venv/bin/python -m pytest tests/ -q
```
