"""CLI entry: ``python -m rallyai.control`` / ``rallyai-control``."""

from __future__ import annotations

import argparse
from pathlib import Path


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="RallyAI control room (jobs, live agent, human drive)"
    )
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument(
        "--runs-dir",
        type=str,
        default=None,
        help="Override runs directory (default: packages/sim/runs)",
    )
    parser.add_argument(
        "--baselines-dir",
        type=str,
        default=None,
        help="Override human-replay save dir (default: docs/baselines/human)",
    )
    args = parser.parse_args(argv)

    import uvicorn

    from rallyai.control.app import create_app

    runs_dir = Path(args.runs_dir) if args.runs_dir else None
    baselines_dir = Path(args.baselines_dir) if args.baselines_dir else None
    app = create_app(runs_dir=runs_dir, baselines_dir=baselines_dir)
    print(f"RallyAI control room on http://{args.host}:{args.port}")
    print("  GET  /api/health                status + version")
    print("  GET  /api/dashboard/summary     active runs / metrics / ckpts")
    print("  POST /api/runs                  start training job")
    print("  GET  /api/runs                  list")
    print("  GET  /api/runs/{id}             detail")
    print("  POST /api/runs/{id}/stop        graceful stop")
    print("  GET  /api/runs/{id}/metrics     JSONL replay (polling fallback)")
    print("  GET  /api/runs/{id}/checkpoints HoF / latest .pt listing")
    print("  GET  /api/runs/{id}/evals       eval JSON beside run")
    print("  GET  /api/replays               viewer public + runs/exports")
    print("  WS   /api/runs/{id}/stream      live JSONL tail")
    print("  POST /api/live                  start live agent (reference pilot default)")
    print("  GET  /api/live                  list live sessions")
    print("  POST /api/live/{id}/stop        stop live session")
    print("  WS   /api/live/{id}/stream      replay frames @ 30 Hz")
    print("  POST /api/drive                 start human drive")
    print("  POST /api/drive/{id}/stop       stop human drive (saves replay)")
    print("  WS   /api/drive/{id}/stream     duplex: inputs @ 30 Hz, frames out")
    print("  keys: WASD/arrows, Space=handbrake, R=reset, P=save")
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
