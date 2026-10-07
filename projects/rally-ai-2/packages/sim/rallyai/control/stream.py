"""Metrics WebSocket: tail JSONL from disk and push new lines.

``WS /api/runs/{id}/stream?since=N`` replays from the client cursor, then tails.
This is a convenience over the file — clients must be able to fall back to
polling ``GET /api/runs/{id}/metrics?since=N`` with the same cursor semantics.
"""

from __future__ import annotations

import asyncio
from typing import Any

from fastapi import FastAPI, WebSocket, WebSocketDisconnect

from rallyai.control.jobs import JobStore


async def _tail_metrics(
    store: JobStore,
    run_id: str,
    since: int,
    websocket: WebSocket,
    poll_s: float = 0.25,
) -> None:
    cursor = max(0, int(since))
    while True:
        try:
            batch = await asyncio.to_thread(store.read_metrics, run_id, cursor, 256)
        except KeyError:
            await websocket.send_json(
                {"type": "error", "message": f"unknown run: {run_id}"}
            )
            return

        for line in batch["lines"]:
            await websocket.send_json(line)
        cursor = int(batch["next"])

        state = await asyncio.to_thread(store.get, run_id)
        if state is None:
            await websocket.send_json(
                {"type": "error", "message": f"unknown run: {run_id}"}
            )
            return

        if state.get("state") in ("stopped", "error") and not batch["lines"]:
            await websocket.send_json(
                {
                    "type": "stream_end",
                    "run_id": run_id,
                    "state": state.get("state"),
                    "next": cursor,
                }
            )
            return

        await asyncio.sleep(poll_s)


def mount_metrics_stream(app: FastAPI, store: JobStore) -> None:
    @app.websocket("/api/runs/{run_id}/stream")
    async def run_stream(websocket: WebSocket, run_id: str) -> None:
        await websocket.accept()
        since_raw = websocket.query_params.get("since", "0")
        try:
            since = int(since_raw)
        except ValueError:
            since = 0

        if store.get(run_id) is None:
            await websocket.send_json(
                {"type": "error", "message": f"unknown run: {run_id}"}
            )
            await websocket.close(code=1008)
            return

        try:
            await _tail_metrics(store, run_id, since, websocket)
        except WebSocketDisconnect:
            return
        except (OSError, RuntimeError, ValueError) as exc:
            try:
                await websocket.send_json({"type": "error", "message": str(exc)})
            except (OSError, RuntimeError):
                pass
            return


def polling_hint() -> dict[str, Any]:
    """Documented fallback for clients that cannot keep a WebSocket open."""
    return {
        "fallback": "GET /api/runs/{id}/metrics?since=N",
        "cursor": "Use the `next` field from the previous response as `since`.",
        "interval_s": 0.5,
    }
