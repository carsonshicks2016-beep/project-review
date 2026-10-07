"""Mission Control -- the dashboard HTTP layer (Starlette).

Serves the single-page UI and a small REST API over the Action Registry + Job
Manager. Jobs run in background threads; the UI polls `/api/jobs/{id}/events` for
live logs, metrics, and artifacts. No build step, no extra deps beyond Starlette.
"""
from __future__ import annotations

import os

from starlette.applications import Starlette
from starlette.responses import JSONResponse, FileResponse, PlainTextResponse
from starlette.routing import Route, Mount
from starlette.staticfiles import StaticFiles

from . import actions as _actions          # noqa: F401  (registers all actions)
from .registry import all_actions, get_action, categories
from .jobs import MANAGER
from .artifacts import list_artifacts, safe_path, kind_of
from . import creatures
from .session import SESSION

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
STATIC = os.path.join(HERE, "static")


async def index(request):
    return FileResponse(os.path.join(STATIC, "theater.html"))


async def console(request):
    return FileResponse(os.path.join(STATIC, "index.html"))


# -- Theater: creature data (sync defs -> Starlette runs them in a threadpool) --
def api_creature(request):
    return JSONResponse(creatures.creature_payload(request.query_params.get("ref", "")))


def api_trajectory(request):
    q = request.query_params
    return JSONResponse(creatures.trajectory_payload(
        q.get("ref", ""), q.get("niche", "locomotion"), int(q.get("steps", 90))))


def api_biomechanics(request):
    q = request.query_params
    return JSONResponse(creatures.biomechanics_payload(q.get("ref", ""), int(q.get("steps", 70))))


# -- Theater: evolution session control --
async def api_session_tree(request):
    return JSONResponse(SESSION.tree())


async def api_session_status(request):
    return JSONResponse(SESSION.status())


def api_session_step(request):
    SESSION.step(int(request.query_params.get("n", 1)))
    return JSONResponse(SESSION.tree())


async def api_session_play(request):
    SESSION.play()
    return JSONResponse({"playing": True})


async def api_session_pause(request):
    SESSION.pause()
    return JSONResponse({"playing": False})


def api_session_reset(request):
    q = request.query_params
    SESSION.reset(q.get("seed_creature", "quadruped"), q.get("niche", "locomotion"),
                  int(q.get("bins", 10)), int(q.get("seed", 0)))
    return JSONResponse(SESSION.status())


async def api_status(request):
    import multiprocessing
    blender = "/Applications/Blender.app/Contents/MacOS/Blender"
    def have(mod):
        import importlib.util
        return importlib.util.find_spec(mod) is not None
    return JSONResponse({
        "cpus": multiprocessing.cpu_count(),
        "torch": have("torch"), "mujoco": have("mujoco"),
        "mjx": have("jax") and have("mujoco.mjx"),
        "blender": os.path.exists(blender),
        "active_jobs": MANAGER.active_count(),
        "categories": categories(),
    })


async def api_actions(request):
    return JSONResponse([a.schema() for a in all_actions()])


async def api_run(request):
    action = get_action(request.path_params["action_id"])
    if action is None:
        return JSONResponse({"error": "unknown action"}, status_code=404)
    try:
        params = await request.json()
    except Exception:        # noqa: BLE001
        params = {}
    job = MANAGER.submit(action, action.coerce(params))
    return JSONResponse({"job_id": job.id})


async def api_jobs(request):
    return JSONResponse(MANAGER.list())


async def api_job(request):
    job = MANAGER.get(request.path_params["job_id"])
    if job is None:
        return JSONResponse({"error": "unknown job"}, status_code=404)
    return JSONResponse(job.detail())


async def api_job_events(request):
    job = MANAGER.get(request.path_params["job_id"])
    if job is None:
        return JSONResponse({"error": "unknown job"}, status_code=404)
    since = int(request.query_params.get("since", 0))
    events = job.events_since(since)
    return JSONResponse({
        "events": events, "next": since + len(events),
        "state": job.state, "stage": job.stage, "error": job.error,
        "result": job.result if job.state in ("done", "error", "cancelled") else None,
        "artifacts": list(job.artifacts), "n_metrics": len(job.metrics),
    })


async def api_job_cancel(request):
    ok = MANAGER.cancel(request.path_params["job_id"])
    return JSONResponse({"cancelled": ok})


async def api_artifacts(request):
    return JSONResponse(list_artifacts())


async def api_file(request):
    rel = request.query_params.get("path", "")
    full = safe_path(rel)
    if full is None:
        return PlainTextResponse("not found", status_code=404)
    return FileResponse(full, media_type=None,
                        headers={"Cache-Control": "no-cache"})


routes = [
    Route("/", index),
    Route("/console", console),
    Route("/api/creature", api_creature),
    Route("/api/creature/trajectory", api_trajectory),
    Route("/api/creature/biomechanics", api_biomechanics),
    Route("/api/session/tree", api_session_tree),
    Route("/api/session/status", api_session_status),
    Route("/api/session/step", api_session_step, methods=["POST"]),
    Route("/api/session/play", api_session_play, methods=["POST"]),
    Route("/api/session/pause", api_session_pause, methods=["POST"]),
    Route("/api/session/reset", api_session_reset, methods=["POST"]),
    Route("/api/status", api_status),
    Route("/api/actions", api_actions),
    Route("/api/actions/{action_id}/run", api_run, methods=["POST"]),
    Route("/api/jobs", api_jobs),
    Route("/api/jobs/{job_id}", api_job),
    Route("/api/jobs/{job_id}/events", api_job_events),
    Route("/api/jobs/{job_id}/cancel", api_job_cancel, methods=["POST"]),
    Route("/api/artifacts", api_artifacts),
    Route("/api/file", api_file),
    Mount("/static", app=StaticFiles(directory=STATIC), name="static"),
]

app = Starlette(routes=routes)
