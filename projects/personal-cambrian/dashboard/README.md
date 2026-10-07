# Mission Control — the Personal Cambrian dashboard

A control plane over the whole engine: every controllable function is a registered
**Action**, run as a cancellable, streaming **Job**, exposed over a small Starlette API
and driven by a dependency-free single-page UI. No build step; the only extra dep is
Starlette (already present).

## Launch

```bash
python3 -m dashboard            # http://127.0.0.1:8077
python3 -m dashboard --port 9000
```

Open the URL in a browser. Pick a workspace (left rail), choose an action, fill the
auto-generated form, and hit **Run**. Long jobs stream a live log + metric chart and
can be stopped mid-run; artifacts (renders, videos, phylogeny trees) appear inline and
in the gallery.

## Architecture

| File | Role |
|---|---|
| `registry.py` | the keystone: `@action` decorator + `Field` schema → auto-generated forms |
| `jobs.py` | job orchestrator: background threads, cooperative cancel, event/metric streaming |
| `actions.py` | the full control surface — every engine domain, with lazy imports |
| `artifacts.py` | indexes + safely serves `runs/` and `renders/` |
| `app.py` | Starlette routes + static UI |
| `static/` | the single-page UI (`index.html`, `style.css`, `app.js`) — vanilla JS |

## Adding a control

Register an action — the UI form, job handling, and streaming come for free:

```python
@action(id="evo.my_thing", label="My thing", category="Evolution", kind="stream",
        streams=["score"],
        fields=[Field("iterations", "int", 50, min=10, max=500)])
def _my_thing(params, ctx):
    for i in range(params["iterations"]):
        ctx.metric(iter=i, score=...)   # live chart
        ctx.check_stop()                # cooperative cancel
    ctx.artifact("renders/out.png")     # appears in the gallery
    return {"summary": ...}
```

## Workspaces (current)

Genome · Simulation · Control · Evolution · Visualize · GPU / Scale · Reality Anchor —
17 actions spanning develop/mutate, rollouts, PPO training, QD / deep-time / POET /
surrogate, Blender render / replay / biomechanics / full report, MJX & Stage-9
benchmarks, and the local-only reality anchor (Whoop + lifting ingest, seed anchoring,
Kalman calibration).

Notes: the reality-anchor actions use **local personal data** and are marked sensitive
(the UI confirms before running). GPU benchmarks are honest about being CPU-bound here.
Blender renders run headless.
