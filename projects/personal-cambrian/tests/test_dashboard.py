"""Dashboard machinery: action registry, schemas, and the job orchestrator.

Tests the control-plane plumbing (no engine, no server): every registered action has
a JSON-serializable schema, fields coerce correctly, and the JobManager runs and
cancels jobs. Skips cleanly if Starlette is absent.

Runs:  python3 tests/test_dashboard.py
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import starlette  # noqa: F401
    _SKIP = False
except Exception:        # noqa: BLE001
    _SKIP = True

if not _SKIP:
    from dashboard.app import app  # noqa: F401  (registers all actions)
    from dashboard.registry import all_actions, categories, get_action, Action, Field
    from dashboard.jobs import JobManager


def _wait(job, timeout=5.0):
    t0 = time.time()
    while job.state in ("queued", "running") and time.time() - t0 < timeout:
        time.sleep(0.02)
    return job


def test_actions_registered_with_valid_schemas():
    acts = all_actions()
    assert len(acts) >= 15                               # the full control surface
    cats = categories()
    for needed in ("Genome", "Evolution", "Visualize", "Reality Anchor"):
        assert needed in cats
    for a in acts:
        s = a.schema()
        json.dumps(s)                                    # serializable for the UI
        assert s["id"] and s["category"] and s["kind"] in ("value", "stream", "render")


def test_field_coercion():
    assert Field("n", "int", 0).coerce("12") == 12
    assert Field("x", "float", 0.0).coerce("1.5") == 1.5
    assert Field("b", "bool", False).coerce("true") is True
    assert Field("b", "bool", False).coerce(False) is False
    assert Field("s", "enum", "a", options=["a", "b"]).coerce("b") == "b"
    assert Field("n", "int", 7).coerce(None) == 7         # missing -> default


def test_action_coerce_fills_defaults():
    a = get_action("evo.run_qd")
    p = a.coerce({"iterations": "25"})
    assert p["iterations"] == 25 and isinstance(p["bins"], int)
    assert p["novelty"] in (True, False)


def test_job_runs_and_returns_result():
    jm = JobManager()
    act = Action(id="t.echo", label="echo", category="test",
                 runner=lambda p, ctx: {"echo": p["x"] * 2}, fields=[Field("x", "int", 1)])
    job = jm.submit(act, act.coerce({"x": 5}))
    _wait(job)
    assert job.state == "done" and job.result["echo"] == 10


def test_job_streams_metrics():
    jm = JobManager()

    def run(p, ctx):
        for i in range(5):
            ctx.metric(iter=i, v=i * i)
        return {"done": True}

    act = Action(id="t.stream", label="s", category="test", kind="stream", runner=run)
    job = _wait(jm.submit(act, {}))
    assert job.state == "done" and len(job.metrics) == 5
    assert job.metrics[-1]["v"] == 16
    assert any(e["kind"] == "metric" for e in job.events)


def test_job_is_cancellable():
    jm = JobManager()

    def loop(p, ctx):
        for i in range(10000):
            ctx.metric(i=i)
            ctx.check_stop()
            time.sleep(0.005)

    act = Action(id="t.loop", label="loop", category="test", runner=loop)
    job = jm.submit(act, {})
    time.sleep(0.1)
    assert jm.cancel(job.id) is True
    _wait(job)
    assert job.state == "cancelled"


def test_job_captures_errors():
    jm = JobManager()

    def boom(p, ctx):
        raise ValueError("kaboom")

    act = Action(id="t.err", label="e", category="test", runner=boom)
    job = _wait(jm.submit(act, {}))
    assert job.state == "error" and "kaboom" in job.error


if __name__ == "__main__":
    if _SKIP:
        print("SKIP  starlette not installed")
        sys.exit(0)
    fns = [v for k, v in sorted(globals().items())
           if k.startswith("test_") and callable(v)]
    passed = 0
    for fn in fns:
        try:
            fn()
            passed += 1
            print(f"PASS  {fn.__name__}")
        except Exception as e:  # noqa: BLE001
            print(f"FAIL  {fn.__name__}: {type(e).__name__}: {e}")
    print(f"\n{passed}/{len(fns)} passed")
    sys.exit(0 if passed == len(fns) else 1)
