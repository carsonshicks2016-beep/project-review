"""Personal Cambrian -- Mission Control dashboard.

A control plane over the whole engine: every controllable function is a registered
Action (registry.py), run as a cancellable streaming Job (jobs.py), exposed over a
small Starlette API (app.py) and driven by a dependency-free single-page UI (static/).

Launch:  python3 -m dashboard      (then open http://127.0.0.1:8077)
"""
