# Web Replay Cockpit

This folder contains the playback-only web spectator surface for The
Cryptographic Heist Engine. It consumes versioned JSONL replay logs through the
dashboard server and never steps the simulator.

Run the dashboard, then open:

```text
http://127.0.0.1:8788/viewer3d/index.html
```

The cockpit can load indexed replay files from `/api/replays`, deep-link to a
specific replay with `?replay=replays/...jsonl`, or load a local JSONL file. It
renders the chase camera, tire trails, smoke cues, waypoint and collision
flares, containment lines, scanner prediction vectors, spoof-highlighted radio
traffic, confidence state, and a browser-gated procedural audio mapping.

This is still intentionally renderer-side only: training, physics, reward
logic, and replay validation remain owned by the Python simulation/dashboard
layers.
