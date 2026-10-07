# UI and Visualization

Supra Drift employs a multi-tiered visualization and control approach. It separates the driving/simulation view from the training management dashboard.

## The Command Center (Web Dashboard)

The Command Center is the project's control room. It is a self-contained web dashboard (Flask + Vanilla JS) running on `http://localhost:8770`.

- **Live SSE Streaming**: Training metrics, logs, and parsed events stream to the dashboard over Server-Sent Events (SSE). It builds live charts showing the `eval` drift and lap scores.
- **The Pit Wall (Fable Five)**: A specialized tab replacing standard status widgets with a race-engineering board rendered from live data. It features:
  - A particle-swarm visualizer mimicking the circuit.
  - An eval timeline and "calm ladder" (learning rate adjustments).
  - A Brain Lab strip showing diagnostics traces (speed vs. reference over elevation) scrub-synced to the map cursor.
- **Task Management**: Launch training runs directly from the UI. Edits to the training code automatically apply to new runs (as it spawns `run.py` fresh).

## The Viewers: 2D Classic (V1) vs 2.5D Projected (V2)

The main application window (PyGame) provides the driving and watching visualization.

### Viewer V2 (Default)
Viewer V2 (`supra/viewer2.py`) is a from-scratch projected 2.5D renderer. 
- **Faux-3D Projection**: One projector draws the road, kerbs, props, skids, and car meshes. 
- **Elevation**: The road renders with real elevation (crests bulge, dips sink). The car's shadow stays glued to the road while the body lifts on jumps.
- **Dynamic Camera**: A heading-up chase camera rotates the world around the car. A Broadcast Director (`V`) auto-cuts between chase, low rear-quarter, locked side pan, wide drone, and fixed trackside shots.
- **Weather & Environment**: Features day/dusk/night cycles with headlight cones (`G`), cycling weather (`H`), and dynamic trackside scenes (camera flashes, wind turbines).

### Viewer V1 (Classic)
Viewer V1 (`supra/app.py`) is the original, purely top-down 2D viewer. It is untouched and always available via the `--classic` flag. Multi-car race modes automatically fall back to V1.

## Fable Five 3D Brain Observatory

The clean-room browser application in `observatory/` is available at
`/observatory/`. It visualizes validated Nordschleife Fable PPO checkpoints for
the Mazda 787B and Porsche 919 Evo without taking authority away from Python.

- Python reconstructs the checkpoint's exact car, observation layout, sensor
  specification, pace or hybrid blocks, RaceBox behavior, fixed simulation step,
  deterministic policy inference, brain probes, and telemetry-driven audio.
- The browser receives immutable telemetry and renders the road, local terrain,
  scenery, car, cameras, Brain X-Ray, Engineer, and Replay overlays. Camera and
  listener state can affect audio perspective only; it never feeds physics,
  observations, or inference.
- The road truth layer retains the simulator's processed 6,944 samples, constant
  width, and zero bank. Terrain, curbs, guardrails, forests, fencing, and landmarks
  are deterministic non-colliding visual context, not surveyed circuit evidence.
- The project-authored 919 proxy is shown at its declared 5.078 m visual length;
  Engineering mode separately shows the Fable collision footprint. The Fable 919
  is permanently identified as a legacy approximation and never as faithful-v2.
- Immutable replay and Follow Active Best are supported. Follow mode can swap only
  to a newly validated same-edition checkpoint at a lap, termination, or reset
  boundary.

The projected PyGame viewer remains the default 2D/2.5D alternative and its launch
path is unchanged.

## Procedural Track Generation

Tracks are generated dynamically using Catmull-Rom splines, and feature varying archetypes:
- **`gp`**: Flowing circuits.
- **`technical`**: Tight hairpins and multi-apex corners.
- **`speedway`**: Fast, sweeping ovals.
- **`touge`**: Mountain switchbacks.

Named hand-crafted tracks (e.g., `circuit`, `crescent`, `esses`, `hook`, `ridge`) are deterministic and provide consistent benchmarks for specialized training. Every track (procedural or named) includes seed-deterministic elevation and banking.
