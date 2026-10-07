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

## Viewer3D (Browser-based)
A standalone browser 3D drive viewer built with Three.js ships in `viewer3d/`. It is accessible from the Command Center's `/3d/` route and provides a fully 3D perspective of the simulation data.

## Procedural Track Generation

Tracks are generated dynamically using Catmull-Rom splines, and feature varying archetypes:
- **`gp`**: Flowing circuits.
- **`technical`**: Tight hairpins and multi-apex corners.
- **`speedway`**: Fast, sweeping ovals.
- **`touge`**: Mountain switchbacks.

Named hand-crafted tracks (e.g., `circuit`, `crescent`, `esses`, `hook`, `ridge`) are deterministic and provide consistent benchmarks for specialized training. Every track (procedural or named) includes seed-deterministic elevation and banking.
