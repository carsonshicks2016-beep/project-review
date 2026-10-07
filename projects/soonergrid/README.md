# SoonerGrid

A local traffic research workspace for Norman game-day mobility: network playback, matched-condition experiments, and explicit model provenance.

**Current status:** exploratory and uncalibrated. The software has numerical safeguards and reproducible experiment records; real-world policy performance remains unvalidated. Read [the research protocol](docs/RESEARCH_PROTOCOL.md) before interpreting results.

## Open the workspace

From this folder:

```sh
python3 -m soonergrid serve
```

Open http://127.0.0.1:8790/visualizer/. The dashboard also works directly from `visualizer/index.html` after building its local data bundle. It uses local assets, system fonts and canvas; no external map, font or chart service is required.

## Reproduce

```sh
python3 -m unittest discover -s tests -v
python3 -m soonergrid benchmark --duration 28800 --dt 15 --workers 1
python3 build_workspace.py
```

Fresh experiments use processed input files already in `data/`, without re-querying OSM. The dashboard provides CSV summaries, original artifact downloads, source/input fingerprints, and clear evidence classifications. The archive preserves earlier viewers with persistent labels.

## Project layout

- `soonergrid/sim/`: flow engines, conservative merge allocation, boundary queues, accounting.
- `soonergrid/policy/`: scheduled and rule-based controllers. Historical MARL naming is retained only for compatibility.
- `soonergrid/research/`: matched experiments and provenance.
- `visualizer/index.html`, `workspace.css`, `workspace.js`: canonical interface.
- `build_workspace.py`: generates dashboard data and provenance inventory; labels older viewers.
- `docs/RESEARCH_PROTOCOL.md`: definitions, validity limits and validation requirements.
- `tests/`: numerical, behavioral and regression checks.

The original scripts remain available for historical workflows. They are not the canonical benchmark path. In particular, `build_resilience_benchmark.py` synthesizes illustrations; it does not execute the network engine. Its output must not be cited as an empirical benchmark.
