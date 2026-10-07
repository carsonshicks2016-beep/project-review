# Car AI Copilot

Local prototype for a visually rich vehicle assistant.

Open `index.html` in a browser to run it. The first version uses seeded demo data so the interface is useful before real integrations are connected.

## Modes

- Copilot: RAG-style assistant over manuals, repair notes, OBD-II logs, maintenance records, trip data, and audio captures.
- Health Map: interactive 3D-style vehicle map where parts glow based on predicted risk.
- Driving Coach: route heatmap with braking, cornering, fuel, and safety feedback.
- Sound Diagnosis: spectrogram visualization with likely cause confidence bars.
- Self-Driving Lab: top-down simulator with lane following, obstacle avoidance, parking, and turn practice.

## Next Integrations

- Replace the seeded `documents` array with real manual chunks and repair notes.
- Load OBD-II CSV/JSON logs and compute risk scores from actual sensor windows.
- Add browser-side audio feature extraction or a local Python service for spectrogram classification.
- Persist vehicle memory locally with SQLite or IndexedDB.
- Swap the heuristic simulator policy for a real reinforcement-learning loop.
