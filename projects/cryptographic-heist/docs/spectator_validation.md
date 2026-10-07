# Spectator UI/Audio Validation

The spectator gate proves that the playback and audio surfaces are still wired
to replay evidence before visual polish work proceeds.

```bash
python3 scripts/verify_spectator.py --json-out logs/spectator_validation.json
```

The manifest checks:

- Confidence-to-soundtrack mapping exposes low/mid/high buckets for chaotic
  jazz, hybrid tension, and structured classical modes.
- The static web replay cockpit still contains the required canvas, replay
  picker, radio transcript, scanner prediction panel, confidence, and audio
  controls, plus `soundtrackMode` / `updateAudio` / 2D prediction and
  containment helpers.
- Three.js remains the primary renderer with a 2D fallback canvas
  (`data-fallback-renderer="2d"` and `#scene.fallback-hidden`).
- `/api/replay` returns sampled replay frames with agents, radio events,
  scanner predictions, confidence, and waypoint data.
- `/api/replays` gives every valid replay a dashboard deep link into the web
  cockpit.
- The Pygame spectator loop can run in dummy-video mode without opening a real
  window.
- Browser visual acceptance (`scripts/verify_viewer_visual.py`) captures
  desktop and mobile screenshots and asserts non-blank canvas pixels when a
  headless Chromium is available.

Dashboard readiness treats `logs/spectator_validation.json` as a gate. If the
manifest is missing or failing, `/api/readiness` recommends the registered
`verify_spectator` command.
