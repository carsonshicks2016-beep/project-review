# Fable Observatory (from-scratch remake)

Clean-room Three.js client for authoritative Fable Five playback. Python owns
physics, policy, probes, and audio (`supra/observatory.py`); this browser app
only renders streamed state and uploads a camera/listener pose.

Frozen pre-remake client: [`legacy/observatory/`](../legacy/observatory/).

## Build

```bash
cd observatory
npm install
npm run assets:check
npm test
npm run build
```

`npm run validate` runs assets:check + tests + build. Open
`http://localhost:8770/observatory/` from Command Center.

Sibling Phase 0 client (same `fable-observatory-v1` sessions, separate package):
[`../watch25d/`](../watch25d/) → `http://127.0.0.1:8770/watch25d/?edition=919`.

## Layout

| Path | Role |
|------|------|
| `src/main.js` | Boot, rAF loop, session lifecycle |
| `src/net.js` | REST session + telemetry WS + controls |
| `src/timeline.js` | Frame normalize / interpolate / discontinuity snap |
| `src/world.js` | Truth/visual GLBs, terrain LODs, vehicle pose |
| `src/sky.js` | Sky dome, sun, fog |
| `src/cameras.js` | Broadcast director (chase/roof/trackside/drone/free) |
| `src/conditions.js` | Weather table + `dry\|dusk\|night` ambience map |
| `src/audio.js` | FOA1 → AudioWorklet bridge |
| `src/fx.js` | Rain + adaptive quality |
| `src/chrome.js` | HUD / panels / identity rail |
| `src/theme.css` | Cold-signal instrument glass |
| `public/audio/pcm-jitter-processor.js` | PCM jitter worklet |
| `public/assets/` | Server-built world + vehicle GLBs |

## Hard invariants

- No envelope heat ribbon on asphalt.
- No client elevation invention — road height comes from `track.road_z_m`.
- Camera / weather never mutate the sim.
- Doppler is server delay-only; client plays PCM as received.
- Edition registry wins over any `?car=` query param.
