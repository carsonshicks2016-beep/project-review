# Sonic Scenery

A desktop app that turns whatever you're playing on Spotify into a 3D,
procedurally-generated living world — scenery, ambient creatures, and weather.
Each song deterministically gets its own unique world; genre sets the biome,
and the live audio (captured natively on macOS) animates it.

> Status: **scaffold**. See [`PLAN.md`](PLAN.md) for the authoritative design
> and [`docs/tasks/`](docs/tasks) for per-module build briefs.

## Stack

- **Renderer:** Electron + TypeScript + Three.js
- **Audio:** Swift + ScreenCaptureKit + Accelerate/vDSP (system-audio capture, no driver install on macOS 13+)
- **Metadata:** Spotify Web API (PKCE OAuth)

## Layout

```
PLAN.md                 authoritative design doc
src/contracts/          shared interfaces (AudioFrame, TrackContext, WorldSpec)
src/spotify/            Agent A — Spotify metadata service
src/generation/         Agent C — deterministic world generation (pure)
src/renderer/           Agent D — Three.js renderer (Electron)
src/weather/            Agent E — weather systems
src/creatures/          Agent F — ambient creatures
src/main/               Electron main process
helper/                 Agent B — Swift audio-capture helper
docs/tasks/             per-agent /batch briefs
```

## Build (not yet wired)

```bash
npm install        # toolchain
npm run dev        # electron-vite dev (renderer)
# helper:
swift build --package-path helper
```
