# Agent B — Swift audio-capture helper

**Files:** `helper/`  ·  **Contract out:** `AudioFrame` (over WS)

## Goal
Capture system audio natively and stream analyzed `AudioFrame`s to the renderer.

## Scope
- ScreenCaptureKit `SCStream` with `capturesAudio = true` (macOS 13+; no driver).
- Request/verify Screen Recording permission; clear error if denied.
- vDSP FFT over the PCM buffer → `rms`, 8 log `bands`, `bass`/`mid`/`treble`,
  `centroid`, `onset` (spectral flux), `beatConfidence`.
- WebSocket server on `127.0.0.1:17653`; emit JSON frames ~60 Hz.

## Hard constraints
- JSON shape MUST match `src/contracts/audioFrame.ts` (`AUDIO_FRAME_BANDS = 8`).
- BPM is approximate — provide onsets + `beatConfidence`, not a locked tempo.

## Acceptance
- `swift build` succeeds; binary runs and prompts for permission once.
- With music playing, a WS client receives well-formed frames at ~60 Hz whose
  `rms`/`bands` visibly track the audio (silence → ~0).
