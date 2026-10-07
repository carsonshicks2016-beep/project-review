# AudioHelper (Agent B)

Native macOS system-audio capture → FFT → `AudioFrame` over a local WebSocket.

- **Capture:** ScreenCaptureKit (`SCStream`, `capturesAudio = true`). On macOS 13+
  this needs only a one-time **Screen Recording** permission grant — no BlackHole
  or virtual driver. BlackHole is the documented fallback for pre-13.
- **DSP:** Accelerate / vDSP FFT → RMS, 8 log bands, bass/mid/treble, spectral
  centroid, spectral-flux onset, approximate beat confidence.
- **Transport:** JSON frames over `ws://127.0.0.1:17653` (`AUDIO_WS_URL`) at ~60 Hz.
- **Contract:** frame shape must match `src/contracts/audioFrame.ts`.

```bash
swift build --package-path .
.build/debug/AudioHelper
```

> Tempo/BPM from capture is approximate — emit `beatConfidence` and onsets;
> the generation core supplies a genre tempo prior. Don't promise beat-lock.
