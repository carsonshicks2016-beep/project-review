# Rally audio: first listening gate

Status: implemented and verified mechanically; human listening approval and final
engine-character tuning are pending. This is not the entire audio milestone signed off.

## Listening references

Raw standalone listener recordings and launch/build manifests are under
`.rally/visual-review/v3/audio-baseline-{chase,driver,trackside}-720` and
`.rally/visual-review/v3/audio-final-{chase,driver,trackside,hood,helicopter,cinematic}-720`.

Selected before/after MP3s are in `art-source/audio/perspective-review/`. Encoding
preserves relative levels; there is no normalization. The quieter exterior mix is
intentional distance attenuation, not a missing-data substitution. Listen with one
fixed playback volume. Review engine balance, gravel audibility, cabin filtering,
trackside approach/recession and shot-cut continuity on both headphones and speakers.

The six selected recordings use the same frozen checkpoint, Crest bare-road course,
seed 2026, deterministic actions, neutral standing start and real-time simulation.
They are fresh inference, not recorded-trajectory playback or official race results.
Original native captures used differing actual window sizes despite the same nominal
720p launch request. Only the driver before/after pair is resolution-matched for the
performance comparison. Dimensions and immutable hashes are recorded in evidence.json.

## Implemented

- Shared camera perspective profile and five independently routed mixer groups.
- Driver filtering, hood induction emphasis, restrained chase wind, positional
  exterior engine/tyres/impacts and observer-local forest ambience.
- Explicit Doppler from Rigidbody velocity and continuous listener movement;
  follow cameras have neutral pitch. Cinematic uses its actual shot type.
- Cuts, teleports and episode resets discard motion history. Reset fades clear
  engine, tyre, wind and impact state without replaying old-generation events.
- Bounded main-to-audio event queues for short shifts, turbo releases, landings
  and physical collisions; overflow is visible and never blocks the audio thread.
- Fixed-step suspension excitation; tyre transients use smooth saturation instead
  of an occasional internal hard ceiling. Engine source-filter character is retained.
- No audio playback allocation in managed non-viewer/headless runs.
- Opt-in bounded listener WAV capture, event/perspective trace and native lifetime audit.

## Verified on 2026-10-04

- Final current player source hash:
  `84703375e756977d4c8835e53322b46c2326d12bccb1483c6787ebeffb92c6fd`.
- Course:
  `f187eaf562a3681aba54681939a54345c84d484556cef1b97baa2048acb5f8f4`.
- Checkpoint:
  `02d659fb4993a27faeb14292e18eff37c39790038be25714d8d296eda6011b19`.
- Frozen course bundle SHA-256 remains
  `79e4495295e40c4164f878244783ee3e0c679474e6ddc90a531803285a1f392e`.
- Physics, ML, generation and ProjectSettings source hashes unchanged; policy
  observation/action/timestep contract unchanged. Resume checks are not relaxed.
- 84 offline checks passed, including event ordering/overflow/recovery, eight
  resets, pause/resume, old-generation rejection, post-reset event retention,
  landing excitation and silence after settling. Both 44.1/48 kHz and
  256/1024/4096 sample buffers tested. Raw baseline output is preserved separately.
- All six standalone mode recordings have one active listener, five sources,
  nonempty stereo output, zero non-finite/full-scale samples and zero event drops.
- Chase/hood/driver pitch stays 1. Trackside range approximately 0.890-1.123;
  helicopter 0.995-1.003. Cuts and resets report unity pitch; settled exterior
  wind is suppressed and driver cutoff settles at 2200 Hz.
- Eight native audio-rig create/reset/destroy cycles retain five sources, five
  clips, five synths, one controller and one listener. These are audio fixtures,
  not eight complete in-process scene reloads. Whole-scene reload coverage remains
  a separate acceptance item; multiple standalone frozen-course loads were checked.
- Headless one-worker PPO smoke `a18b9bc2482d4206` completed a 1000-step budget,
  exit 0, with exported checkpoints and zero live synths/audio clips/controllers/
  ambience/impact voices. No learning-performance claim follows from this smoke.
- 45 Python platform tests, three checkpoint-default tests and dashboard production
  build passed. No player/worker remains running after the bounded checks.
- Native capture output was 44.1 kHz stereo, DSP buffer 1024 x 4. Offline synthesis
  plus analysis cost roughly 20-29 ms per second of generated audio; this is not
  Unity's complete mixer DSP CPU cost, which the release profiler does not expose.
- Matched 720p driver median/p95/p99: before 16.67/17.32/17.61 ms, after
  16.67/17.25/17.59 ms. Ordinary no-audio/no-screenshot capture overhead check:
  16.67/16.80/17.57 ms. Whole-review means include startup and are lower than 60 FPS.
  These are bounded observations, not a guaranteed frame-rate result.
- macOS reported no recorded thermal/performance warning. No temperature sensor
  measurement or guaranteed thermal limit is claimed.

## Reproduce

Run `sh Tools/audio_check.sh` with Mono installed; it writes a unique review folder,
WAV exports and checks.txt. Native bounded review:

```sh
.venv/bin/python Tools/visual_review.py unique-audio-review --camera Trackside --audio --no-screenshots
.venv/bin/python Tools/visual_review.py unique-lifetime-review --audio-lifecycle --no-screenshots
```

Use `Tools/audio_review_report.py` to validate final capture folders. It verifies
identities, WAV accounting and trace invariants; it does not certify sound quality.
Audio asset preparation is serialized with the existing Unity build path. The
editor-only mixer authoring bridge uses Unity's internal authoring API, isolated in
one file; missing methods/groups fail preparation rather than silently bypass routing.

## Next review decisions

Listen to the matched chase, driver and trackside pairs before changing engine
timbre. Settle exterior level, cabin filtering, induction and impact audibility.
Then tune idle/load/overrun character with an explicit A/B reference. Full listening
acceptance, native collision/landing listening cases and whole-scene reload coverage
are still pending. Do not push or declare the full audio/V3 milestone complete.
