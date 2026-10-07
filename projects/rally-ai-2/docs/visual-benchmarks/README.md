# WATCH visual benchmarks

These PNGs are fixed visual-regression views, not hand-picked beauty shots.
Each filename matches a `?benchmark=<id>` route in the viewer. That route owns
the replay, timestamp and camera and reconstructs bounded presentation history
before the first paused frame.

Every timestamp below is the measured extreme of the thing it gates, taken from
the recorded frames themselves rather than chosen by eye. Where a frame is
*not* on the peak, the reason is stated.

| capture | replay time | camera | validates | measured at that frame |
|---|---:|---|---|---|
| `corner-roll.png` | proving ground 8.7000 s | chase | ordinary roll, load transfer, grounded shadows | 0.103 rad wheel slip at 21.5 m/s — deliberately an ordinary frame, not the 0.113 rad peak |
| `landing-pitch.png` | proving ground 22.0333 s | chase | landing compression, brake lights, tarmac contact | 23,615 N total load, the peak, 0.17 s after the 21.8417 s touchdown |
| `airborne-droop.png` | proving ground 21.2333 s | sideline | droop, airborne shadow fade and separation | 3.243 m height — the exact recorded apex |
| `gravel-spray.png` | demo 48.4667 s | sideline | wheel-origin spray under cornering load | 0.110 rad wheel slip at 31.1 m/s — the corpus-wide lateral maximum |
| `launch-wheelspin.png` | demo 2.8000 s | chase | driven-wheel spray under power | 0.336 slip ratio at full throttle, 16.4 m/s — the corpus-wide longitudinal maximum on gravel |
| `tarmac-skid.png` | proving ground 22.1333 s | sideline | four tyre-origin skid ribbons | trail accumulated across the 21.87–21.97 s lock-up, which peaked at 0.833 slip ratio |

`tarmac-skid` is deliberately sampled after its lock-up rather than during it:
the ribbons are a trail built from wheel history, so the peak-slip frame has
almost no mark laid down yet.

## Capturing

```bash
npm run benchmarks --prefix packages/viewer
```

The script boots the dev server itself, reads the benchmark list out of
`src/benchmarks.ts` so there is one source of truth, visits each route at a
1184×757 viewport and writes the PNGs here. Captures a route twice and fails on
any pixel difference with `--verify`; all six are currently byte-identical
across repeat runs.

Do not recapture by hand. The comparison is only meaningful if the replay time,
camera and viewport are identical, and the script is what guarantees that.

## Spray visibility

Measured by capturing each view with `surfaceEffects.root.visible` true and
false and counting pixels that differ by more than 8/channel. Emission rates and
gating are unchanged — only particle rise, size, life, opacity and contrast.

| capture | before | after |
|---|---:|---:|
| `launch-wheelspin` | 0.10% | 5.02% |
| `gravel-spray` | 0.68% | 6.80% |
| `corner-roll` | 0.38% | 6.51% |

Targets were `launch-wheelspin` > 3% and `gravel-spray` > 5%. Both clear.
Presentation fix: puffs rise 1.5–3 m over a 1.2–2.1 s life, grow larger, sit at
82% material opacity, and brighten above road albedo with a minority dark core.
Capacity stayed at 180 — longer life alone leaves peak occupancy well under the
pool.

## Input gates

- Brake lights: recorded `in.b`.
- Exhaust: recorded `rpm` and `in.t`.
- Surface particles and tracks: recorded wheel contact, load, slip, vehicle
  speed and surface.
- Damage: explicit replay impact, crash, collision, contact, damage or puncture
  events only.
- Anti-lag: documented viewer-only detection of a recorded throttle drop above
  3,800 RPM.
- Grime: documented viewer-only accumulation of recorded distance weighted by
  the recorded surface.
- Camera vibration: deterministic replay-time waveform gated by recorded
  surface and speed.
- Landing kick: analytic response to explicit replay `landing` events.

No benchmark effect mutates replay, physics, sensing, rewards or training.

## What the replays do not contain

No replay in `packages/viewer/public/replays/` contains a slide. The largest
wheel slip angle anywhere in either file is 0.110 rad — 6.3°. These runs come
from a scripted driver that tracks cleanly, so any benchmark claiming opposite
lock, a rooster tail or a recovered spin would be naming something the recorded
data has never held. That has to wait for an agent that actually slides.
