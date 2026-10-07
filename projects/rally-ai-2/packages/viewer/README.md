# WATCH

Replay playback. Load a recorded run, render the stage it embeds, and watch the
car drive it.

**This package never simulates.** Python owns the truth — physics, sensing,
reward and termination all live in `packages/sim`. WATCH renders state it is
handed and never steps the world. If you find yourself wanting to integrate
something here, the value is missing from the replay contract and that is where
it should be added.

## Run it

```bash
npm install --prefix packages/viewer
```

Export a replay for it to load, then start the dev server:

```bash
packages/sim/.venv/bin/python -m rallyai.env.export --seed 201 --tier 2 --name demo
```

```bash
npm run dev --prefix packages/viewer
```

Then <http://localhost:5173>, or `?r=proving_ground` for a different file.
Replays are written to `public/replays/` and are **not committed** — they are
generated, and regenerating one is a single command.

## Program dashboard

One page for the whole program: what is training, what it is scoring, what has
been proven, and what is left. It is a separate surface from WATCH, not a
frame around it.

### Opening it

The dashboard renders without a server — every section falls back to committed
JSON under `public/dashboard/`, so a fresh clone shows real measured numbers
rather than empty panels. Only the live sections (Train, Live, Drive, and the
live parts of Checkpoints and Pipeline) need the control room.

```bash
npm run dev --prefix packages/viewer
```

```text
http://localhost:5173/?mode=dashboard
```

For the live half, start the control room first — a second terminal, left
running:

```bash
packages/sim/.venv/bin/python -m rallyai.control --port 8765
```

It serves jobs (`/api/runs`), the live agent stream (`/api/live`) and human
drive (`/api/drive`). The dev server proxies `/api` to `127.0.0.1:8765` with
`ws: true`, so the browser talks to one origin and WebSocket metrics and 30 Hz
frame streams work without CORS. Nothing needs configuring for the common
case; `?api=http://host:port` points at a control room somewhere else, and a
production build defaults to `http://127.0.0.1:8765` because there is no proxy
to lean on.

Panels never block on the control room. If it is down they say so and keep
showing their static data.

### Routes

Deep links keep `?mode=dashboard` and put the section in the hash, e.g.
`http://localhost:5173/?mode=dashboard#/evals`. An unknown or missing hash
normalises to `#/overview`.

| route | panel | what it answers | needs control room |
|---|---|---|---|
| `#/overview` | `panels/overview.ts` | Identity, run status, headline measured numbers, phase strip A–J | optional |
| `#/roadmap` | `panels/roadmap.ts` | Phase dependency graph and work-item status | no |
| `#/pipeline` | `panels/pipeline.ts` | Curriculum tiers, stage reward weights, mastery gates, live tier occupancy | optional |
| `#/train` | `panels/trainMetrics.ts` | Start/stop runs; reward, completion, PPO and throughput as they stream | yes |
| `#/live` | `panels/live.ts` | Watch a checkpoint drive now, with the sensor overlay | yes |
| `#/drive` | `panels/drive.ts` | Record a human baseline; held-out seeds and current times | yes |
| `#/watch` | `panels/watch.ts` | Replay catalogue; deep-links into the WATCH player | no |
| `#/evals` | `panels/evals.ts` | Held-out evaluation evidence, per-seed and per-sector | no |
| `#/checkpoints` | `panels/checkpoints.ts` | Hall-of-Fame slots per run, with eval commands | yes |
| `#/throughput` | `panels/throughput.ts` | Measured steps/s vs workers, sync and async, scaling efficiency | no |

The other two surfaces are unchanged and reachable from the nav footer: WATCH
is the default (`/`) and TRAIN is `?mode=train`.

### Where each panel gets its data

| source | used by |
|---|---|
| `public/dashboard/measured.json` | Overview |
| `public/dashboard/roadmap-status.json` | Roadmap, Overview phase strip |
| `public/dashboard/stages.json` | Pipeline |
| `public/dashboard/throughput.json` | Throughput |
| `public/dashboard/replays.json` | Watch |
| `public/dashboard/evals/index.json` | Evals |
| `public/dashboard/drive/times.json` | Drive |
| control room `/api/*` | Train, Live, Drive, Checkpoints, and live overlays elsewhere |

These are committed on purpose. A dashboard that only works next to a running
trainer is a dashboard nobody opens.

### How a panel is wired

`src/dashboard/` is the shell: `shell.ts` (layout), `nav.ts` (section list and
order), `router.ts` (hash routing), `registry.ts` (section → mount),
`DashboardApp.ts` (boot and route → panel), `types.ts`, `ui.ts` (shared DOM
helpers), plus `theme.css` and `dashboard.css`. Panels live in
`src/dashboard/panels/` and own their own markup and stylesheet.

A panel is a mount function that fills the element it is handed and returns an
optional disposer:

```ts
import { registerPanel } from "../registry";
import type { PanelMount } from "../types";

export const mountEvals: PanelMount = (el, ctx) => {
  // ctx: { apiBase, root, navigate }
  el.textContent = "evals panel";
  return () => { /* tear down timers, sockets, listeners */ };
};

registerPanel("evals", mountEvals);
```

Import `registerPanel` from `../registry`, not from `../DashboardApp`. Both
work, but the registry is a leaf module and cannot produce an import cycle.

`panels/index.ts` is the one place sections are guaranteed to be wired. It
imports every panel — which runs any self-registration — and then fills in
anything still unregistered from a `Record<SectionId, PanelMount>` table.
Because that table is exhaustive over `SectionId`, adding a section without
wiring a panel is a compile error rather than a dead nav item. Self
registration wins where both exist, so a panel that deliberately registers
something other than its headline export keeps that choice.

Two things the shell guarantees, so panels do not have to:

- **The host is reset between sections.** Panels are free to set `el.id` and
  add their own class for stylesheet scoping; `clearMount` restores
  `#dash-main` and its class on the way out, so one section's CSS never paints
  another's markup.
- **A section always renders.** An unregistered route falls back to a titled
  stub instead of a blank page.

### Checking the wiring

```bash
npm run check --prefix packages/viewer
```

That is `tsc --noEmit` plus `scripts/check-dashboard.mjs`, which asserts the
shell files exist, that `SectionId`, the nav and the panel table agree, that
every section resolves to a panel file exporting the mount it names, and that
the Vite `/api` proxy still targets `:8765` with WebSockets enabled. It parses
sources rather than importing them, needs no browser and no server, and runs
in well under a second. `npm run check:dashboard --prefix packages/viewer`
runs it alone.

The failures it exists to catch are the ones that compile perfectly: a section
added to `SectionId` that nobody put in the nav, a nav item with no route
behind it, or a proxy edit that quietly points the panels at nothing.

## Fixed visual benchmarks

`?benchmark=<id>` selects the named replay, exact recorded timestamp and fixed
camera, pauses playback, and deterministically reconstructs particles, grime
and tyre tracks. These are the comparison views for later visual work:

Each timestamp is the measured extreme of what it gates, read out of the
recorded frames rather than picked by eye:

| id | replay / time / camera | gate |
|---|---|---|
| `corner-roll` | proving ground / 8.7000 s / chase | ordinary roll and lateral spring load |
| `landing-pitch` | proving ground / 22.0333 s / chase | peak landing load, 23,615 N |
| `airborne-droop` | proving ground / 21.2333 s / sideline | recorded apex, 3.243 m, droop and separated shadow |
| `gravel-spray` | demo / 48.4667 s / sideline | peak cornering slip, 0.110 rad at 31.1 m/s |
| `launch-wheelspin` | demo / 2.8000 s / chase | peak gravel slip ratio, 0.336 at full throttle |
| `tarmac-skid` | proving ground / 22.1333 s / sideline | trail laid through the 0.833 slip-ratio lock-up |

For example:

```text
http://localhost:5173/?benchmark=airborne-droop
```

Recapture the checked-in baselines with:

```bash
npm run benchmarks --prefix packages/viewer
```

That script owns the 1184×757 viewport and reads the routes from
`src/benchmarks.ts`, so a comparison can never drift on framing. `--verify`
shoots each route twice and fails on any pixel difference.

## Frame budget

Dev builds expose `watch.frameBudget` (rolling mean + p95 over the last 120
frames). Measured at the six benchmark routes, paused, 1184×757:

| route | mean (ms) | p95 (ms) |
|---|---:|---:|
| `corner-roll` | 1.34 | 2.40 |
| `landing-pitch` | 1.46 | 2.40 |
| `airborne-droop` | 1.31 | 2.00 |
| `gravel-spray` | 1.40 | 2.70 |
| `launch-wheelspin` | 1.34 | 2.60 |
| `tarmac-skid` | 1.40 | 2.50 |

Worst p95 is 2.70 ms against a 16.6 ms budget. Re-measure with
`npm run frame-budget --prefix packages/viewer`.


## Procedural audio

Everything is synthesised with Web Audio at runtime — there is not a single
sample file. `src/audio/RallyAudio.ts` is the facade `main.ts` talks to; it
owns five layers on a shared bus (`AudioBus`: master + per-layer gain splits,
autoplay unlock on the first gesture, persisted mute):

| layer | source | driven by |
|---|---|---|
| `EngineAudio` | saw/square/triangle oscillator stack + band-passed noise | RPM, throttle, brake, boost; gear changes duck the stack and fire a shift blip; a throttle lift above mid-RPM triggers overrun |
| `TyreAudio` | looped pink noise (scrub) + gated white noise (peppering) | per-wheel slip and grip from the recorded wheel frames, surface type; loose surfaces pepper with speed |
| `WindAudio` | band-passed pink noise | speed squared, slightly up when airborne |
| `AmbientAudio` | three filtered noise beds (snow / forest / tarmac) | stage biome, crossfaded on surface change |
| `CoDriverAudio` | short synthesised tone/formant sequences | the same `frame.note` text the HUD shows; direction picks the contour, severity adds syllables |

Two rules keep it honest and cheap:

- **The audio reads the same interpolated `Playback.sample()` as the
  renderer**, so RPM glides between the 30 Hz recorded frames instead of
  stepping audibly. Discrete values (gear, surface, note) come from the frame
  at-or-before, like the HUD.
- **No allocation on the audio thread.** The graph is built once; noise
  buffers are prebuilt on the main thread (`noise.ts`); every per-frame change
  is AudioParam automation (`setTargetAtTime`) on native nodes. There is no
  ScriptProcessor and no AudioWorklet `process()` to allocate in.

Pausing ducks the master bus to silence rather than droning the engine at a
frozen RPM; seeking resets the co-driver's call memory so notes fire again.

`RallyAudio.update()` is timed every frame (`watch.audio.costSample`, also
`__rallyAudioCost`). Measured over 10 s of real playback per shipped replay,
audio unlocked, 1184×757:

| replay | audio update ema (ms) | share of 16.67 ms |
|---|---:|---:|
| `demo` | 0.050 | 0.30% |
| `proving_ground` | 0.045 | 0.27% |

Re-measure with `npm run audio-cost --prefix packages/viewer`.

## Camera director

By default WATCH is "directed": `src/camera/director.ts` builds a cut list
once per replay, purely from the recorded pace notes and frames — sideline for
jumps and crests, tight bonnet through severity ≤ 2 corners, the chase boom
held through fast sweepers and everything else. Shots have minimum length and
spacing so the edit never strobes, and a cut re-seeds the chase damping so it
reads as a cut rather than a boom swing. Deterministic by construction: the
same replay always cuts at the same timestamps, and nothing feeds back into
playback. Camera vibration, slip lag and landing kick are untouched — they run
inside `ChaseCamera.apply()` in every mode. Press `c` to leave the director
for a fixed camera; `?benchmark=` routes always pin their own camera and never
direct.

| key | |
|---|---|
| `space` | play / pause |
| `←` `→` | seek 2 s (`shift` for 10 s) |
| `[` `]` | playback rate, 0.1x to 8x |
| `c` | camera: auto (director) → chase → bonnet → sideline |
| `m` | mute / unmute audio (persisted) |
| `p` | toggle the restrained retro colour pass |
| `r` | restart |

## What it is not, yet

- **TRAIN and the dashboard exist now.** The trainer and control room landed,
  so the studio (`?mode=train`) and the program dashboard
  (`?mode=dashboard`) read a real metrics stream instead of an invented one.
  WATCH is still the only surface that needs nothing running.
- **Geometry-led early 3D art direction.** The viewer renders sharply, while
  flat-shaded faceted cars, terrain, trees and effects carry the late-1990s
  character. The post-process is deliberately restrained: it no longer emulates
  a visibly enlarged 320×240 framebuffer.
- **Audio is synthesised, not sampled.** Engine, tyres, wind, ambience and
  co-driver calls are all Web Audio synthesis driven by recorded RPM, boost,
  gear, slip, surface and pace notes. See "Procedural audio" above.
- **Wheel presentation has one source.** A shared four-wheel visual state
  contains road-projected position, contact, compression, load, slip, speed and
  surface. Shadows, spray and tracks consume it rather than guessing from the
  car centre.
- **Loose-surface spray is viewer-derived.** Separate wheel emitters use
  recorded contact, speed, load, slip and surface. Braking and throttle select
  the presentation response; seeking clears and deterministically prewarms the
  bounded pools.
- **Loose-surface spray is visible.** Diffing each spray benchmark with
  emitters shown vs hidden (pixels >8/channel): `launch-wheelspin` 5.02%,
  `gravel-spray` 6.80%, `corner-roll` 6.51%. Emission gating is unchanged;
  particles rise 1.5–3 m, live longer, grow larger, and contrast above road
  albedo. See `docs/visual-benchmarks/README.md`.
- **Tracks are replay-safe.** Gravel disturbance, tarmac skid marks, mud ruts
  and compressed snow are rebuilt from recorded 30 Hz wheel frames after every
  seek and hard-capped at 1,400 segments.
- **Car-state input boundary is explicit.** Brake lights use recorded brake
  input; exhaust cadence uses recorded RPM and throttle; damage appears only
  for explicit impact/crash/contact events. Anti-lag is a viewer-only
  derivation from a throttle drop above 3,800 RPM, and grime is a viewer-only
  accumulation of recorded distance weighted by recorded surface.
- **Camera dynamics are render-only.** Surface vibration uses replay time,
  speed and surface; lateral lag uses recorded chassis slip; landing kick uses
  explicit landing events. FOV gain is capped at 3°. No camera value feeds
  playback or simulation.
- **Suspension presentation stays replay-led.** The sprung body pivots about the
  CarSpec CG while the wheels remain on the recorded road plane. Direct
  per-wheel compression is used when present; existing replays fall back to a
  conservative bump/droop reconstruction from their recorded vertical loads.
- **Human drive is server-owned.** The control room runs the sim loop at 30 Hz
  and writes the replay; the browser sends input and renders frames it is
  handed. It is what milestone 6's human baseline depends on, and it lives at
  `?mode=dashboard#/drive`.
- **Reference-led personal livery.** Deep blue, large yellow 555-era graphics,
  Subaru marks, gold wheels and gravel grime stay bold and immediately readable.

## Layout

| file | |
|---|---|
| `src/coords.ts` | **The only place axes convert.** Sim is z-up; three is y-up |
| `src/replay.ts` | The replay/stage contracts as types, plus structural checks |
| `src/playback.ts` | Clock, seeking, frame interpolation |
| `src/benchmarks.ts` | Fixed replay times and cameras for visual regression |
| `scripts/capture-benchmarks.mjs` | Boots the server and writes every baseline PNG |
| `src/scene/stage.ts` | Corridor, verges, terrain skirt, start/finish |
| `src/scene/car.ts` | The car, sized from `evo_rally`'s `CarSpec` |
| `src/scene/surface.ts` | Shared sampled road plane and four wheel visual states |
| `src/scene/shadows.ts` | Road-aligned car and tyre grounding shadows |
| `src/scene/dust.ts` | Pooled per-wheel spray and fragments |
| `src/scene/tracks.ts` | Bounded replay-reconstructed tyre marks |
| `src/scene/carEffects.ts` | Brake, exhaust, anti-lag, grime and event damage |
| `src/scene/trees.ts` | Textured near vegetation, far-fog cards, rocks and props |
| `src/render/ps1.ts` | Restrained colour pass and optional environment wobble |
| `src/scene/textures.ts` | Every texture, drawn procedurally at load |
| `src/scene/sky.ts` | Sky dome, clouds, layered distant hills |
| `src/camera.ts` | Chase, bonnet, sideline |
| `src/camera/director.ts` | Deterministic replay-time camera cuts |
| `src/audio/` | Procedural engine, tyres, wind, ambience, co-driver |
| `scripts/measure-audio-cost.mjs` | Audio update cost share of the frame budget |
| `src/hud.ts` | Stopwatch, speed, gear, progress, pace call |
| `src/main.ts` | Picks the surface: WATCH, `?mode=train` or `?mode=dashboard` |
| `src/dashboard/` | Program dashboard shell (nav, router, panel registry) |
| `src/dashboard/panels/index.ts` | Section → mount table; exhaustive over `SectionId` |
| `src/dashboard/panels/` | One file per section, plus its stylesheet |
| `src/train/` | TRAIN studio and the control-room clients panels reuse |
| `public/dashboard/` | Committed measured data so panels work with no server |
| `scripts/check-dashboard.mjs` | Asserts sections, nav, panels and proxy agree |

## Two things that will bite you

**Coordinates convert exactly once, at load.** v1 stored stages in the viewer's
y-up convention and bridged on every step; that mismatch produced its
floating-car bug. If you are swizzling axes outside `coords.ts`, it is being
done twice.

**Scenery is not the stage.** `trees.ts` builds two different things: obstacles
that come from the stage document and are solid, and a forest that is invented
here and cannot be collided with. If a tree can hit you it came from the stage.
Anything you add to the scenery must stay beyond the corridor edge.

**Positive camber raises the right edge**, which is what banks a road into a
left-hander. The sim shipped with this backwards at the physics bridge, so every
banked corner was played off-camber. If the car ever looks like it is leaning
*out* of banked corners, that bug is back — and it is not a renderer bug.
