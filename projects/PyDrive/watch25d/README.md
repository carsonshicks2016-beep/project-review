# Fable Five Watch 2.5D (Phase 5)

Browser sibling of Observatory — **not** a skin inside `observatory/`.
Phase 5: cinema/arcade polish — harder PS1 snap + CRT finish, weather as
palette only, denser roadside props, and `?audio=1` auto-enable after the
first page gesture. Still driven entirely by Python-truth
`fable-observatory-v1` frames.

## Open

With Command Center on `:8770` and this client built:

```text
http://127.0.0.1:8770/watch25d/?edition=919
http://127.0.0.1:8770/watch25d/?edition=787b
http://127.0.0.1:8770/watch25d/?edition=919&audio=1&weather=rain
```

From the dashboard: **Fable Five → Superhuman Watch → Watch Best / Selected 2.5D**
(mirrors the 2D / 3D buttons; opens `/watch25d/?edition=…&checkpoint=…`).

| Param | Default | Notes |
| --- | --- | --- |
| `edition` | `919` | `919` or `787b` (catalog ids; also accepts car hints) |
| `car` | _(empty)_ | optional override (`porsche_919evo` / `mazda787b`) |
| `checkpoint` | `active-best` | registry selection |
| `mode` | `replay` | or `follow-active-best` |
| `seed` | `7` | |
| `audio` | off | `?audio=1` arms sound — first click/key after session up auto-enables once |
| `weather` | `clear` | presentation palette: `clear` · `dusk` · `night` · `rain` |

### Keys

| Key | Action |
| --- | --- |
| `C` | Cycle chase ↔ broadcast camera |
| `A` | Toggle audio (same as **AUDIO** button) |
| `G` | Cycle weather palette (clear → dusk → night → rain) |
| `V` | Cycle CRT overlay (mild → hard → off) |

## Audio

- Default **off** — no Autoplay-policy fight at boot.
- Session still allocates an FOA1 socket so the toggle can connect later; the
  browser FOA1 websocket is created only on first enable.
- With `?audio=1`, the first pointer/key gesture after the session is up
  auto-enables audio once (no second **AUDIO** click required).
- Manual **AUDIO** / `A` still works anytime.
- Listener pose is uploaded on the telemetry WS; Doppler stays server-side.
  Never invent client pitch shift.

## Weather (presentation only)

Fog density + sky/asphalt tint (+ cheap camera-local rain streaks for `rain`).
Never mutates sim physics, grip, or policy. Cycle with `G` or set
`?weather=dusk` etc.

## Build

```bash
# Refresh the published ribbon from the sim track (deterministic):
PYTHONPATH=$PWD python3 watch25d/tools/build_track.py
PYTHONPATH=$PWD python3 watch25d/tools/build_track.py --check

cd watch25d
npm install
npm run build
```

Serve happens via Command Center (`/watch25d/` → `watch25d/dist`).
Static Command Center JS/HTML is no-cache — refresh the browser after dashboard edits.

## Architecture

- **Protocol:** same `fable-observatory-v1` sessions / WS as Observatory.
  Does not replace or hijack `/observatory/`.
- **Track:** static `public/assets/track/nordschleife_ribbon.json` generated
  from `named_track('nordschleife')` (centerline + half-width + road_z +
  heading). Presentation only — car height still comes from frame `road_z_m`.
- **Dressing:** Armco (~5.5 m), denser instanced trees (incl. occasional
  second row), billboard ads, marshal posts with flat flags. Short FogExp2.
- **Car:** procedural PS1 proxies — `src/vehicle-919.js` (white/red LMP) and
  `src/vehicle-787b.js` (Renown green/orange coupe). Kind from
  `?edition=` / `?car=` or session hello checkpoint car.
- **HUD:** fat digital SPD / GEAR / PROG% + episode time; CONN / CAM / WX /
  CRT / SEQ + AUDIO toggle. Not Observatory chrome.
- **Audio:** copied FOA1 bridge (`src/audio.js` +
  `public/audio/pcm-jitter-processor.js`) — same path as Observatory.
- **Look:** flat/vertex colors, nearest-neighbor curb stripes with affine-ish
  UV wobble, strong vertex-snap shader, mild/hard CRT scanline+dither overlay
  (`V`), `antialias: false`, capped DPR. No PBR, no bloom, no pace heat
  ribbon on asphalt.
- **Coord map:** Three `(x,y,z) = (sim.x, sim.z, -sim.y)`.

## Out of scope (v1 complete for the locked brief)

Still not in this sibling (by design):

- Observatory DGM terrain tiles / cinema GLBs
- Pace heat on road
- Mid-session server FOA1 allocation without create-time `audio: true`
  (client socket already deferred until enable)
