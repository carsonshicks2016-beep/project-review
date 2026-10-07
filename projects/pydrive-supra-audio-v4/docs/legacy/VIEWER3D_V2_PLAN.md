# 3D Viewer V2 — From-Scratch Build Plan

**Mission:** rebuild the browser 3D viewer from scratch as **3D v2**, matched to
the two inspiration frames (night touge with mist + amber lamps; alpine day
pass with snow peaks + tunnel). Keep the existing viewer fully working as
**legacy**. Redesign the car as a **voxel Mk4 Supra** with optional decals.
After the scenery ships, the next big push is **physics correctness**
(elevation gain/loss + pitch/yaw/roll) for the viewer AND PPO/GA training —
outlined here as Phase 8, detailed in its own doc when we start.

This doc supersedes `SCENERY_PLAN.md` (which planned incremental v1 patches —
deleted; v1 is now frozen as legacy).

---

## 1. The aesthetic contract (read the pics, not vibes)

Two target frames, both chase-cam, car low in frame, road dominating:

**Frame N (night):**
- Deep blue sky, very few stars (one bright). Horizon slightly lighter.
- 2–3 mountain silhouette layers, **soft mist banks rolled between them**,
  pines descending the valley slope into the mist.
- Warm sodium streetlamps on curved-arm poles **tracing the road's curve into
  the distance** — the only saturated warmth in an all-cool scene.
- Guardrails both sides catching lamplight; rocky cliff on the right warmly
  lit; single yellow centerline leading to a vanishing bend.
- Road surface readable: subtle asphalt texture, faint sheen under lamps.

**Frame D (day):**
- Vivid blue sky; jagged **snow-capped peaks** — snow sits on the flatter
  facets, gray rock shows on steep facets. Ridges fade lighter/bluer with
  distance (aerial perspective).
- Faceted gray cliff walls; **stone masonry retaining wall with snow on top**;
  **tunnel portal** in the rock face ahead (focal destination).
- Pines: dark multi-tier conifers, scattered on slopes and perched on cliff
  tops. Snow patches on shoulders. Candy-stripe delineator posts.
- Soft sun shadows (car + posts cast onto road).

**Style rules derived from the pics:**
1. **Clean, not retro.** No scanlines, no dither quantize, no chromatic
   aberration, no 0.56 render scale. Full-res render, antialiasing ON, soft
   shadows, gentle color grade only. (Legacy keeps the CRT look; v2 is the
   clean stylized look.)
2. **Flat-shaded low-poly world.** Faceted terrain/rock/peaks via flat
   normals + vertex colors. Textures only where they earn it (asphalt,
   masonry). Detail comes from geometry + palette, not texture noise.
3. **Voxel hero car.** The Supra is deliberately chunkier than the world —
   built from boxes, glowing rectangle taillights, white wheel caps. The
   contrast (detailed world / toy car) IS the charm.
4. **One warm accent at night** (lamps + taillights). Everything else cool.
5. **Detail gradient**: dense near the road corridor, bold and simple far away.

**Palette anchors (sampled off the pics, tune in Phase 7):**
- Night: sky `#0b1220→#2a3a50` (zenith→horizon), mist `#5a6b7d`, lamp
  `#ffb257`, road `#3a3b3d`, cliff lit `#6b5d49`, pine `#16241c`.
- Day: sky `#4a7fc4→#bdd2e2`, snow `#e9eff4` (shadow `#c2d2e0`), rock
  `#7d7f7c` / dark `#55585a`, pine `#2c4a32`, road `#4b4d4f`, masonry
  `#9a958a`, marker red `#c33`.

---

## 2. Architecture

### 2.1 Legacy split (Phase 0)
- `viewer3d/legacy/static/` ← the pre-v2 working tree (the day/night-capable
  viewer the owner was iterating on), with the unapproved 2026-06-09 sky
  experiments reverted edit-by-edit before the move. `graphics_v1/` (an older
  approved snapshot, pre day/night) stays as a second archive. `vendor/`
  copied in.
- `viewer3d/static/` ← emptied, becomes v2's home (fresh files).
- `viewer3d/routes.py`: `/3d/` serves v2 · `/3d/legacy/` serves legacy ·
  both share the existing WS bridge `/api/3d/ws` (sessions are
  per-connection; zero backend changes).
- Command Center: "Open 3D Viewer" → v2; add a small "legacy" link.
- `graphics_v1/` snapshot stays as the immutable archive.

### 2.2 Backend (unchanged this phase)
`viewer3d/session.py` keeps running real `supra` physics at fixed dt and
streaming 30 Hz state. v2 consumes the same payloads. **Interface rule:** the
client never invents elevation — it uses the `z` the server sends. Today
that's the fake sine profile (`visual_elevation`); in Phase 8 it becomes real
track elevation and v2 re-drapes with zero renderer rework. Same for car
pitch/roll: v2 derives them from road sampling for now, but reads
`pose.pitch/roll` the moment the server provides them.

### 2.3 v2 frontend module map (plain ES modules, no build step)
```
viewer3d/static/
  index.html            shell + HUD
  style.css             HUD styling (new, cleaner; keep the terminal-green soul)
  app.js                boot + frame loop + net + input (thin orchestrator)
  vendor/three.module.min.js
  v2/
    palette.js          ALL colors/intensities, day+night "mood" configs
    materials.js        material factory, registered for mood lerping
    sky.js              gradient dome shader, sun/moon, stars, cumulus
    corridor.js         track spine: frames, arc param, side assignment,
                        curvature classes (straight/bend/hairpin)
    terrain.js          carved heightfield, vertex-color zones, facet shading
    mountains.js        peak rings, slope-based snow, silhouette strips
    atmosphere.js       fog config, valley mist banks, aerial tint
    road.js             surface, lane lines (geometry), shoulders
    furniture.js        guardrails, lamps, delineators, chevrons, signs,
                        retaining walls, tunnel portals
    vegetation.js       multi-tier pines, rocks, placement rules
    supra.js            voxel Mk4 + decals + lights + wheels
    camera.js           chase/cinematic/hood rigs
    hud.js              DOM wiring
```

### 2.4 Day/night as a "mood" system
One config object per mood (night/day) holding every tunable: sky uniforms,
fog color/density, light intensities, material color/emissive sets, mist
opacities, lamp glow, post grade. Toggle = **lerp between moods over ~1.5 s**
(no hard swap). Adding dusk later = adding one config. This replaces legacy's
200-line `applyMaterialTimeOfDay` if-chains with data.

### 2.5 Performance budget
Target 60 fps on this Mac at full res (legacy hit 120 Hz at 0.56 scale with a
heavier post chain — headroom exists). Rules: instanced or merged geometry
only (no per-object meshes in loops); chunked frustum culling along
arc-length for corridor furniture; max ~8 active point lights (nearest-N
lamps) + 2 shadow-casting directionals→1; shadow map 2048 follow-cam focused;
`renderer.info.render.calls` checked at every phase gate (budget ≤ ~120).

---

## 3. Build phases (each ends with a screenshot gate vs the pics)

### Phase 0 — Scaffold + legacy preservation  *(small)*
Legacy split per §2.1. v2 boots: renderer, sky placeholder, WS connect, flat
road ribbon from track payload, drivable with keyboard, HUD live.
**Gate:** `/3d/legacy/` is pixel-identical to the approved v1 look ·
`/3d/` drives on a flat gray world at 60 fps.

### Phase 1 — Corridor spine + terrain massing
- `corridor.js`: arc-length frames (pos/tangent/normal), curvature classes,
  **side assignment**: cliff-side vs valley-side per stretch (from curvature
  sign + noise, like a real touge cut into a slope; swaps at saddle points).
- `terrain.js`: heightfield carved from the road outward — valley drops away
  on the valley side (benches → slope → floor), mountain mass rises on the
  cliff side. fbm + ridge noise; the road sits in a graded cut. Vertex-color
  zones (grass/scree/rock/snow-ready) + flat facets.
- Faceted cliff walls where the cut is steep (displaced ribbon, sharp creases).
**Gate:** day+night orbit screenshots show believable massing; no z-fighting;
60 fps; terrain conforms exactly under road (no floating/clipping).

### Phase 2 — Sky, light, atmosphere
- `sky.js`: gradient dome (pic-matched stops), night: sparse stars + one hero
  star + moon w/ soft halo; day: sun disc + glow + chunky cumulus (fbm puffs,
  planar projection).
- Lights: day = warm sun directional (soft shadows) + blue hemisphere;
  night = cool moon directional (faint shadows) + dark hemisphere.
- `atmosphere.js`: distance fog tuned per mood + **valley mist banks** — soft
  alpha-gradient billboard banks seeded in valley pockets *between* terrain
  ridgelines, slow drift. Aerial-perspective tint on far terrain/peaks
  (authored per distance ring, not just fog).
**Gate:** sky/haze read matches both pics side-by-side.

### Phase 3 — Mountains + snow
- `mountains.js`: 2–3 rings of low-poly peak meshes (displaced cones/ridge
  strips, flat facets). **Snow by facet rule:** vertex color = snow where
  facet slope < threshold AND altitude > snowline, rock otherwise — this is
  exactly the day pic's peak look. Night: peaks go silhouette-dark with the
  snow faintly moonlit.
- Far ring: jagged silhouette strips, pre-tinted toward sky (cheap, no fog).
**Gate:** day = "jagged snow peaks against blue"; night = layered silhouettes
with mist between (with Phase 2 banks).

### Phase 4 — Road + furniture (the night money shot)
- `road.js`: asphalt material (subtle texture + lane-wear darkening), solid
  yellow centerline + white edge lines as **geometry ribbons** (crisp at any
  res), gravel shoulders.
- `furniture.js`:
  - Guardrails: real W-beam profile extruded along edges, posts, slight
    per-post jitter, end terminals; catches lamp/sun light.
  - Streetlamps: pole + **curved arm** + head (pic 1 silhouette), additive
    glow sprite, elongated road pool, nearest-N point lights; spacing irregular
    (clusters near bends).
  - Candy-stripe delineator posts; chevron boards on hairpins (from corridor
    curvature classes); km stones.
  - **Stone retaining walls** (masonry texture + snow cap, day) on steep
    cliff-side stretches; **tunnel portal** (stone arch + dark bore + interior
    amber lights) at the best cliff pinch — road passes through, visual only.
**Gate:** night frame ≈ pic 1 (lamp string leading the eye, readable road,
yellow line to the bend); day shows wall/posts/rail like pic 2.

### Phase 5 — Vegetation + set dressing
- `vegetation.js`: pines = trunk + 3 stacked cones, per-instance scale/hue
  jitter, snow-dusted variant (day, by altitude). Placement rules: dense bands
  descending the valley side (into the mist), clusters on cliff tops and rock
  shelves, exclusion zone along the road, thinning with altitude until snow.
- Boulders/talus at cliff feet; rare dead tree / stump accents.
**Gate:** pic 1's "pines into the mist" + pic 2's "pines on the rocks".

### Phase 6 — The voxel Mk4 Supra
- `supra.js`, built from boxes on purpose: long hood, set-back black-glass
  canopy, wide rear haunches, **basket-handle rear wing**, ducktail, big
  glowing rectangle taillights (emissive + sprite glow), headlight squares +
  night beam cones + road pool, white-cap wheels (steer + spin), side mirrors,
  exhaust tip, contact shadow blob.
- **Decals via small canvas textures:** windshield banner ("SUPRA"), door
  number roundel, optional side stripe / "TRD"-style tail text. Decals are a
  config list → easy to toggle/extend.
- Body shading: per-face palette variation (top panels darker maroon, sides
  bright red) like the pics — vertex/face colors, not lighting tricks.
**Gate:** chase-cam day+night car matches the pics' toy-like charm; taillights
glow like pic 1.

### Phase 7 — Polish, perf, ship
- Camera framing matched to the pics (car low-center, ~58–62° FOV, speed-pull).
- Post: subtle grade only (slight lift, warm/cool per mood, ~0.03 vignette).
  Optional later: tasteful bloom for lamps if sprites aren't enough.
- Perf pass on heaviest tracks (`pass`, `superspeed`): draw calls ≤ budget,
  steady 60 fps; chunk-cull verification.
- QA every named track + all 4 `--gen` styles + both moods + 3 cameras.
- Update README/HANDOFF + Command Center labels; archive note in
  `graphics_v1/README.md` pointing at legacy route.
**Gate:** user approval on side-by-sides; then tag this as `graphics_v2`
baseline (snapshot dir like v1's).

### Phase 8 — Physics correctness (next big push; own plan doc)
Scope agreed now, detailed in `PHYSICS_3D_PLAN.md` when scenery ships:
- **Track gains authoritative elevation + banking** (`supra/track.py`):
  per-point z, grade, camber; generators produce them; `frame()` exposes them.
  Viewer's fake `visual_elevation()` in `session.py` is deleted — the server
  streams real z (and later pitch/roll), v2 re-drapes automatically (§2.2).
- **Physics 2.5D** (`supra/physics.py`): slope forces (±m·g·sinθ along
  heading), normal-load scaling (cosθ + banking component → per-axle weight
  transfer from pitch/roll), engine-vs-grade interplay, downhill brake fade
  realism optional.
- **Training-safety decision (made 2026-06-09):** the owner accepts breaking
  existing checkpoints — hills must physically impact driving/drifting/racing
  properly. So take the full path: **extend the obs vector with grade/banking
  look-ahead and retrain a new checkpoint generation.** Still ship a
  flat-track regression harness (like `tools/regression_drivetrain.py`)
  proving dynamics match the old sim when elevation ≡ 0 — as a physics sanity
  gate, not for checkpoint compatibility.
- PPO/GA: curriculum gains hill difficulty axis; eval tracks get elevation
  variants.

---

## 4. Working method

- **One phase per pass.** After each: headless captures, night + day
  (`?day=1` supported from Phase 0), compare against the pics, tune, then
  advance. Capture (server on 8770):
  ```bash
  "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome" \
    --headless=new --hide-scrollbars --use-angle=swiftshader \
    --enable-unsafe-swiftshader --window-size=900,1100 \
    --virtual-time-budget=15000 --screenshot=/tmp/v2_check.png \
    "http://localhost:8770/3d/?day=1"
  ```
- Flask serves viewer static with `no-store`, so edits show on reload; still
  bump `?v=` tags on ship.
- Legacy is the rollback at every moment — never edit `viewer3d/legacy/` or
  `viewer3d/graphics_v1/`.
- Seeds: all procedural placement keyed off track meta seed → same track
  always dresses the same way (and bugs reproduce).

## STATUS TRACKER

- [x] Phase 0 — Scaffold + legacy split (2026-06-09: v2 at `/3d/`, legacy at
      `/3d/legacy/`, shared WS bridge; v2 boots, connects, builds track,
      moods + `?day=1` work; legacy verified pixel-faithful)
- [x] Phase 1 — Corridor + terrain massing (2026-06-09: corridor gains
      curvature/classes + loop-periodic cliff/valley side signal; terrain =
      151² facet heightfield w/ vertex-color zones + merged cliff/valley
      bands that pinch at side swaps; road/lines double-sided; camera snaps
      on load/reset; `?track=` param. Gate passed via headless captures
      (club + akina, day + night); fps on real GPU pending owner check)
- [x] Phase 2 — Sky, light, atmosphere (2026-06-09: full sky shader — night
      stars/moon halo, day sun + planar-projected cumulus, mood-mixed via
      skyMix uniform; night lighting lifted so silhouettes separate; valley
      mist banks merged to one draw call, mood-bound color/opacity)
- [x] Phase 3 — Mountains + snow (2026-06-09: mid ring of merged two-tier
      jagged peak clusters, snow by facet rule [flat + above snowline]; two
      far silhouette ridge strips, unlit/fog-exempt, authored aerial tints;
      night peaks go properly silhouette-dark. ⚠️ COLOR PIPELINE GOTCHA: the
      vendored three build reads hex as LINEAR but encodes output to sRGB —
      dark values must be authored ~gamma darker than intended (see palette
      night fog comment); consider enabling ColorManagement in Phase 7)
- [x] Phase 4 — Road + furniture (2026-06-10: textured asphalt w/ lane wear;
      guardrails (band ribbons + instanced posts, skipped on wall stretches);
      curved-arm lamps (instanced) w/ one-draw-call glow Points, additive road
      pools, and a 6-light PointLight pool that follows the car; candy
      delineators; hairpin chevrons via curvClass; masonry retaining walls w/
      snow caps on strong cliff stretches; tunnel portal (bore + facades +
      interior glows) at the longest cliff pinch. Night gate: lamp + pools +
      lit rails verified in capture; wall/tunnel pending owner drive-by)
- [x] Phase 5 — Vegetation (2026-06-10: multi-tier pines merged to ONE
      instanced draw call w/ per-instance scale + hue jitter; placement by
      side signal — valley bands descending into the mist, cliff-top
      clusters, neutral scatter, road exclusion + treeline; jittered
      boulders at cliff toes/valley benches; terrain height model exported
      as terrainHeightAt so everything sits exactly on the ground)
- [x] Phase 6 — Voxel Mk4 Supra + decals (2026-06-10: one merged voxel-box
      body w/ per-face shading [maroon tops, bright sides] + slight emissive
      lift; basket-handle wing, ducktail, haunches, mirrors, exhaust; glowing
      taillights + halo Points; headlight beams + road pool (night); contact
      shadow blob; white-cap wheels; canvas decals [windshield banner, door
      roundel, tail badge] via DECALS config. Night gate = inspiration pic 1)
- [x] Phase 7 — Polish, perf, ship (2026-06-10: chase cam reframed to pic
      proportions; mood-bound exposure grade; QA matrix pass/speedbowl/tech/
      random × both moods — fixed boulder + peak placement guards for tight/
      large tracks, removed headlight beam cones (end-on additive artifact;
      road pool carries the effect); README/HANDOFF/graphics_v1 docs updated.
      Chunked culling deferred — draw calls ~60-90, within budget.
      → graphics_v2 snapshot pending owner approval)
- [~] Phase 8 — Physics: elevation + pitch/roll — **PHYSICS_3D_PLAN.md rev 2**
      execution: Stages 0–6 + 8 DONE (2026-06-12): hills + emergent jumps live
      across sim/training/viewer; obs 42→60; v2 renders server-authoritative
      car z/pitch/roll w/ AIR pill + detached shadow; Command Center = the 3D
      training environment (terrain levers, layout badges, airtime charts);
      docs/dashboard polished. REMAINING: Stage 7 retraining program (owner
      compute) — then this phase closes
