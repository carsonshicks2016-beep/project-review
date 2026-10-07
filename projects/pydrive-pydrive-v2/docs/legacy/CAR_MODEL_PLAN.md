# CAR MODEL PLAN — A Real Mk4 Supra for the 3D Viewer

**Mission:** replace the deliberately-blocky voxel Supra in the v2 viewer with
a car that reads as an actual **A80 Mk4 Supra** — correct silhouette, real
curves, recognizable from every camera — while still sitting naturally in the
clean flat-shaded low-poly world.

**Decision (owner, 2026-06-12):** this supersedes VIEWER3D_V2_PLAN §1 rule 3
("voxel hero car — the toy-car contrast IS the charm"). The world keeps its
style; the car graduates from toy to hero. The voxel car is preserved as a
fallback/easter egg, never deleted.

**Status: reference contract FILLED (owner pics, 2026-06-12) — building.**

---

## 1. The reference contract (FILLED)

Four owner pics (in the build conversation, 2026-06-12): **side** (blue,
lowered, factory basket-handle wing, lip + skirts, gunmetal multi-spokes,
red calipers) · **rear** (red, the quad-round taillight bar, single fat
oval exhaust, diffuser) · **front** (white, stock — quad projectors under
glass, nose badge, twin bumper slots, front lip) · **top** (blue render —
glasshouse plan, hatch glass, mirror placement).

**Style calls (owner):**
- Trim: **mild tune** — stock body (NOT the rear pic's widebody) + front
  lip + side skirts + lowered stance per the SIDE pic (the master reference)
- Wing: **factory basket-handle, body color** ✓
- Color: **deep metallic blue** per the side/top pics (one-line config —
  easy to swap later); dark glass, gunmetal wheels, **red caliper accent**
  visible through the spokes
- Finish: **SMOOTH / realistic** — smooth-shaded curves with crisp creases,
  glossy paint. NOT flat-faceted. The car is the hero against the low-poly
  world (this revises §3's facet language; the world's palette discipline
  still applies so it sits in the scene).

**The A80 recognition checklist (gates score against the pics):**
long curved hood falling to a low wide nose · fixed quad projector
headlights under glass covers · big rounded fenders front AND rear (the
haunches) · fast double-curve roofline into the short high tail · the
wraparound bar with FOUR ROUND taillights · basket-handle wing · deep round
arches over multi-spoke wheels · single fat oval exhaust, left of center ·
twin bumper intake slots + lip up front.

## 2. Pipeline decision: Blender-authored glTF (not more code-boxes)

Three candidate pipelines, one winner:

| Pipeline | Verdict |
|---|---|
| **Hand-coded Three.js geometry** (status quo, more boxes/lathes) | ❌ real automotive curves (arches, roof, haunches) are miserable to author vertex-by-vertex in code; iteration is blind |
| **Downloaded/AI-generated model** (Sketchfab / Hyper3D / Hunyuan via the Blender MCP) | ⚠️ backup only — license risk (Sketchfab) or messy topology + non-flat shading (AI gen); could seed a blockout to trace over |
| **Hand-modeled low-poly in Blender → .glb → GLTFLoader** | ✅ THE PLAN — real curves with full control, mirror-modifier box modeling is fast for cars, the Blender MCP lets the build iterate against the inspiration pics with viewport screenshots, and a .glb is a static asset (no build step, fits the vendored-modules architecture) |

Blender is already connected via MCP (model, screenshot, export — all
scriptable from a session). The asset ships at
`viewer3d/static/assets/supra_a80.glb`; `GLTFLoader.js` (matching the
vendored three release, single file, no Draco) joins
`viewer3d/static/vendor/`.

## 3. Target spec

- **Budget:** ≤ ~15k triangles total (smooth curves need more than facets:
  body ~9k, wheels ~800 each shared geometry, glass/details the rest). The
  world runs 60–90 draw calls; the car adds ≤ 8. 120 fps stays untouched.
- **Style (revised by §1):** SMOOTH-shaded body with crisp edge creases
  (auto-smooth / split normals on panel lines), **glossy paint** — Phong or
  Standard material with strong specular off the sun/moon directional, and a
  cheap gradient envmap baked from the sky dome colors if the paint needs
  more life (no PMREM of the live scene). ≤ 6 materials: body paint, dark
  glass, dark trim/rubber, gunmetal wheels, red calipers, lights. No photo
  textures. Decals (windshield banner, door roundel) stay canvas textures on
  thin overlay planes, reused from the voxel car.
- **Recognition over realism:** every gate is scored against §1's checklist
  from the actual game camera (chase, low, car small in frame) — detail that
  doesn't survive that framing doesn't get modeled.
- **Conventions (must match the sim):** forward **+X**, up **+Y**, left
  **−Z** in three-space; origin at ground level mid-wheelbase; real meters
  (A80: 4.51 L × 1.81 W × 1.27 H, wheelbase 2.55 — matches CarSpec).

## 4. Integration contract (zero churn outside supra.js)

`createSupra(mood)` keeps its exact return shape — `{ group, wheels, blob }`
— so app.js doesn't change:

- **Named nodes in the .glb**: `wheel_FL/FR/RL/RR` (pivot at hub center;
  loader wraps each in steer/roll pivots exactly like the voxel wheels),
  `body`, `glass`, `taillight_bar`, `headlights`, `wing`.
- **Mood bindings re-wired by material name**: body color set per mood,
  emissive taillight bar + halo Points (kept from voxel car), headlight road
  pool (kept), contact shadow blob (kept — it's world-side now anyway).
- **Async load**: GLTFLoader is async; `createSupra` returns the group
  immediately and populates it on load (the WS handshake takes longer than a
  200 KB glb).
- **Fallback:** load failure OR `?car=voxel` → the current voxel builder
  (file renamed `supra_voxel.js`, untouched). Rollback is a query param.

## 5. Build stages (each ends with a screenshot gate vs the pics)

### Stage A — Blockout *(Blender, via MCP)*
Reference images on background planes (side/front/top). Mirror-modifier box
blockout: main body mass, glasshouse, wheel cutouts. Proportions locked
against the side profile — wheelbase/overhang/roof-peak positions measured,
not eyeballed.
**Gate A:** viewport side-by-side with the side-profile pic; silhouette
matches before ANY detail.

### Stage B — The body
Arch flares (front + the big rear haunches), hood curve to the low nose,
double-curve roof, tail with the wraparound light bar recess, bumpers,
skirts, mirrors, wing per §1, exhaust. Then the low-poly pass: decimate to
budget, **crease the panel lines, split normals for flat facets**, vertex
colors / material slots (body, dark trim, glass, lights, chrome-ish).
**Gate B:** front-3/4 + rear-3/4 viewport renders vs the pics; §1 checklist
scored; tri count ≤ budget.

### Stage C — Wheels + export plumbing
5-spoke wheel (one geometry, 4 placements), named nodes per §4, origin/axes
per §3, real-meter scale check against CarSpec wheelbase/track. Export
`supra_a80.glb`; verify in a bare three.js scene snippet.
**Gate C:** glb loads clean, nodes found by name, dimensions within 2 cm.

### Stage D — Viewer integration
Vendor `GLTFLoader.js`; new `supra.js` = loader + pivots + mood bindings +
lights/decals/blob carry-overs; voxel builder → `supra_voxel.js` +
`?car=voxel` fallback path; async-load placeholder handling.
**Gate D:** drive club + ridge, both moods, all 3 cameras: taillights glow at
night, wheels steer/spin, decals show, shadow detaches over jumps, zero
console errors, fps unchanged.

### Stage E — Polish + ship
Mood-tuned body colors (night needs a lifted emissive floor like the voxel
car had), pitch/roll attitude sanity at speed (the new body shows attitude
far better than boxes — check it doesn't clip the road on hard landings),
QA matrix (named tracks × moods × cameras), README/HANDOFF/plan notes.
**Gate E:** owner side-by-side approval vs the inspiration pics → tag as the
new baseline; voxel stays reachable.

## 6. Risks

| Risk | Mitigation |
|---|---|
| Model reads "realistic" and clashes with the low-poly world | flat-shaded facets + the world's own palette discipline; Gate B judges IN the world, not in Blender |
| Async glb load races the first state packet | group returned sync, content attached on load; placeholder = nothing visible for ~100 ms (fine) |
| Wheel pivots misaligned → wobble at speed | hub-centered origins in Blender, verified numerically at Gate C |
| GLTFLoader version mismatch with vendored three | take the loader from the exact same three release as the vendored module |
| Scope creep into rx7/skyline/etc. | Supra only; the loader is generic so other cars become "just add a glb" later |
| The blocky charm is missed | `?car=voxel` forever |

## STATUS TRACKER

- [x] §1 reference contract filled (2026-06-12: side/rear/front/top pics in;
      mild tune, basket-handle wing, metallic blue, SMOOTH/realistic finish)
- [x] Stage A — Blender blockout (2026-06-12: parametric 17-station loft,
      silhouette matched vs the side pic)
- [x] Stage B — Body detail (2026-06-12: arches boolean-cut + dark tubs,
      glasshouse material zone, 5-bolt wheels w/ rim+red-caliper children,
      quad-round taillight bar, oval exhaust, headlight glass covers, bumper
      slots + front lip, rocker skirts, door mirrors, basket-handle wing
      [blade uprights + spanning foil]. 9766 tris / 11 mats — under budget.
      Reads as an A80 from all 4 views. NITS for Stage C/E: glass-zone edge
      is subsurf-jagged (needs real panel-line geo), headlights are simple
      ovals, wheels are solid discs [no spokes yet])
- [x] Stage C — Wheels + export (2026-06-12: 5-spoke wheels joined to ONE
      node each [tire+barrel+hub+spokes], origin at hub, axle +Y; calipers
      split to static brake_* nodes; supra_a80.glb exported Y-up to
      viewer3d/static/assets/ [308 KB]. Gate C verified by re-import: all
      4 wheel nodes + body found by name, L=4.59 m [body on-spec, extra is
      wing/lip overhang], wheel origins at hub z=0.315 on the axles, all 11
      materials intact, nose at +X)
- [x] Stage D — Viewer integration (2026-06-12: vendored GLTFLoader r160
      [+ gltf_bufferutils helper], patched to local imports. supra.js rewrote
      as the glb loader: Box3 auto-fit [nose +X, scale to 4.5 m, grounded],
      wheels pulled into steer/roll pivots via attach(), paint/taillight/
      headlight materials wired to mood, shared night-glow effects + contact
      shadow kept. Voxel car preserved as supra_voxel.js behind ?car=voxel
      (also the load-failure fallback). REAL HEADLIGHTS added: emissive
      lenses + two SpotLights lighting the road at night (low decay so the
      beam carries), mood-gated off by day. Verified live: drives club day +
      night, correct facing/scale/grounding, wheels turn, taillights glow,
      twin headlight beams on the road, 120 fps, zero console errors)
- [ ] Stage E — Polish, QA matrix, owner approval, ship
