# Goal: Finish Visuals V3

## Objective

Finish the existing RallyAI3 Visuals V3 milestone in
`/Users/REVIEW_USER/RallyAI3`: a convincing early-2000s forest rally presentation,
with the approved dense/dark branch-based forest, coherent ground treatment,
a finished blue/gold sedan and cockpit, readable contact-driven effects,
smooth cameras, and coherent audio. Deliver and inspect the actual standalone
player, not just Editor screenshots. Preserve the working research platform.

Use the original full specification at
`/Users/REVIEW_USER/.codex/attachments/ce6fe86a-7255-48e7-b9ad-0a856a2893c3/pasted-text-1.txt`
and the evidence ledger `Tools/VISUALS-V3.md`. This document makes the remaining
work executable; it does not replace or reduce the original scope.

## Starting State

The user approved the denser/darker tree direction on October 4, 2026.
The kit currently covers only 190-430 m of the bare-road Crest seed 41200.
It contains four editable species and three mesh LODs, with 1,287 trees in
11 cells. Native validation found unchanged collision geometry and random
state, no decorative colliders, and stable mesh counts over four reloads.

The latest player build job `02edde02dc494b4d` completed. Runtime dust ownership
and episode-end cleanup were repaired; eight populated native reset tests and
43 Python tests passed. Trackside framing still loses the car in a real capture.
Car refinement, ground integration, audio listening, full-stage rollout and
all-family/headless/runtime validation remain incomplete.

Baseline tags `visuals-v2` and `visuals-v2-refined` must remain intact.
Find the new local review checkpoint by tag `visuals-v3-forest-review` and inspect
current Git state before editing. Preserve unrelated changes and artifacts.
No GitHub push is implied by this continuation.

## Non-Negotiable Boundaries

- Keep Built-in rendering, blue/gold identity and procedural/frozen course workflow.
- Do not alter vehicle physics, suspension/wheel positions, collision meshes,
  damage behavior, observations, actions, rewards, course hashes or training logic.
- Decorative objects must have no colliders and stay outside sensor semantics.
- Visual randomness must be isolated from course sampling and Unity global state.
- Use a 60 FPS viewer target, bounded effects and one compute job at a time.
  Do not promise thermals or a locked frame rate; report actual measurements.
- Serialize all Editor import, preparation and build jobs. Do not change assets
  while the Editor is compiling/importing/building those assets.
- Never launch training just to validate art. A brief headless smoke test is a
  specific verification step, not permission for a long training campaign.
- Strict environment/build compatibility remains authoritative. A visual-only
  intention does not justify weakening resume validation or rewriting lineage.

## Step 1: Restore and Verify the Review Harness

Why: trustworthy comparisons prevent art changes from hiding runtime regressions.

How: inspect current source, Git state, service health, active jobs and processes.
Verify course/checkpoint files against the ledger hashes. Use the actual saved
course bundle, deterministic policy, seed 2026, neutral start and simulation scale 1.
Keep baseline and changed views matched by resolution, pose, lighting and mode;
record separate source/build identities rather than claiming different source
revisions use the same build hash. Preserve all outputs in distinct folders.

Run the fixed-angle Editor comparison and standalone bounded review tool. Capture
start, straight, sharpest bend and crest, at 720p/1080p, plus rear chase,
front/rear three-quarter, side, hood and driver views. If the Crest course does
not contain a genuinely tight corner, add a separately labeled technical-course
check rather than mislabeling its shallow bend.

Record median/p95/p99 frame times, standard draw calls, available memory counters,
process memory and particle counts. Distinguish instrumented captures from normal
viewing. Acquire a short moving capture through an available supported recording
path; still images do not prove camera or LOD smoothness.

Gate: reproducible identity-bearing evidence, with missing counters shown as
unavailable and normal-viewing performance checked separately from screenshots.

## Step 2: Fix Camera Framing Before More Art

Why: the latest trackside frame misses the moving car, and camera problems obscure
both driving cues and visual review.

How: inspect SpectatorDirector and the shared render/physics pose path. Fix
trackside shot placement, occlusion and shot handover. Do not restore the previous
stair-step bumping through inconsistent timing or competing camera writers.
Check chase, helicopter, hood, driver, trackside and cinematic modes through
straights, bends, crests, airborne motion, landings and episode resets.
Favor framing and smooth pose interpolation over heavy shake or motion blur.

Gate: continuous moving verification in every mode; car/road remain framed,
handoffs and resets behave, and HUD does not overlap important cues at either size.

## Step 3: Roll Out the Approved Tree Kit

Why: the approved section still borders legacy cone forests; consistent assets
are the largest remaining stage-quality improvement.

How: separate production kit settings from seed-specific review bounds. Extend
the existing shared dressing path across frozen and generated stages without
regenerating course definitions. Retain four species, irregular crowns and
approved dark/cool palette. Tune density by slope, distance and roadside clearance.
Coordinate or replace the older distant forest so it cannot overlap or conflict
with the new near forest. Avoid abrupt section transitions and repeated clumps.

Inspect alpha edges, branch undersides, shadows, distant silhouettes and LOD
crossfades during movement, including trunk behavior. Optimize merged cell sizes,
culling and LOD complexity from measurements, not by reverting to solid cones.

Gate: full-stage chase and trackside forest looks consistent with no conspicuous
popping; all species work; repeated reloads preserve RNG/collision hashes and
bounded mesh/material counts; costs are documented at 720p and 1080p.

## Step 4: Finish One Coherent Forest Ground Treatment

Why: better trees alone leave the road and repeated bright plant clumps looking
like separate asset sets.

How: unify gravel, compacted racing line, shoulders, soil, pine litter, stones,
ferns and dry vegetation. Blend shoulders visually without changing colliders.
Use patches and several plant silhouettes rather than uniform crossed-card rows.
Keep rut/wear treatments subtle and explicitly visual. Make decorative stones
clearly distinguishable from actual road hazards. Bare-road course audits must
still show zero hazards, regardless of added art.

Gate: one finished section looks natural close-up and at speed; road boundaries
remain readable; apply that same coherent treatment through the shared scene path.

## Step 5: Finish the Sedan and Real Cockpit

Why: the existing silhouette is serviceable, but flat glazing and simple details
still break the intended early-2000s rally presentation.

How: refine existing visual mesh generation, not physics or an unvalidated shell
replacement. Improve fender/body transitions, bumpers, seams, grille depth,
integrated lamps, recessed wheel spokes/hubs and tire shoulders. Differentiate
paint, rubber, metal and lamp lenses. Use restrained dirt and panel-aware livery.
Give windows shaded glass with real interior geometry behind them. Build a
proper dashboard, instruments, seats, cage and steering wheel for driver mode.

Gate: front/side/rear three-quarter and close-up views hold up; hood/driver views
have no clipping, hollow shell or floating components; collision and wheel
placement are unchanged. Review the car/cockpit slice before unnecessary redesign.

## Step 6: Complete Effects, Not Just the Initial Repair

Why: removing duplicate dust and clearing resets is only part of the effects goal.

How: preserve single runtime dust ownership. Check actual contact/load/slip inputs
for rolling, acceleration, braking, sliding, airborne wheels and landing.
Tune fine dust separately from gravel debris. Improve the current trail ribbons
with textured, feathered marks and contact/load-driven intensity. Explicitly bound
particle counts, trail positions/lifetimes and retained memory; repeated resets
must clear both current effects and old segments without bridges across teleports.
Audit runtime-created material/texture ownership and teardown as well as meshes.
Add near-camera fading where needed without hiding useful exterior effects.

Gate: native populated reset checks plus actual repeated-episode drives; effects
communicate motion in all six cameras without hiding the car/road or leaking.
Do not treat the eight existing reset fixture checks as full lifecycle validation.

## Step 7: Listen to and Finish Audio Separately

Why: audio changed substantially before this milestone; preserve what sounds good
instead of rewriting it from source inspection alone.

How: obtain matched recordings and listen to idle, pull through gears, shifts,
lift-off/coasting, braking, sliding and landing in exterior and cockpit modes.
Compare pitch to RPM and load, check shift transients and surface tire layers,
and inspect peak levels/clipping. Eliminate clicks and implausible transitions.
Keep camera changes from causing exaggerated Doppler/pitch jumps. Replace only
layers that demonstrably remain synthetic or repetitive; track asset provenance.

Gate: actual listened recordings and a live drive, coherent transitions and
documented levels. If listening/recording tools are unavailable, record the
limitation and request a listening review; do not claim this gate passed.

## Step 8: Integrated Platform and Standalone Verification

Why: Editor art and passing unit tests do not prove the deployed research workflow.

How: rebuild serially, confirm current manifest and shader/material availability,
then inspect all six starter combinations (gentle/technical/crest, clear/obstacles)
in the actual player. Validate frozen course hashes and actual hazard audits.
Exercise repeated reloads/episodes and monitor objects, meshes, materials, tracks,
particles and memory for growth. Run a short real headless smoke job and verify
viewer-only objects/effects are absent. Preserve its artifacts and stop cleanly.

Through the dashboard, watch a specialist checkpoint with its correct default
course/settings; verify generalist selection behavior, inference-only isolation,
identity labels, viewer stopping and recorded replay distinction. Re-run Python
tests and production dashboard build. Check compatibility refusals remain honest;
do not silently change training schedules or checkpoint contracts to pass them.

Gate: evidence for native saved-course coverage, headless isolation, unchanged
training/course contracts, functioning browser-to-viewer workflow and clean shutdown.

## Step 9: Save the Actual V3 Release and Handoff

Why: the final milestone needs a reproducible delivered state, not scattered previews.

How: update `Tools/VISUALS-V3.md` with requirement-by-requirement evidence. Save
editable assets, source provenance, selected matched captures, moving/audio
recordings where available, performance JSON, build identities and test summaries.
Keep local experiment data separate from versioned visual evidence. Commit the
finished source and create a final V3 tag only after the completion audit passes.
Push only when requested. Explain remaining limitations and measured costs.

Completion: every original gate has direct evidence in the actual deployed player;
no missing audio, cockpit, camera, headless or all-family gate is hidden behind
an overall completion claim. No driving-performance claims come from art reviews.

## Useful Existing Entry Points

- Editable kit: `art-source/forest/tree-kit-v3.json` and identical Resources copy.
- Tree meshes: `Assets/Core/Environment/StageDressing.TreeKit.cs`.
- Foliage shader: `Assets/Art/Shaders/TreeFoliageV3.shader`.
- Native forest checks: `EditorScripts.ForestKitAssets.Prepare` / `.Validate`.
- Native effect reset fixture: `EditorScripts.EffectsReview.Validate`.
- Fixed shots: `EditorScripts.StageReview.Comparison`, with `-reviewCourse` and
  a new `-reviewLabel` to preserve prior output.
- Runtime review: `.venv/bin/python Tools/visual_review.py LABEL --height 720`
  (also 1080, `--camera Trackside`, and `--legacy-forest`).
- Tests: `.venv/bin/python -m unittest discover -s tests -q` (pytest is not installed).
- Build/status: loopback service `/api/build` and `/api/jobs/{id}`.

All paths above are relative to `/Users/REVIEW_USER/RallyAI3`. Inspect current
files/API schemas before using them; historical job IDs are evidence, not live handles.
