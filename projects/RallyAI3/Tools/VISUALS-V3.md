# Visuals V3 delivery ledger

## Boundaries

Keep Built-in rendering, blue/gold sedan, 60 FPS viewer target, frozen course,
collision, physics, observations, rewards and training behavior unchanged.
`visuals-v2` preserves the user's uploaded baseline; `visuals-v2-refined` marks
the first refinement at `ddc958b`.

## Review protocol

- Frozen bare-road Crest course: `f187eaf562a3681aba54681939a54345c84d484556cef1b97baa2048acb5f8f4`.
- Course bundle SHA256: `79e4495295e40c4164f878244783ee3e0c679474e6ddc90a531803285a1f392e`.
- Policy: `02d659fb4993a27faeb14292e18eff37c39790038be25714d8d296eda6011b19`.
- Deterministic inference, seed 2026, standing neutral start, simulation scale 1.
- Fixed shots use the frozen bundle (not regenerated geometry), 1280x720 and 1920x1080.
- Runtime review uses `Tools/visual_review.py LABEL --height 720|1080`; output is
  isolated from experiment artifacts. Warm-up frames are excluded from statistics.
- Profiler values of -1 mean unavailable, not zero. Instrumented frame timings
  include sampling and screenshot overhead; they are not an uninstrumented benchmark.
- Unity preparation/build/review jobs must run serially against the project.

## Gates

1. Repeatable captures and performance baseline: in progress.
2. Four-species branch-based tree kit, near/mid/far LODs, representative section:
   in progress. User visual approval required before broad rollout.
3. Unified ground/road/vegetation treatment: pending.
4. Car body, wheel, material and cockpit finish: pending.
5. Contact-driven motion effects, reset/storage/camera checks: pending.
6. Audio listening and matched recordings: pending.
7. Saved-course standalone coverage, headless isolation, cleanup, tests and V3 save:
   pending.

No overall completion claim until every gate has direct evidence. In particular,
a passing build or unit suite does not prove visual quality or audio quality.

## Denser / darker review revision (2026-10-04)

User requested fuller and darker forest after the open-canopy preview. Increased
candidate retention, crown widths and branch counts; cooled/darkened foliage tint.
Still restricted to the 190-430 m review section on seed 41200, not broad rollout.

- Trees: 1,287, up from 979; all four species represented in 11 LOD cells.
- Four reloads: stable mesh count, unchanged Unity random state, no decorative
  colliders; frozen collision hash remains
  `047b860ede36d69234dfaaeed41190f960a2aa6078953f26563bd430eb2c7c7e`.
- Unity preparation and standalone build `0db2608e334445d0` completed.
- 43 Python unittest tests passed; whitespace checks passed.
- Fixed-angle comparisons: `.rally/visual-review/stage/v3-dense-*-final-*`.
- Actual standalone captures and frame/resource records:
  `.rally/visual-review/v3/dense-review-720` and `dense-review-1080`.
- 720p: median 16.67 ms, p95 17.54 ms, instrumented average 53.70 FPS.
- 1080p: median 16.67 ms, p95 17.38 ms, instrumented average 54.38 FPS.
- Draw samples now bind Unity 6's `Render/Standard Draw Calls Count`, rather than
  the unavailable legacy counter. These are standard draws, not an asserted total
  across all rendering categories. Missing other counters remain -1.
- macOS reported no recorded thermal/performance warnings, not a temperature
  reading or a guarantee about thermals.

Road and driving identities unchanged. Captures are presentation checks, not
official evaluation evidence. User approved the denser/darker tree direction on
2026-10-04; broad rollout is now authorized, but has not been implemented.
Car, ground integration, effects and audio gates remain pending.

## Saved stopping point

- Runtime wheel emitters now own dust; the legacy plume is cleared and disabled
  when that owner is configured. Offline review tooling retains its legacy plume.
- Episode-end notification clears particles and tracks; teleport detection remains
  a fallback. Native populated-effects reset fixture passed eight repeated resets.
- StageDressing refuses managed non-viewer runs even with a graphics device.
  A real headless smoke test remains outstanding.
- Latest standalone build job: `02edde02dc494b4d`, completed.
- Actual trackside review: `.rally/visual-review/v3/effects-single-owner-720`;
  median 16.67 ms, p95 17.39 ms, instrumented average 53.87 FPS.
- No exceptions found in that player log. Trackside capture `drive-02.png` loses
  the car: fix trackside placement/framing and verify the other five modes.
- No Unity Editor or standalone player process was running at handoff.
- Existing generated editor episode log is preserved but excluded from the
  visual source commit; it is not performance evidence.
- Selected captures are archived under `art-source/forest/review/` alongside
  this revision. Full raw measurements remain in ignored `.rally` directories.

Continue with `Tools/VISUALS-V3-CONTINUATION.md`. This checkpoint is a reviewed
forest slice and initial effects repair, not a completed Visuals V3 release.

## Full-stage review implementation (2026-10-04)

- Production tree coverage now spans all courses; review bounds remain editable
  for explicitly bounded configurations. Legacy forest remains opt-in for comparisons.
- Trackside stations aim continuously at the smoothed vehicle, use clear shoulder
  positions, visibility checks, selection hysteresis and cuts rather than travel
  through scenery. Missing stations fall back to helicopter framing. Episode-end
  notifications invalidate camera smoothing and station state even for small resets.
- Generator-owned collidable trees retain their collider transforms, tags and
  dimensions. Their old merged cone renderer is disabled; replacement branch
  trees are anchored to those exact positions and preserve trunk collision scale.
- Preserved the distance-aware ground shader instead of overwriting it in the
  presentation pass. Added cooler verge coloration, litter patches and more varied
  fern/tuft silhouettes; no collision or road mesh modification.
- Review tooling supports selected saved courses, per-frame framing/station data,
  material counts, screenshot-free timing and optional sampled moving sequences.
  Ten-frame-per-second review clips are not 60 FPS smoothness benchmarks.
- Eight native reload checks per corrected starter course passed collision/RNG
  invariants and stable generated mesh/material/object counts. Reports are in
  `.rally/visual-review/v3/validation-final-*.json`.
- Six camera-mode drives, 720p/1080p chase drives and all six saved course variants
  were exercised in the standalone player. Full-route completion by this Crest
  policy is not asserted on unfamiliar courses; those runs prove loading and
  sampled presentation, not learning performance or exhaustive route traversal.
- Corrected trackside trace: 1,090/1,090 supported-station frames after warm-up
  inside the safe frame. Settled cinematic trace had no FOV jumps over one degree;
  97.2% car-centre safe framing, so transition framing is not claimed perfect.
- Screenshot-free 1080p: median 16.67 ms, p95 17.43 ms, p99 17.59 ms;
  capture-independent settled camera-trace mean 59.53 FPS. Whole-run review mean
  51.37 FPS includes initialization and low-rate instrumentation. Peak working set
  reached approximately 1.9 GB in some full-forest runs, substantially above the
  bounded slice. Further memory optimization is warranted for longer stages.
- Dashboard Watch defaults and actual launch were inspected: specialist course,
  checkpoint, neutral start, deterministic action selection and runtime scale 1
  matched. Viewer `3ed3c217c60941ea` stopped cleanly.
- Real fresh 1,000-step one-worker headless training job `ef828cb631c64f78`
  completed, exported checkpoints and returned exit 0. Opt-in runtime inventory
  found zero dressing, renovation, legacy dust, wheel effect and forest LOD systems.
  This is a platform smoke result, not evidence of driving quality.
- 45 Python tests, three checkpoint-default JavaScript tests and dashboard
  production build passed. Strict build/resume validation was not relaxed.

Stage-only review artifacts are in `art-source/forest/stage-review/`. Car/cockpit
and audio work has intentionally not begun. Driver mode still exposes unfinished
interior geometry; completing it remains the subsequent car milestone. Exhaustive
all-route camera/landing coverage and runtime multi-episode resource testing are
not proven by these bounded drives; retain them as explicit follow-up verification.

Final stage-review build: `b7d3dc24f3de4659`, completed/current. Six corrected
starter-course variants were checked again in this build after replacing legacy
collidable-tree visuals. Their logs contained no exceptions or shader errors.
`art-source/forest/stage-review/evidence.json` records the final build manifest,
14 selected runtime summaries, framing traces and six native reload-audit reports.
The clips and bare-road comparisons precede the last anchor-trunk refinement,
which only affects courses with collidable scenery trees; no such trees exist
on their pinned bare-road course. No Unity player/Editor was left running.
Local tag `visuals-v3-stage-review` marks this review revision; it is not the
finished overall V3 release. No push was performed.
