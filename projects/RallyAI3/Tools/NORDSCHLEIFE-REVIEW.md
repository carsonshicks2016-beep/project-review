# Stage C — complete detailing review, 2026-10-05

The complete circuit is detailed and inspected; the single user visual-review gate remains open. The new candidate is **Nordschleife / complete detailing review**. Open http://127.0.0.1:8765/circuit-review/index.html for the map, elevation, moving captures, profiles, references and requirement evidence ledger.

Exact identities:

- Frozen course: `f0d8d95ac7d70cc5a9c7ebf056f755f1f18bee3844c7d2cf42d9f5354b90f2b2`.
- Circuit revision: `1d518c856f6aa3508d7a75e1f25f871075351f2853cfaa189da326fe450363bc`.
- Generator: `circuit-geometry-v7`; profile schema 2, with explicit schema-1 compatibility.
- Completed standalone build: `bf0d211d15a04012`; player source `76790a0c5ccd542e64604b8b129fff1660271d784eb541d1142582ef9e975e38`.
- Player: `.rally/player/RallyTraining.app`; executable SHA-256 `a6bcb62787a76cd80582c42bf05609c1899bc80e854e2c36b844a700c0c5da12`.
- Preserved continuous-terrain baseline: `50bb4aa4507c1a2a4d882dae1f65cfeafab09214460e54f190ec13246a6a068b`.

Evidence is in `art-source/circuits/nordschleife/stage-c-review/`; exact original native outputs are `.rally/circuit-review/stage-c-delivery-*`. The evidence manifest binds course, profile, terrain, source and player identities; `requirements.json` links each acceptance item. Every one of the 16 overlapping road-level and elevated regions, 12 signature areas and all six camera modes at both resolutions was inspected. No terrain hole, exposed underside, buried pavement or vegetation intrusion was observed within these captures. The native audit found zero failures among 221,674 ground samples, zero stacked ground contacts or mesh/collider mismatches, and zero sampled vegetation intrusion. These are bounded sampled checks, not proof of every possible camera/contact scenario.

Eighteen real vehicle probes cover both bowls/bypasses, entry transitions, kerbs in both directions, grass transitions, crests/dips, repaired hillside, seam and barrier impact. They deliberately cap low-speed diagnostic time and cannot establish a valid lap or driving competence. Camera traversals are kinematic inspection only and cannot produce ranked times. Eight reconstruction and three true unload/reload cycles had stable resource counts after the initial frozen-to-runtime allocation change; working-set variation and existing cockpit shutdown warnings are documented in findings.

Python tests, checkpoint-default checks, dashboard production build, Unity compile and standalone build are recorded in `regression.json`. Original 90 course-file and 12 physics-file hashes are preserved. Performance measurements and equivalent repaired-baseline comparisons are recorded without a locked-60-FPS claim; see the review page for measured median/p95/p99, memory and unavailable counters.

Profiles author selective side kerbs, tapered physical edges, barriers, shoulders and descending concrete transverse bowls. Shared constrained tiled terrain meets the road/kerb/shoulder perimeter, resolves neighbouring roads together and extends/skirted ground beyond the camera envelope. Forest cell identities are deterministic and only entering/leaving cells rebuild. Historical manufacturer references support features, while numerical widths, crossfall, bowl angles, barriers and overpass dimensions remain estimated. Source centerline/elevation remain unchanged, surrounding terrain is reconstructed and T13 is source-landmark placement rather than a surveyed timing line. See `SOURCE-ATTRIBUTION.md` and `reference-ledger.json`.

The final candidate stays unreviewed for training. No PPO smoke test or sustained campaign was started by Stage C, no historical run resumed, and no remote push performed. A separate active training job `21cb2eeff33940d3` was stopped normally at 2,280,000 steps at the user's explicit request; checkpoints and job history were retained. Other chats' course eligibility and records remain untouched.

To reproduce after finishing the review, use the tools' `--help`: `Tools/circuit_inspect.py` provides explicit ranges/duration/cameras and route/signature/matrix/lifecycle modes; `Tools/circuit_probes.py` runs real physical probes; `Tools/circuit_review.py --performance-only` measures without startup audit; `Tools/circuit_review_report.py --prefix stage-c-delivery` packages the reviewed exact identities. Regenerating profiles requires a new revision, terrain preparation, import and standalone build; old frozen bundles remain loadable and are not reinterpreted.

---

## Historical first-review evidence (retained)

# Nordschleife port — first review gate, 2026-10-04

## Status and next action

Stages A/B are implemented as the **first reconstruction review candidate**.
The complete source layout is imported, with asphalt, basic full-route kerbs and
barriers, shared tiled terrain, streamed existing forest, and estimated concrete
bowl profiles. Native chase, hood and driver captures were inspected. The selected
Karussell slice now needs user visual review before extending reference-driven
detailing around the entire circuit, exactly as the approved plan requests.

This is not the complete detailed-circuit/trainability milestone. No course was
marked reviewed, no historical run was resumed, no training smoke/campaign was
started, and no changes were pushed. Unrelated preexisting dirty work is preserved.

## Delivered candidate and source

- Frozen course: `d46ec342e3b72ba250caa1dfb6bd370ca85b71e0468b0a645c7ed27573acfb31`.
- Circuit data/profile revision:
  `cc808da4b0950c1ce54753432da96d7f79727bf597fb707cee035525fa47459f`.
- Current build job: `1dd99eba628c431f`, terminal completed.
- Player source identity:
  `cd0ae89a6b539750120670dee48e830e5ba01ccf9ef060b35cd45ce5ebc0164f`.
- Original snapshot, attribution, importer report and editable estimated bowl
  profiles: `art-source/circuits/nordschleife/`.
- Converted periodic source polyline: `Assets/Resources/Circuits/Nordschleife.json`.
- Preserved before-work patch, build/contract and original course hashes:
  `.rally/circuit-review/baseline/`.

The original centerline and elevation vertices are preserved, translated near
Unity's origin and reindexed at the source T13 landmark start. Subdivision preserves
the source polyline with at most 1.5 m between converted samples. This is not a
surveyed timing-line placement or additional survey precision. The source's
approximately 295.7 m elevation range is not stretched. Width is the source's
constant estimated 11 m. Existing global grip presets, not measured Ring friction,
govern contact. Source attribution remains OpenStreetMap ODbL/RLP dl-de/by-2-0.

## Verified evidence

- Python suite: 65 passing tests. Checkpoint-default tests and dashboard production
  build pass. Semantic compile check passes; its source list now includes the
  Presentation directory, which it previously omitted.
- The actual standalone frozen course passed 417 centerline station inspections,
  834 kerb and 834 barrier rays, 417 tarmac surface checks, and gate-neighborhood
  projection checks. Reported failures: zero. Seam position error: zero.
- Native full-lap budget: 180,000 physics steps; fixed timestep 0.01 s. This checks
  configuration, not that an agent can finish before timeout.
- A bounded fixed-input native drive contacted `Barrier_59_1`, tagged `Obstacle`,
  at about 4.44 m/s. Its recorded outcome is `HitObstacle`, invalid time. This is
  one physical contact case, not complete barrier crash/damage validation.
- Inspection found no nonlocal source centerline pairs within 20 m in a stride-4
  proximity check excluding stations within 100 m along the route. Closest found
  separation is approximately 28 m around the Karussell approach/exit. This is
  a sampled screening result, not proof of every mesh/terrain clearance.
- Original frozen-course files retain all baseline hashes. All 12 recorded
  vehicle-physics source/meta files match baseline. Driving contract is identical:
  vectors/stacks/actions/sensors/fixed timestep are preserved.
- Dashboard Courses shows the imported map/elevation and honest unreviewed labels.
  Its **Review Karussell (unranked)** button launched native review job
  `b77f8f78a0d54145`; the job completed and its player exited. No active managed
  jobs remained at handoff.

Native folders:

- `.rally/circuit-review/karussell-final-chase-720/`
- `.rally/circuit-review/karussell-final-driver-720/`
- `.rally/circuit-review/karussell-final-hood-720/`
- `.rally/circuit-review/karussell-barrier-contact-720/`

Selected captures, performance/audit JSON, contact evidence, and a bounded
10 FPS screenshot-derived preview are versioned under
`art-source/circuits/nordschleife/review-20261004/`. The video is not a native
60 FPS recording. Captures are diagnostic waypoint-follow drives with no policy
and deliberately invalid ranking; they do not establish learning performance.

## Implemented foundation

- Independent imported branch alongside existing procedural generation.
- Chunked road meshes/colliders, surface-consistent asphalt/kerb/concrete contacts,
  corrugated rail presentation and batched posts around physical barrier envelopes.
- Shared terrain grid plus exact shoulder strips; no procedural roadside hazards.
- Streamed forest built from the existing V3 kit with isolated station RNG and
  clearance checked against all nearby road sections.
- Separate geometry samples and approximately 50 m driving gates; wrapped road
  sampling/lookahead and station-bounded projection protect nearby parallel roads.
- Imported finish includes the final gate rather than the procedural path's
  preceding-gate convention. Multiple crossings are bounded; discontinuous
  movement cannot collect circuit gate rewards.
- Imported fall-out is road-relative, so the circuit's low elevations do not
  trigger the old absolute-world-height cutoff.
- Footprint-based invalidation, imported budget defaults and extra attempt
  metadata. Current source still refuses historical-environment resume.
- Explicit allowlisted `/api/circuits/import` accepts full lap or a non-wrapping
  contiguous range of the 16 equal-distance sector definitions. These are not
  official timing sectors. Procedural 1–5 km API limits are unchanged.
- `/api/circuits/{id}/preview` allows bounded diagnostic review of an unreviewed
  circuit without granting training eligibility.

## Remaining after user review

1. Apply approved detailing to the complete route: reference-supported variable
   width, kerb placement, barrier clearance, markings and banking. Current full-route
   edges are generic estimated candidates, not researched exact placements. Bowl
   transitions/dimensions are estimated and must be reviewed.
2. Improve/inspect surrounding terrain and tight-return shoulder ownership. It is
   synthesized from road elevation, not cached surveyed surrounding terrain.
   The grid is shared; wide exact shoulder strips still require clearance review
   around the closest returns. Do not claim no terrain overlap from centerline rays.
3. Create and inspect the full 16-sector library and longer combinations. Implement
   seeded distributed training starts, explicit budget controls and complete
   revision/start-profile/budget resume invariants through API, worker and UI.
4. Complete deterministic runtime fixtures for full-lap start/finish, reverse and
   missed gates, teleportation, multiple crossings, seam, nearby-road/height
   ambiguity, retry, validity and timeouts. Current audits verify geometry and
   projection, not all episode-state behavior.
5. Verify selected-sector map/elevation highlighting, native sector/result labels,
   end-to-end policy watch/evaluation, and training record separation. Diagnostic
   station starts currently count skipped gates as displayed progress; those
   unranked counts are not driven gate completions and must not become benchmarks.
6. Full-route moving inspection, all six modes at 720p/1080p, contact/landing and
   off-road surface cases, eight reset/reconstruction cycles and three full reloads.
   Current review covers three moving camera modes at 720p, not this entire matrix.
7. After reviewed-course eligibility, run the two planned one-worker 2,000-step PPO
   smoke jobs, verify exports/shutdown/headless isolation, then final acceptance.
8. Preserve final evidence and save the completed release only when these gates
   pass. No sustained training or remote push is authorized by this milestone.

## Reproduction

Through the local dashboard at `http://127.0.0.1:8765/#courses`, find
**Nordschleife / full lap** and select **Review Karussell (unranked)**. The candidate
is intentionally unreviewed; do not mark it reviewed merely to enable training.

For new distinct evidence folders, use `Tools/circuit_review.py LABEL --course ID`
with `--camera Driver` or `--camera Hood`, and optionally `--sequence` or
`--barrier-contact`. The tool checks player freshness and course integrity and
holds the shared compute/viewer locks. Each review is bounded and exits its player.

## Terrain-gap follow-up

User reported a large opening gap on the left of the Karussell review approach.
The old grid omitted triangles near the road and its swept shoulders did not
cover the hillside cutout. The replacement keeps every terrain triangle, limits
height jumps between nearby circuit returns, and connects multi-band shoulders
to the shared triangulated height field. No pavement/source geometry was changed.

Updated frozen course: `50bb4aa4507c1a2a4d882dae1f65cfeafab09214460e54f190ec13246a6a068b`.
Import `74e83a9bfa2044e3` and standalone build `fb211dbd6d654a15` completed.
The earlier candidate remains preserved and is named superseded terrain review.

Native 720p moving chase inspection confirms the reported opening hole is closed.
Full-route audit: 15,012 ground-support rays on both sides, at 50m stations and
2m lateral increments through 35m beyond half-width, with zero failures. Existing
417 center/834 kerb/834 barrier checks, surface and projection checks also report
zero failures. This sampled audit does not replace the deferred every-section
visual inspection. Compile and all seven circuit data tests pass.

Evidence: `art-source/circuits/nordschleife/review-20261004/terrain-joins/`.
720p chase median/p95/p99: 16.667/17.344/20.245ms. Geometry remains an estimated
reconstruction, and the original staged review/training gates remain in effect.
