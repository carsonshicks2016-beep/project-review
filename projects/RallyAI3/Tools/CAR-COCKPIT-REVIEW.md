# GC8 cabin review gate

The first candidate extends the roof 0.10 m fore and aft while retaining the
windshield and rear-glass bases. Connected glazing/frame tops move with it.
A shallow 0.012 m maximum roof crown replaces the flat roof.

Body length, bonnet, boot, wheel arches, lights, wing and wheels are unchanged.
The exported body has 9,468 triangles. Wheel export is identical to the baseline.
No physics, collision, suspension, damage, audio or training code is edited.

Local preserved source and matched assembled-car studio captures:
`.rally/visual-review/car-cabin-gate/` (baseline-source, before, after).
Eight views are captured at 720p and 1080p. Driver/hood captures retain the
current glass limitations; they are baseline evidence, not a completed cockpit.
The review tool disables simulation components in its temporary scene and does
not save that scene. It captures the existing assembled car when available.

Review must settle this silhouette before interior modeling. Cockpit construction,
material refinement, glass replacement, full lifecycle tests and the final
car/cockpit milestone remain pending. Audio bumping is unresolved and untouched.

Validation: 46 Python tests and three checkpoint-default tests pass. Unity batch
captures at both resolutions and managed build 01e5ee07735f4535 succeeded.
Standalone chase capture verified course F187EAF5 and checkpoint 02D659F8.
Moving evidence: `.rally/visual-review/v3/cabin-gate-chase-720/`; preview:
`.rally/visual-review/car-cabin-gate/chase-preview.mp4` (10 fps screenshot sequence,
not a 60 fps recording). Capture-run frame times: median 16.67 ms, p95 32.79 ms,
p99 33.33 ms. Screenshot overhead is included; no performance improvement claimed.
No new matched pre-change moving sequence or full camera/reload/headless campaign
was run at this first gate. Those checks remain for the approved full milestone.

## Approved silhouette and cockpit candidate

The user approved the longer cabin. The next candidate adds a separately exported
4,248-triangle interior, right-hand-drive dash/console, seats/harnesses, cage,
door cards, floor/tunnel and levers. The body has 11,200 triangles; each wheel
has 2,000. Tow hardware, exhaust and brake discs are visual only. The steering
rim now has 48 segments. Window surrounds are open frames, with permanent
transparent glass instead of the camera-mode material override. Specular material
settings now reach the Built-in specular shader. No new dirt simulation is added.

Cockpit instruments use shared geometric glyphs because runtime font text produced
black rectangles in the first standalone capture. RPM needle, gear, speed and
shift indication use existing telemetry, not invented instruments. Cockpit objects
and resources are installed only for viewing. The original steering binding is
reused and restored during teardown.

Onboard cameras now use the shared smoothed vehicle orientation and take the
already-smoothed target position directly. The old second filter trailed the car
far enough to place the eye behind the seat at speed. Exterior shot framing and
mode identifiers are unchanged. The interior follows the same pose in driver mode.

Evidence: `.rally/visual-review/car-cockpit-complete/` and `cockpit-*` folders in
`.rally/visual-review/v3/`. Selected captures/video are versioned under
`art-source/gc8/cockpit-review-20261004/`. Video is a 10 fps image sequence,
not a 60 fps recording. Both 720p/1080p studio views and actual standalone driving
were inspected. All six modes ran on build 0281d6edab1546ae; subsequent build
a04e4a9667d54ff0 includes the refined steering rim and was checked at 1080p in
driver mode, in ordinary viewing, and with native lifecycle instrumentation.

Eight editor rebuilds kept 17 car transforms, one vehicle collider, 1,230 kg mass,
unchanged suspension positions and steering binding. Eight native cockpit
reconstructions kept one cockpit, 20 transient meshes and stable material/collider
counts. Forty instrument samples matched source telemetry within frame tolerance.
These are rebuild/cockpit lifetime checks, not eight full course scene reloads.

Ordinary 720p driver view without recording: median 16.667 ms, p95 16.744 ms,
p99 17.531 ms. Capture sequences slowed tail frame times; no locked-60-FPS or
performance-improvement claim is made. Compare resource samples in the saved
performance manifests; unavailable counters remain -1, not zero.

Headless PPO smoke c053249d59604290 completed with exit 0, one worker and a
1,000-step budget. Isolation report: zero cockpit, audio and forest presentation
instances. Eighty-seven physics/training/audio/scene/config files match the prior
Git baseline; frozen course bundle SHA remains unchanged. No thermal/performance
warning was recorded by macOS; temperatures were not measured.

The candidate is ready for visual review, not final artistic approval. Full course
scene reload coverage and exhaustive manually observed mode-switch/reset/landing
interaction remain limitations. The audio bumping is still unresolved and untouched.

## Cockpit/body coupling correction

User review found the interior rising/falling independently of the shell. The
initial driver-mode world-pose writer was responsible: its smoothed height and
rotation differed from the interpolated body transform. That writer is removed.
The cockpit stays at zero local position/rotation under the vehicle in every mode.
Driver and hood cameras use the vehicle's interpolated render transform directly,
without a second positional filter. Exterior camera smoothing is unchanged.

Build 9919908ec3114810 passed a moving driver review and eight cockpit reconstructions.
All 40 native coupling samples have zero local position/rotation error; instrument
checks also pass. A regression test rejects detached cockpit motion. Python suite:
51 passed; checkpoint-default suite: three passed. Evidence is under
`.rally/visual-review/cockpit-body-coupling/` and
`.rally/visual-review/v3/cockpit-body-coupled-driver-720/`.
The older review captures are preserved as prior-version evidence, not overwritten.
