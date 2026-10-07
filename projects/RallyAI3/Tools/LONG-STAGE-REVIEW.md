# Longer stage and frozen result review - 2026-10-04

## New course candidate

Forest Endurance / 3 km / Clear, seed 43001, technical family:
`5524a5f1c2554c7845af9791a36a7fed5f9faf9cba41a4d3e470753aeb1aeb10`.
Resolved length 3000 m, 63 control points, zero hazard objects.
Heading range -58.63 to 69.97 degrees; largest adjacent heading change 74.71
degrees. Elevation range 8.96 m. Shared road/terrain construction retains 50 m
control spacing rather than stretching the original twenty segments.

Length is configurable from 1000 to 5000 m through the course dialog/API.
Existing requests default to 1000 m. Longer new assets use a smooth alternating
fast/technical heading envelope; existing frozen courses are not regenerated.
Family heading limits remain intact. Technical long stages add elevation.
True hairpins remain outside this forward-progress generator's safe topology.

The course map and first driving sector were inspected in the actual standalone
player. The existing Crest specialist fell off early: this is not a valid time,
nor evidence of a defective course by itself. Full-route road/terrain/forest
inspection remains required; the course stays unreviewed and cannot yet train.
The existing 120-second episode budget is unchanged, including on long courses.

## Result presentation

The viewer records an attempt before freezing simulation. The spectator director
remains the only camera writer and orbits the frozen car once per 24 seconds.
Nearby terrain can raise the orbit camera. Audio playback pauses with the result.
Retry resets the existing viewer episode; Exit closes the player.
Automated evaluation and training do not enter this presentation path.

The panel identifies watch inference, checkpoint, outcome, attempt, ordered gate
progress, peak speed in MPH and final sector duration. Valid finishes show their
recorded simulation time. Failures show no valid time and retain elapsed time.
The main timer uses recorded time while frozen, excluding pre-start time.
Full gate split arrays and trajectories remain in existing attempt artifacts.

Final build job `dc74238dfc7a410b` is current. Corrected native finish evidence is
`.rally/visual-review/v3/finish-orbit-final-720/`, with fourteen frozen samples
passing validation and a bounded 10 FPS screenshot-derived `finish-preview.mp4`.
This clip shows part of the continuous orbit, not a full 360-degree recording.
Matched one-attempt headless evaluation `c8468310073d4c09` completed normally at
10x simulation speed: one valid Finished attempt, 27.35 s, equal to the captured
watch result. This verifies the non-viewer exit path, not general determinism or
a fixed-budget performance comparison. No active managed jobs remain.

Native evidence: `.rally/visual-review/v3/finish-orbit-crest-720/`.
Fourteen paused samples kept one car position and one simulation time while the
camera changed position in every sample. Result time/validity match its episode
record. This first capture predates the main-timer display correction.
Managed viewer `270e4a3b4de64375` subsequently recorded two finishes and its
updated timer/result panel was visually inspected. A further automated Retry click
was blocked by the native UI connection; do not claim exhaustive button testing.
The viewer was stopped through its owned dashboard API path.

First long-stage player evidence: `.rally/visual-review/v3/endurance-first-view-720/`.
Failure-result capture: `.rally/visual-review/v3/endurance-failure-orbit-720/`.
Its fourteen frozen samples passed the same validation: FellOff, invalid, elapsed
9.59 s. The failed car is inside collider-free forest decoration; trees can obscure
parts of an off-road orbit. Terrain clearance does not solve foliage occlusion.
That remains a presentation limitation rather than adding decorative colliders.
`Tools/check_finish_review.py` checks frozen pose/clock, moving camera and recorded
result agreement. These are watch artifacts, not ranked benchmark evaluations.

## Verification boundaries

Python suite: 58 passing tests, including length bounds, defaults and result
evidence regressions. Production dashboard build passed; refreshed service exposes
the length schema and the course appears in the connected library.
Recorded moving-view median frame time was about 16.67 ms; capture/startup overhead
affects averages and tails. No locked-60-FPS or thermal guarantee is made.
Physics, suspension, colliders, rewards, actions, observations and training budget
remain unchanged. No new training job was launched.

The prior cockpit polish is preserved but paused at its visual review gate.
Audio bumping and full-stage art approval remain separate outstanding work.
