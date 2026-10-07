# Visuals V2 baseline

Saved 2026-10-04 before the next presentation pass.

- Source commit and local tag: `c89cf878973ef8cfa22c26883cee9af02b8a4405`, `visuals-v2`.
- Current player source fingerprint: `147ad19d287025d094d01917aafc8efe1339f86cf3d50c623088337db169078c`.
- Reviewed course: `f187eaf562a3681aba54681939a54345c84d484556cef1b97baa2048acb5f8f4`, seed 41200, Crests / bare road.
- Reviewed policy: `02d659fb4993a27faeb14292e18eff37c39790038be25714d8d296eda6011b19`.
- Existing matched car captures: `.rally/visual-review/car/`, 720p and 1080p.
- User reports the baseline was uploaded to GitHub; remote synchronization has not been independently verified. The local tag has not been pushed.

Review: improved road detail, route furniture, vegetation density and atmospheric depth. Remaining priorities: overly bright dust, repeated conifer silhouettes and ground-cover clumps, opaque-looking glass and overly clean car materials, uniform tire trails.

Next pass is viewer-only. Preserve course geometry, collision, vehicle behavior, observations, rewards and training configuration. Rebuild and inspect the standalone player before claiming completion. Keep the baseline tag unchanged.

## Follow-up pass

- Correct opaque dust material to alpha blending; neutralize color and reduce opacity.
- Reduce uniform wheel-track darkness; vary new marks with measured load/slip and spatial grain.
- Add gaps and longitudinal jitter to the distant forest; vary species mix.
- Break up near-tree canopy outlines with more irregular, off-centre tiers.
- Reduce verge density, broaden plant heights, and mute yellow-green tint.
- Separate glass, rubber, paint and lamp finish more clearly without changing geometry.
- Final combined build `8298dd30167c408d` completed successfully; build status current.
- 40 Python tests passed; `git diff --check` passed.
- Inspected standalone viewer `537f8fa149904174` on the same course and policy at normal simulation speed. Dust no longer hides the car; verge shows more gaps; wheel trails are softer. Forest outlines remain stylized and need a future authored branch/foliage treatment for a stronger fidelity improvement.
- The newer DustPlume shader already used alpha blending; its excessive scattering and opacity were reduced separately from the legacy opaque wheel-spray material.
- No training jobs started; physics, course assets and policy contracts were not edited. Frame rate and thermal behavior were not measured during this review.
