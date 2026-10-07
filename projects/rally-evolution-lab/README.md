# Rally / Evolution Lab

A runnable machine learning experiment with 3, 6 and 10 km courses: 48 neural rally drivers evolve on a procedural gravel circuit with uneven elevation. The objective is to complete a full lap, then minimize lap time.

## Run the interactive project

Requires Python 3 and a modern browser with WebGL:

```sh
python3 -m http.server 5173 --directory dist
```

Open http://localhost:5173 and click **Start training**. No installation, account, GPU training service, or API key is needed. Three.js is vendored; online fonts are optional and have system fallbacks.

- **Simulation speed:** 1×, 5×, 20×, or 60×, subject to device performance.
- **Overview / Follow car:** orbit the circuit or follow the leading driver. Drag or use the arrow keys to orbit; scroll or +/- to zoom. Overview frames the whole terrain when the road still reads at that distance, and otherwise orbits the leading driver over one sector at a fixed, legible scale — the minimap carries the full lap. The scene label reads SECTOR when it is doing that.
- **Course length:** 3 km club circuit (default), 6 km long circuit, or 10 km endurance circuit. These are about 3.6×, 7.3× and 12.2× the original circuit length.
- **Sector timing:** four equal-distance splits, session-best times, and an ideal lap assembled from the fastest sectors.
- **Minimap / telemetry:** leader position, grade and tire load; long-course training automatically enters Follow car view.
- **Racing lines:** speed-colored champion trajectories, with the last four improvements fading underneath.
- **Algorithm comparison:** genetic search, hill climbing and random search on the same course and initial population, with 20/50/100 generations at 48 evaluated drivers each.
- **Copy driver link:** saves exact weights and course settings in the URL fragment. The recipient needs access to the app; localhost links work only where this project is running.
- **Surface:** dry gravel, or wet for 25% less grip in cornering and in drive/brake alike. A dry-trained champion will not lap a wet circuit at its dry pace; the population re-adapts over a dozen generations. Dry physics is bit-for-bit unchanged, so every stored lap time and exported driver still replays exactly.
- **What the driver uses:** ablation zeroes one of the 12 inputs and replays the champion, pricing each sense in lost lap time (or showing where the lap breaks without it). Beside it, the genome's 24 weights are drawn as a grid, with the same numbers in a screen-reader table.
- **Population diversity:** mean pairwise genome distance, plotted under the learning curve on its own scale. Watching it collapse is what explains a lap-time plateau.
- **Seed / elevation:** generate a repeatable track with a different layout or hills. Generating transfers the current best policy but resets lap records and generation history for the new circuit. Clicking Generate with unchanged settings picks a new random seed.
- **Replay best:** run the fastest learned policy at real time. Return to training with the same button.
- **Export driver:** download the best policy, track settings, and lap record as JSON.
- **Import driver:** load an exported driver back in. The saved track is rebuilt, the policy is replayed once to establish its lap time, and the next generation starts from it. Files without track settings load onto the current circuit.
- **Reset learning:** restart from a fresh population on the current track.

Training lives in memory in the current tab. Closing or refreshing it resets the session. Export a driver you want to retain.

## Train without graphics

Requires Node.js 20 or later; there are no npm dependencies:

```sh
node train.mjs --seed 2077 --elevation 18 --length 3 --surface dry --algorithm genetic --generations 100 --output driver.json
npm test
```

The CLI prints generation results and saves the champion weights with its history. `newCar(track, weights)` and `tickCar(track, car)` replay exported weights; see `tests/verify.mjs` for a complete example. A driver saved here loads into the UI through **Import driver**; training history is not restored.

## Machine learning

The policy is a small neural network with 12 inputs and two tanh output neurons: steering and signed throttle/brake. Its 24 weights are the genome. There is no prerecorded driving path, hidden autopilot, synthetic training chart, or external model call. The visualization reads the actual simulated cars and training results.

Inputs are heading errors to four preview points (8, 18, 35, 60 meters ahead); normalized lateral offset, speed, slip angle, upcoming curvature, grade, tire-load deviation, prior steering; and a constant bias. This agent receives exact track-relative observations, rather than learning from camera pixels.

Initialization uses a noisy road-following prior, not a pretrained policy. This makes a small browser experiment practical. A generation evaluates 48 genomes. Eight top drivers are eligible parents; the all-time champion and five elites are copied unchanged. Other drivers receive crossover and Gaussian mutation with a gradually decreasing noise scale; occasional fresh policies encourage exploration. After 25 generations without improvement, mutation reheats to at least 0.12 and fresh immigrants enter each generation. The champion is retained. Hill climbing mutates the incumbent champion, and random search independently samples the same prior distribution. Comparing a single seed and track is not evidence of universal algorithm superiority. Equal evaluation budgets do not imply equal CPU time. Evolution operates on full episodes at a fixed 30 Hz timestep.

Incomplete attempts score `9000 × maximum forward lap fraction`, minus a small off-track penalty. Completed laps score `10000 + 40 × (attempt limit − lap seconds)`. Completion therefore dominates incomplete progress; among completed laps the fastest wins. Attempts end after a lap, its course-specific time limit, prolonged off-road driving, a major departure, reversing too far, or stalling. Progress uses ordered local road segments with start-line wrapping, so simply crossing the finish line or driving backwards cannot count as a lap.

## Track and physics

A seeded radial circuit blends several periodic curves. Long courses add extra bends, are scaled to the selected horizontal centerline length, and use roughly 1.7-meter road samples. Terrain extents, hill wavelength and amplitude, camera framing and replay limits scale with the course. Relief amplitude scales with the same factor as the wavelength, so a longer course gains real climb (roughly 190 m of ascent at 3 km and 750 m at 12 km) rather than the same hills spread thinner; because amplitude and wavelength scale together, grade per metre is unchanged and learned policies still transfer. The legacy generator remains unchanged for older driver files. Attempt limits are max(180, ceil(length in meters / 6)) seconds, leaving time for slower learners. The radial generator keeps a positive radius, so its ordered centerline does not self-intersect; arbitrary intersecting track imports are unsupported. A smooth height field gives it continuous elevation, including across the finish line. Road surfaces, cars, terrain, and elevation charts use that same height field. Road width is 12 m.

Surface grip multiplies one friction budget, so wet reduces cornering grip and drive/brake force together. Cars record whether they are off-road and how far past the tire limit they are asking to corner; that drives gravel plumes and the tire marks left on the road, each a single instanced mesh.

The lightweight planar vehicle model has actual position and velocity, steering-induced yaw, lateral tire force saturation, sliding, acceleration/braking, aerodynamic drag, gravity along the road grade, off-road resistance, and normal-load changes over crests and dips. Elevation changes dynamics, not just appearance.

This is an educational simulation, not a high-fidelity rally simulator. Cars remain attached to the height field; there are no airborne jumps, suspension simulation, body collisions, or damage. The population's cars do not collide with one another. Steering dynamics and tire forces are approximations. Track transfer is warm-start adaptation, not evidence of generalization to all possible tracks. Fastest found is not a proof of global optimality.

## Files

- `dist/engine.mjs` — deterministic track, observations, neural policy, vehicle physics, evolutionary trainer.
- `dist/app.mjs` — 3D scene, real-time simulation controls, graphs, export, import, replay.
- `dist/index.html`, `dist/style.css` — responsive interface.
- `dist/vendor/three.module.js` — Three.js 0.170.0 (MIT; license included).
- `train.mjs` — headless training and export.
- `tests/verify.mjs` — reproducibility, terrain variation, successful learning, exact champion replay, and the driver-import round trip.
- `tests/pace-notes.mjs` — frame-loop recovery, instancing counts, driver links, algorithm comparison, input ablation, diversity, and surface grip.
- `tests/long-courses.mjs` — long-course geometry, overview framing, relief scaling, and link replay.

## Verification

Legacy 822 m circuit, seed 2077, elevation 18 m, trainer seed 99: first lap completed in generation 2 at 43.63 seconds. By generation 20, the fastest lap was 35.47 seconds (18.7% faster), and 743 evaluation laps had completed. The exported champion's replay exactly reproduced its lap time. Results describe this deterministic default configuration, not a guarantee for every seed.


The 3/6/10 km circuits (seed 2077, elevation 18 m, 30 generations) completed in 114.63 / 195.60 / 304.73 seconds, climbing 187 / 375 / 625 m respectively. The 6 and 10 km times are slower than before relief began scaling with course length, because those circuits now climb twice and roughly three times as much; the 3 km circuit is unchanged. These are measured sample runs, not guaranteed convergence times for every map.

## Runtime and rendering improvements

The frame loop schedules again in `finally`, pauses on faults, and shows a retryable error. Population attempts no longer allocate unused point arrays. A packed Float32 trace is generated only for improved completed champions, and just four are retained. Rendering batches the original 1,025 mesh scene into 23 mesh submissions (14 shadow-casting); longer maps add trees and road posts as instances without multiplying draw submissions. These are structural draw-count measurements, not claims of a particular frame-rate gain on every device.

Driver links use Float64 weights so replay is exact. Legacy links carry a 274-character payload; length-aware links use 284 characters. Both formats are validated before replacing a session. Input values and camera controls have keyboard support, progress exposes its numeric value, comparisons have a text table, and motion transitions respect reduced-motion preferences.
