# Rally / Evolution Lab

A runnable machine learning experiment: 48 neural rally drivers evolve on a procedural gravel circuit with uneven elevation. The objective is to complete a full lap, then minimize lap time.

## Run the interactive project

Requires Python 3 and a modern browser with WebGL:

```sh
python3 -m http.server 5173 --directory dist
```

Open http://localhost:5173 and click **Start training**. No installation, account, GPU training service, or API key is needed. Three.js is vendored; online fonts are optional and have system fallbacks.

- **Simulation speed:** 1×, 5×, 20×, or 60×, subject to device performance.
- **Overview / Follow car:** see the full terrain or follow the leading driver. Drag to orbit; scroll to zoom.
- **Seed / elevation:** generate a repeatable track with a different layout or hills. Generating transfers the current best policy but resets lap records and generation history for the new circuit. Clicking Generate with unchanged settings picks a new random seed.
- **Replay best:** run the fastest learned policy at real time. Return to training with the same button.
- **Export driver:** download the best policy, track settings, and lap record as JSON.
- **Import driver:** load an exported driver back in. The saved track is rebuilt, the policy is replayed once to establish its lap time, and the next generation starts from it. Files without track settings load onto the current circuit.
- **Reset learning:** restart from a fresh population on the current track.

Training lives in memory in the current tab. Closing or refreshing it resets the session. Export a driver you want to retain.

## Train without graphics

Requires Node.js 20 or later; there are no npm dependencies:

```sh
node train.mjs --seed 2077 --elevation 18 --generations 100 --output driver.json
node tests/verify.mjs
```

The CLI prints generation results and saves the champion weights with its history. `newCar(track, weights)` and `tickCar(track, car)` replay exported weights; see `tests/verify.mjs` for a complete example. A driver saved here loads into the UI through **Import driver**; training history is not restored.

## Machine learning

The policy is a small neural network with 12 inputs and two tanh output neurons: steering and signed throttle/brake. Its 24 weights are the genome. There is no prerecorded driving path, hidden autopilot, synthetic training chart, or external model call. The visualization reads the actual simulated cars and training results.

Inputs are heading errors to four preview points (8, 18, 35, 60 meters ahead); normalized lateral offset, speed, slip angle, upcoming curvature, grade, tire-load deviation, prior steering; and a constant bias. This agent receives exact track-relative observations, rather than learning from camera pixels.

Initialization uses a noisy road-following prior, not a pretrained policy. This makes a small browser experiment practical. A generation evaluates 48 genomes. Eight top drivers are eligible parents; the all-time champion and five elites are copied unchanged. Other drivers receive crossover and Gaussian mutation with a gradually decreasing noise scale; occasional fresh policies encourage exploration. Evolution operates on full episodes at a fixed 30 Hz timestep.

Incomplete attempts score `9000 × maximum forward lap fraction`, minus a small off-track penalty. Completed laps score `10000 + 40 × (180 − lap seconds)`. Completion therefore dominates incomplete progress; among completed laps the fastest wins. Attempts end after a lap, 180 simulated seconds, prolonged off-road driving, a major departure, reversing too far, or stalling. Progress uses ordered local road segments with start-line wrapping, so simply crossing the finish line or driving backwards cannot count as a lap.

## Track and physics

A seeded radial circuit blends several periodic curves and samples 480 road segments. A smooth height field gives it continuous elevation, including across the finish line. Road surfaces, cars, terrain, and elevation charts use that same height field. Road width is 12 m.

The lightweight planar vehicle model has actual position and velocity, steering-induced yaw, lateral tire force saturation, sliding, acceleration/braking, aerodynamic drag, gravity along the road grade, off-road resistance, and normal-load changes over crests and dips. Elevation changes dynamics, not just appearance.

This is an educational simulation, not a high-fidelity rally simulator. Cars remain attached to the height field; there are no airborne jumps, suspension simulation, body collisions, or damage. The population's cars do not collide with one another. Steering dynamics and tire forces are approximations. Track transfer is warm-start adaptation, not evidence of generalization to all possible tracks. Fastest found is not a proof of global optimality.

## Files

- `dist/engine.mjs` — deterministic track, observations, neural policy, vehicle physics, evolutionary trainer.
- `dist/app.mjs` — 3D scene, real-time simulation controls, graphs, export, import, replay.
- `dist/index.html`, `dist/style.css` — responsive interface.
- `dist/vendor/three.module.js` — Three.js 0.170.0 (MIT; license included).
- `train.mjs` — headless training and export.
- `tests/verify.mjs` — reproducibility, terrain variation, successful learning, exact champion replay, and the driver-import round trip.

## Verification

Default seed 2077, elevation 18 m, trainer seed 99: first lap completed in generation 2 at 43.63 seconds. By generation 20, the fastest lap was 35.47 seconds (18.7% faster), and 743 evaluation laps had completed. The exported champion's replay exactly reproduced its lap time. Results describe this deterministic default configuration, not a guarantee for every seed.
