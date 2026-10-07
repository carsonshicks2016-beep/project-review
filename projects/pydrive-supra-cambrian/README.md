# Supra Drift — Teaching AI to Master the Nürburgring

A from-scratch reinforcement learning simulator built in Python, training a **Mazda 787B** to drive a superhuman lap on the **Nürburgring Nordschleife** — the most dangerous and demanding race circuit on Earth.

The project features custom four-wheel vehicle dynamics (Pacejka tyres, combined-slip grip, weight transfer, turbo spooling), 2.5D terrain with emergent jumps, real-time engine audio synthesis, and a multi-stage training pipeline benchmarked against **Stefan Bellof's legendary 6:11.13** — the lap that stood for 35 years.

---

## The Fable Five Pipeline — Chasing Bellof

The centrepiece of this project is **Fable Five** (`supra/fable5.py`): a dedicated pipeline that trains the 787B's quad-rotor R26B to attack the full 20.8 km Nordschleife.

A **physics-true speed envelope** computed from the real car model feeds the AI's observations — it sees the correct speed for its position plus 6 points down the road, letting it anticipate braking zones from physics rather than trial-and-error. Five gated stages push the car from "survive" to "superhuman":

| Stage | Goal | Gate |
|-------|------|------|
| **Foundation** | Learn the 170+ corner layout at safe pace | Basic lap completion |
| **Flow** | Connect sectors with smooth, linked inputs | Sector consistency |
| **Finish** | Close a full clean lap (30% of training spawns in the final sectors) | Clean flying lap |
| **Fast** | Push the speed envelope to 96% of physics capability | Adaptive segments |
| **Frontier** | Exceed the envelope — find racing-line gains beyond the centerline floor | Scale up to 1.15× |

An automated **Pit Wall supervisor** watches every evaluation: it rolls back collapses immediately, reseeds from the best brain by failure mode, and decays the learning rate when refinement stalls — keeping overnight runs productive instead of grinding dead policies for hours.

**Benchmarks the AI races against:**
- 🏁 Theoretical centerline lap: **~6:40**
- 🏆 Stefan Bellof '83 (Group C): **6:11.13**
- 🚀 Porsche 919 Evo '18 (LMP1): **5:19.55**

→ [Full Fable Five documentation](docs/04_Fable_Five_Pipeline.md)

---

## Faithful-v2 Porsche 919 Evo Record Program

The Porsche effort is now a separate, identity-isolated program under
`supra/faithful/` and `supra/record/`. Its contract is the official 20.832 km
T13 flying lap, strictly below **319.546 seconds**, with three mandatory
boundaries: licensed-data physics validation, an authoritative MuJoCo replay of
a CasADi/IPOPT whole-lap oracle, and independent certification of a
driver-equivalent recurrent policy.

The stopped legacy 919 run is preserved as
`legacy_misstamped_noncertifiable`. It cannot be resumed or promoted because it
carried Mazda drivetrain/global-artifact provenance. The new program never
tunes power, grip, aero, geometry or braking to force the benchmark; missing
licensed tyre, aero, suspension, control, telemetry or June 2018 survey data
keeps the faithful and record gates closed.

The checked-in foundation includes immutable physical/telemetry schemas, a
signed external evidence vault with permanent fixture labeling and holdout
audit, a digest-pinned Linux authority-runtime contract, a runnable
noncertifying MuJoCo baseline, fail-closed MJX and oracle boundaries,
edition/run artifact isolation, adversarial flying-lap and certificate
validators, the recurrent driver network, an original dimensionally anchored
919 GLB, and a telemetry-only A/V adapter. It does not yet include the licensed
authority model, correlated MJX twin, whole-lap CasADi/MPC solver, faithful
audio runtime, or end-to-end training/certification runner. It therefore
remains a **telemetry-constrained approximation scaffold** until those
implementations and the private licensed evidence are supplied and validated.

All claim-producing APIs currently fail closed: the protocol trust store starts
empty, fixtures can never certify, and self-authored hashes, straight-line
timing traces, or scaffold certificates cannot open physics, oracle, training,
or record gates.

→ [Faithful 919 record-program contract](docs/07_Faithful_919_Record_Program.md)

---

## The 787B — Built from Reference

The Mazda 787B is modeled from real specifications: the quad-rotor R26B engine with proper firing harmonics, overrun voice, and inter-rotor imbalance beating. The car art (`supra/carart.py`) reproduces the cab-forward bubble canopy, long kamm tail, and Renown QUADRANT livery (orange/green swap at the cockpit, white spine stripe). The drivetrain is versioned as `mazda787b-5spd-ring-v1`: an authentic five-forward-gear Mazda–Porsche architecture with Nordschleife-optimized ratios. Its torque-aware `RaceBox` unloads the clutch, shifts sequentially, re-engages, walks down through braking zones, and rejects money shifts; the AI retains a bounded gear-offset action on top of that safe recommendation.

The real-time audio engine synthesizes the R26B's scream live from telemetry: no samples, pure additive synthesis with load-dependent harmonics, turbo flutter, exhaust crackle on overrun, tyre squeal, and kerb-strike rattle.

→ [Physics model deep dive](docs/02_Physics_Model.md) · [Audio synthesis](docs/06_Audio_Synthesis.md)

---

## The Simulator

Beyond the Nordschleife program, Supra Drift is a complete driving AI platform:

- **Custom Physics**: Per-corner Pacejka tyres, combined-slip friction ellipse, tyre relaxation length, weight transfer, limited-slip diff, twin-turbo spooling, 2.5D hills with emergent jumps
- **Multiple AI Paths**: Genetic Algorithm (NumPy MLP brains) and PPO reinforcement learning (PyTorch actor-critic with GAE)
- **Race & Drift**: One network architecture trains both — just swap the reward and flip a mode flag
- **5 Vehicle Presets**: Toyota Supra, Mazda RX-7, Nissan Skyline R34, Land Rover LR4, Mazda 787B
- **Procedural Tracks**: GP circuits, technical hairpins, speedways, touge switchbacks — all with seed-deterministic elevation
- **Command Center**: Flask web dashboard with live SSE metric streaming, the Pit Wall particle-swarm training screen, Brain Lab diagnostics, and the Fable Observatory launcher
- **Fable Five 3D Brain Observatory**: Local Three.js visualization of validated 787B and legacy-approximation 919 Evo Fable PPO brains, driven exclusively by authoritative Python playback and brain probes
- **Drive It Yourself**: Full manual control with handbrake, clutch, manual gears, weather cycling, broadcast director camera, and replay

```bash
# Install
pip3 install -r requirements.txt

# Train the 787B on the Nordschleife
python3 run.py --fable 30000 --fable-stage auto

# Watch it drive
python3 run.py --watch-fable

# Or launch the Command Center and open 3D brain playback
python3 command-center/server.py    # -> http://localhost:8770/observatory/

# Drive it yourself
python3 run.py --drive --track club --car supra

# Launch the Command Center dashboard
python3 command-center/server.py    # → http://localhost:8770
```

---

## Documentation

| Guide | Contents |
|-------|----------|
| 📖 [Overview & Installation](docs/01_Overview.md) | Dependencies, architecture, controls |
| 🏎️ [Physics Model](docs/02_Physics_Model.md) | Pacejka tyres, combined-slip, 2.5D hills, emergent jumps |
| 🧠 [Neural Networks & Training](docs/03_Neural_Networks_and_Training.md) | 60-dim observation space, GA, PPO, race vs drift rewards |
| ⏱️ [Fable Five Pipeline](docs/04_Fable_Five_Pipeline.md) | The Nordschleife 787B superhuman program |
| 🖥️ [UI & Visualization](docs/05_UI_and_Visualization.md) | Command Center, unchanged 2.5D viewer, and the Fable 3D Brain Observatory |
| 🔊 [Audio Synthesis](docs/06_Audio_Synthesis.md) | Real-time additive engine/exhaust/environment synthesis |
| 🧾 [Faithful 919 Record Program](docs/07_Faithful_919_Record_Program.md) | Evidence gates, authority/twin, oracle, driver boundary and certification |

Historical design documents and technical handoffs are preserved in [docs/legacy/](docs/legacy/).
