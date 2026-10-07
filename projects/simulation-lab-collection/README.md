# 🏎️💨 ML Drift Master: Reinforcement Learning for High-Speed Drifting & Trick Chaining

Teach autonomous AI racecars to drift at extreme slip angles, carry maximum cornering velocity, clip apexes, and chain together complex stunt maneuvers (Manji transitions, Scandinavian flicks, backward entries, wall taps, 360 entries) to achieve legendary arcade-style combo scores!

![Drift Action](gymkhana_drift_screenshot.png)

---

## 🌟 Highlights

- **Dynamic Bicycle Model with Pacejka Magic Formula**: Realistic non-linear tire friction curves, longitudinal weight transfer during braking and acceleration, counter-steering dynamics, and friction circle rear-wheel spin.
- **Arcade Stunt & Trick Detection Engine**:
  - 🔄 **Scandinavian Flick / Feint**: Counter-steer away before snapping violently into the slide.
  - ⚡ **Manji / Chokudori Transitions**: Chaining alternating slides (left-to-right) down straights without breaking combo.
  - 🔙 **Backward Entry**: Entering corners at extreme slip angles ($> 75^\circ$) at high speed and cleanly recovering.
  - 🧱 **Wall Tap / Barrier Kiss**: Drifting rear bumper within centimeters of guardrails at speed without crashing.
  - 🎯 **Clipping Point Precision**: Hitting inner apex and outer clipping zones for massive bonus multipliers.
  - 🌪️ **360 Drift Entry**: Seamless $360^\circ$ continuous rotation initiated before corner entry.
- **Continuous Gymnasium Environment (`DriftGymkhanaEnv`)**:
  - 35-dimensional continuous state vector: normalized chassis velocities, slip angle, yaw rate, combo state, waypoints, and **16-ray LIDAR**.
  - Multi-track support: **Touge Mountain Pass**, **Gymkhana Playground Arena**, and **Formula Drift Stadium**.
- **Pygame Real-Time Visualizer**:
  - Dynamic tire smoke particle physics billowing behind sliding wheels.
  - Persistent rubber skid marks along tire trajectories.
  - Arcade HUD: Drift angle meter, speedometer, combo multiplier bar, and animated trick popup banners.
  - **Human vs AI Playable Mode**: Grab the keyboard (WASD / Arrows + Spacebar for Handbrake) and see how hard real drifting is against the AI!

---

## 🕹️ Quick Start: Play in Human Drive Mode

You can immediately test the vehicle dynamics and stunt engine yourself using keyboard controls:

```bash
python3 -m drift_rl.visualizer --track gymkhana
```

### Controls
| Key | Action |
|---|---|
| **W / Up Arrow** | Throttle / Acceleration |
| **S / Down Arrow** | Footbrake / Reverse |
| **A / Left Arrow** | Steer Left |
| **D / Right Arrow** | Steer Right |
| **Spacebar** | **Handbrake (E-Brake)** (Locks rear wheels into instant oversteer slide) |
| **M** | Toggle between **Human Mode** and **AI Autonomous Mode** |
| **R** | Reset car position and clear skid marks |
| **1 / 2 / 3** | Switch tracks: 1 = Touge Pass, 2 = Gymkhana Arena, 3 = Formula Stadium |

---

## 🧠 Training AI Drift Agents

Train continuous control agents using **PPO** or **SAC** (via `stable-baselines3`):

```bash
# Train PPO agent on Touge Pass
python3 -m drift_rl.train --algo ppo --track touge --timesteps 50000

# Train Soft Actor-Critic (SAC) agent on Gymkhana Arena
python3 -m drift_rl.train --algo sac --track gymkhana --timesteps 50000
```

### Options
- `--algo`: `ppo` (default) or `sac`
- `--track`: `touge` (default), `gymkhana`, or `stadium`
- `--timesteps`: Total training environment steps (e.g. `100000`)
- `--lr`: Learning rate (default `3e-4`)
- `--save-dir`: Checkpoint output folder (default `models/`)
- `--log-dir`: Tensorboard logs (default `logs/`)

---

## 📊 Evaluation & Telemetry Benchmarking

Benchmark any trained checkpoint and get a complete drift telemetry report (mean drift score, peak combo multiplier, average drift angle, speed, and trick counts):

```bash
python3 -m drift_rl.evaluate --model models/drift_ppo_touge_final.zip --track touge --episodes 10
```

Sample output:
```text
============================================================
 DRIFT TELEMETRY SUMMARY
============================================================
 Mean Drift Score:      42,850.0 pts
 High Score:            78,400.0 pts
 Max Combo Multiplier:  x6.4
 Mean Drift Angle:      34.8°
 Peak Drift Angle:      62.1°
 Mean Drift Speed:      48.2 km/h (13.4 m/s)
 Tricks Executed:       19 total
   - MANJI TRANSITION: 11x
   - CLIPPING ZONE (INSIDE): 4x
   - WALL TAP: 2x
   - SCANDINAVIAN FLICK: 2x
 Crash Rate:            10.0%
 Spinout Rate:          0.0%
============================================================
```

---

## 🧪 Unit Tests

Run the full automated test suite verifying dynamics, trick recognition, and Gym compliance:

```bash
python3 -m unittest discover -s tests_drift -p "test_*.py"
```

---

## 📁 Project Structure

```
drift_rl/
├── __init__.py         # Package entry point
├── dynamics.py         # 2D/3D dynamic bicycle physics with Pacejka tire curves
├── tracks.py           # Touge, Gymkhana, and Stadium track layouts & 16-ray LIDAR
├── scoring.py          # Drift scoring & arcade trick detection engine
├── env.py              # Gymnasium environment (DriftGymkhanaEnv)
├── train.py            # Stable-Baselines3 PPO & SAC training pipeline
├── evaluate.py         # Telemetry benchmark runner & JSON trajectory exporter
└── visualizer.py       # Interactive Pygame GUI with smoke particles & HUD
tests_drift/
├── test_dynamics.py    # Unit tests for tire friction, weight transfer & RK4
├── test_scoring.py     # Unit tests for trick detection & combo multiplier
└── test_env.py         # Unit tests for Gym API compliance and track resets
```
