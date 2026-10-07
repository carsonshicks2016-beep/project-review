# Supra Drift Overview

Supra Drift is a from-scratch neuro-evolution and reinforcement-learning **drifting & grip-racing simulator** built in Python. It features custom four-wheel vehicle dynamics, procedural tracks, a genetic-algorithm path and a PyTorch PPO path (for both race and drift), live engine audio synthesis, an in-app watch viewer, and a self-contained web dashboard known as the Command Center.

The project is built entirely in 2D (top-down) but models real vehicle physics across a 2.5D space. The primary training vehicle is a Toyota Supra, with R34 Skyline, Mazda RX-7, and Mazda 787B presets available. 

## High-Level Status & Features

| Area | Feature Details | State |
|------|------|-------|
| **Sim** | Configuration, four-wheel physics, procedural tracks, drivable 2D app | ✅ |
| **Hills & Jumps** | 2.5D slope physics with emergent airborne mode (tracks have real elevation) | ✅ |
| **Sensors** | 9 beams, 25 proprioceptive signals, look-ahead curvature, 18-dim hill/air block (60-dim obs total) | ✅ |
| **GA** | NumPy MLP brains, elitism + tournament + crossover + mutation (race) | ✅ |
| **PPO Race** | PyTorch actor-critic, GAE, curriculum, deterministic eval + keep-best | ✅ Completes laps |
| **PPO Drift** | Slip-angle reward, anti-spin shaping, completion-gated curriculum | ✅ Drifts (hardest touges still spin out) |
| **Hybrid** | One goal-conditioned race/drift policy | ⏳ Next steps |

## Quick Start & Dependencies

The simulator is built with the following core dependencies:
- **NumPy**
- **Pygame**
- **PyTorch** (CPU-only, no GPU required)
- **Flask** (For the Command Center dashboard)
- **Sounddevice** (Optional, for live audio synthesis)

### Installation
To install the dependencies on a Mac (using `pip3` and `python3`):
```bash
pip3 install -r requirements.txt
```

### Command Center (Web Dashboard)
The Command Center is the one-stop control panel for everything: driving, training (GA / PPO race / PPO drift), watching, managing checkpoints, and browsing tracks. It is an all-Python Flask application.

To start the Command Center:
```bash
python3 command-center/server.py
```
Then navigate to `http://localhost:8770`.

Alternatively, use the launcher scripts: `~/Desktop/Supra Command Center.command` or `command-center/SUPRA.command` in the repository.

### Driving Manually
You can drive the car yourself using the `run.py` CLI:
```bash
# Drive on a named track
python3 run.py --drive --track club             

# Drive the tail-happy RX-7 on a touge track
python3 run.py --drive --track akina --car rx7  

# Drive a procedurally generated technical track
python3 run.py --drive --gen technical --difficulty 0.85 --length 1400
```

**Controls:** 
- **Arrows / WASD**: Drive
- **Space**: Handbrake
- **Shift**: Clutch
- **T**: Toggle auto gearbox
- **Q / E**: Manual down/up shift
- **C**: Camera mode
- **V**: Broadcast director
- **1-5**: Direct director shots
- **P**: Replay
- **I**: Complete AI Vision overlay
- **B**: Sensor beams
- **G / H / Y / O**: Visual toggles (time of day, weather, wet surface, HQ post-processing)
- **F3**: Performance diagnostics
- **F11**: Fullscreen
- **M**: Mute audio
- **R**: Reset 
- **Esc**: Quit

## Core Architecture & Directory Structure

- **`run.py`**: The CLI entry point. Handles arg parsing and dispatch to all modes (drive, train, watch).
- **`command-center/`**: The web dashboard (Flask + vanilla JS).
- **`viewer3d/`**: The standalone browser 3D drive viewer.
- **`supra/`**: The core simulation and intelligence logic.
  - `config.py`: Dataclasses for CarSpec, SimSpec, SensorSpec, PPOSpec, Rewards, etc.
  - `physics.py`: 4-wheel dynamics, Pacejka tyres, weight transfer, drivetrain.
  - `track.py`: Catmull-Rom circuits, raycasts, procedural generators.
  - `sensors.py`: Generates the 60-dim observation vector.
  - `brain.py` & `evolution.py`: NumPy MLP genome and Genetic Algorithm logic.
  - `ppo.py` & `ppo_env.py`: PyTorch PPO actor-critic implementation and Gym-style environment.
  - `drift.py`: Drift scoring and anti-spin rewards.
  - `app.py`: The PyGame drive/watch loop, dashboard, and visualization.
  - `fable5.py`: The dedicated Nordschleife superhuman pipeline.
  - `sound.py`: Live real-time engine audio synthesis.
