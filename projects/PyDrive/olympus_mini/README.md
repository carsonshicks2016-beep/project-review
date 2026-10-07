# Olympus Mini

A modular, lightweight, high-performance reinforcement learning sandbox in **MuJoCo** and **PyTorch** designed to discover optimal biomechanical form across three classic track-and-field events:

1. **100m Sprint**: Maximizing forward impulse and stride cadence while minimizing ground contact time.
2. **Hurdles**: Clearing 1.06m obstacles while keeping center-of-mass (CoM) flight flat (lead-leg snap + trail-leg tuck).
3. **Pole Vault**: Kinetic approach $\rightarrow$ elastic pole compression $\rightarrow$ inversion & recoil $\rightarrow$ crossbar clearance.

---

## Interactive Web Laboratory Dashboard

Olympus Mini includes a built-in dark-mode real-time laboratory dashboard where you can watch the simulation, train interactively yourself using keyboard controls, and launch background PPO training runs.

### Launch the Lab:
```bash
python3 run_lab.py --port 5050
```
Open **`http://localhost:5050`** in your web browser.

### Dashboard Features:
* **Live Video Viewport**: Zero-lag MJPEG stream directly from MuJoCo with selectable cameras (**Side Elevation**, **Orbit**, **Chase Cam**, **Frontal**).
* **Train Myself (Manual Drive Mode)**:
  * Toggle between **Autonomous AI** and **Manual Drive**.
  * Use keyboard controls to control gait and balance in real time:
    * `[W / S]` or `[Up / Down]`: Modulate forward/backward torso lean.
    * `[A / D]` or `[Left / Right]`: Increase/decrease stride cadence (Hz).
    * `[Spacebar]`: Trigger explosive jump impulse.
* **Biomechanical Telemetry Deck**:
  * **Ground Reaction Force (GRF) Heatmap**: Dual-foot sole diagrams showing 4 contact pads per foot (toe inner/outer, heel inner/outer) that illuminate dynamically upon ground impact.
  * **Knee Phase-Plane Orbit Canvas**: Real-time $(\theta_{\text{knee}}, \dot{\theta}_{\text{knee}})$ limit-cycle attractor plot visualizing walking/running dynamic stability.
  * **Real-time Metrics**: Velocity ($v_x$), distance, CoM height, gait cycle phase, hurdle clearance counter, and peak pole vault apex height.
* **Training Hub**:
  * Launch, pause, and monitor background PPO training directly from the UI.
  * Live learning curves plotting episode return and loss metrics.
  * Automatically saves new champion checkpoints.
* **1-Click GIF Capture**:
  * Click **"Capture GIF"** to record a 3-second highlight reel with telemetry HUD overlay directly to `olympus_mini/lab/recordings/`.

---

## Command-Line Training & Demos

### 1. Run Unit Tests (11/11 passing)
```bash
python3 -m unittest discover -s olympus_mini/tests
```

### 2. Render an Animated GIF from CLI
```bash
# Render a 100m sprint rollout:
python3 -m olympus_mini.demo --task sprint --max-steps 100 --render-gif olympus_mini/sprint_demo.gif

# Render the hurdle traversal track:
python3 -m olympus_mini.demo --task hurdle --max-steps 100 --render-gif olympus_mini/hurdle_demo.gif

# Render the pole vault approach and plant box:
python3 -m olympus_mini.demo --task vault --max-steps 100 --render-gif olympus_mini/vault_demo.gif
```

### 3. CLI PPO Training
```bash
# Train the 100m sprint policy:
python3 -m olympus_mini.train --task sprint --total-steps 50000 --rollout-steps 1024

# Train the Hurdle traversal policy:
python3 -m olympus_mini.train --task hurdle --total-steps 50000

# Train the multi-task athlete across all disciplines:
python3 -m olympus_mini.train --task multi --total-steps 100000
```
