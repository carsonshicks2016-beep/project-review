# Off-Road Rover Research Command Center 🚜
### 3D Multi-Body Physics Simulation, 16-Beam LIDAR Telemetry & Neuroevolution Operations

A research-grade 3D simulation and neuroevolution operations platform modeling an overland 4x4 rover learning to navigate procedural, uneven terrain, boulder fields, cross-axial moguls, and steep rock walls. Combines a **100 Hz internal multi-body physics solver** (semi-implicit Euler with quaternion kinematics) with a **20 Hz policy control loop**, live **16-beam raycast LIDAR radar**, **Center-of-Mass plumb-bob tracking**, **dynamic 3D tire traction vectors**, and a **full training management suite**.

---

## Research Command Center Decks

The dashboard is organized into 5 research decks accessible via the top navigation rail:

1. **🎮 3D Telemetry & Viewport:**
   * High-framerate Three.js WebGL simulation viewport with realistic terrain shading, shadows, and dynamic spotlights.
   * Multi-Camera Director: **Chase Cam**, **Free Orbit**, **Cockpit POV**, and **Top-Down Aerial**.
   * Toggleable 3D Overlays: 16-beam raycast LIDAR lasers, Center-of-Mass plumb-bob & attitude gimbal ring, 3D tire force vectors, and boulder collision wireframes.
   * Cockpit HUD: Analog/digital speedometer, inclinometer (pitch/roll with critical rollover alerts), 4-corner suspension travel meters (FL, FR, RL, RR) in millimeters, underbelly skid plate clearance, and 150m course progress.
   * **Manual Drive Mode Override (WASD / Arrows):** Take manual control of the vehicle in real-time to test human driving skill on the terrain.
   * Real-time Canvas oscilloscopes: Course elevation cross-section and 6-DOF attitude stability horizon.

2. **⚡ Training Ops & Evolution Deck:**
   * Real-time neuroevolution training orchestrator.
   * Training actions: **Start Continuous Evolution**, **Pause Evolution**, **Step 1 Generation**, and **Reset Population**.
   * Live hyperparameter tuning:
     * **Mutation Rate ($\mu$):** $0.02 \text{ to } 0.40$
     * **Mutation Strength ($\sigma$):** $0.05 \text{ to } 0.60$
     * **Elitism Count:** $1 \text{ to } 12$
     * **Tournament Selection Size ($k$):** $2 \text{ to } 8$
   * Dual training progress charts:
     * Generational Fitness Convergence (Best vs Mean vs Elite average).
     * Course Traversal Distance (m) vs Rollover/Flip Rate (%).

3. **🧬 Population & Genome Inspector:**
   * Interactive roster table listing every individual evaluated in the active generation (40+ agents).
   * Detailed metrics per agent: Rank, ID, Fitness Score, Max Distance Reached, Steps Survived, and Rollover Status (Upright vs Flipped).
   * **Individual Agent Replay:** Click **"Watch Replay"** on any row to immediately load that exact agent's run into the 3D viewport.
   * Neural network topology diagram ($30 \to 32 \to 16 \to 2$ MLP with 1,554 weights) and input feature saliency weights.

4. **📡 LIDAR & Sensory Analytics:**
   * Polar radar scanner displaying all 16 raycasts in a top-down $\pm 55^\circ$ forward azimuth sweep with 2m, 5m, and 10m range rings.
   * Sector proximity breakdown (Long-range forward fan, near-bumper rock detector, left/right wheel-track terrain scanners).
   * Tri-axial Accelerometer ($a_x, a_y, a_z$) and Gyroscope ($p, q, r$) telemetry in body coordinates.

5. **📐 Vehicle Dynamics & Invariants:**
   * Rigorous mechanical specification table: 800 kg chassis mass, 2.2m wheelbase, 1.6m track width, Center of Mass offset, and principal inertia tensor ($I_{xx}, I_{yy}, I_{zz}$).
   * 4-corner suspension spring stiffness ($18,500\text{ N/m}$), damping rate ($2,400\text{ N}\cdot\text{s/m}$), and anti-roll bar coupling ($3,500\text{ N/m}$).
   * Empirical Pacejka tire traction curve ($F_x$ vs longitudinal slip, $F_y$ vs slip angle).
   * Course topography specs across the 5 distinct obstacle zones.

---

## Quickstart & Launching

* **Launch Full Research Command Center (with Backend API):**
  Double-click [Launch Rover Lab.command](file:///Users/REVIEW_USER/Desktop/Launch%20Rover%20Lab.command) on your Desktop, or run:
  ```bash
  cd /Users/REVIEW_USER/Desktop/offroad-rover-rl
  "/Users/REVIEW_USER/Desktop/melee bot/.venv/bin/python" dashboard/server.py 8780
  ```
  *(Starts the local Python API server and opens http://127.0.0.1:8780/index.html in your browser).*

* **Open 3D Dashboard Standalone (No Server Required):**
  Double-click [Open Rover Dashboard.command](file:///Users/REVIEW_USER/Desktop/Open%20Rover%20Dashboard.command) on your Desktop, or open:
  * [3D Rover Telemetry Dashboard](file:///Users/REVIEW_USER/Desktop/offroad-rover-rl/dashboard/static/index.html)
  *(Runs completely offline in any browser with pre-bundled simulation trajectories).*

* **Evaluate Policy Benchmark via Terminal:**
  ```bash
  cd /Users/REVIEW_USER/Desktop/offroad-rover-rl
  "/Users/REVIEW_USER/Desktop/melee bot/.venv/bin/python" evaluate.py
  ```
