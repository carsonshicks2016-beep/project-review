# Stage 1 Notes: Warmup on Walker2d-v5

## 1. Objective & Setup
- **Goal**: Establish and validate the complete continuous-control robotics training pipeline, CleanRL-style vectorized PPO, TensorBoard logging, and evaluation `.mp4` video recording on a fast 2D biped before advancing to the 3D 348-dimensional Humanoid.
- **Environment**: `Walker2d-v5` (MuJoCo 3.9.0, Gymnasium 1.2.3).
- **Execution Platform**: macOS Apple Silicon (12 logical cores).
- **Throughput**: ~7,000–14,000 steps per second across 8 parallel environments (`AsyncVectorEnv`), completing 1,000,000 steps in **2.5 minutes**.

---

## 2. Robotics & Physics Deep-Dive

### Observation Space Design (17 Dimensions)
The observation vector in MuJoCo consists of generalized positions $\mathbf{q} \in \mathbb{R}^8$ and generalized velocities $\mathbf{\dot{q}} \in \mathbb{R}^9$:

1. **Generalized Positions (`qpos[1:]`, 8 dims)**:
   - Note: The global horizontal position $x$ is omitted from observations to ensure **translation invariance** (the policy should behave identically whether it is at $x = 0$ or $x = 100\text{ m}$).
   - `obs[0]` ($z$): Torso vertical elevation above ground (nominal standing $\approx 1.25\text{ m}$).
   - `obs[1]` ($\theta_y$): Torso pitch angle relative to the gravity vector (rad).
   - `obs[2:5]`: Right leg joint angles (thigh/hip, leg/knee, foot/ankle).
   - `obs[5:8]`: Left leg joint angles (thigh/hip, leg/knee, foot/ankle).

2. **Generalized Velocities (`qvel`, 9 dims)**:
   - `obs[8]` ($\dot{x}$): Torso forward horizontal velocity (the primary locomotion objective).
   - `obs[9]` ($\dot{z}$): Torso vertical velocity.
   - `obs[10]` ($\dot{\theta}_y$): Torso pitch angular velocity.
   - `obs[11:14]`: Right leg joint angular velocities ($\text{rad/s}$).
   - `obs[14:17]`: Left leg joint angular velocities ($\text{rad/s}$).

3. **External Forces & Contact Dynamics**:
   - In `Walker2d-v5`, contact forces are implicit in velocity changes and state transitions. In `Humanoid-v5` (Stage 2), explicit 6D external spatial contact forces (`cfrc_ext` for each body segment) are appended to the observation space, allowing the robot to feel ground reaction forces directly.

### Simulation Dynamics & Timestepping
- **MuJoCo Base Timestep**: $dt = 0.002\text{ s}$ ($500\text{ Hz}$).
- **Frame Skip / Substepping**: `frame_skip = 4`.
- **Policy Control Step**: $\Delta t = 4 \times 0.002\text{ s} = 0.008\text{ s}$ ($125\text{ Hz}$).
- **Actuators**: 6 continuous torque motors driving the 3 joints of each leg with symmetric torque limits $[-1, 1]$. Gear ratios in the MJCF scale normalized actions to joint torques (Nm).

### Reward Function Structure & Tradeoffs
$$R_t = R_{\text{forward}} + R_{\text{healthy}} - R_{\text{ctrl}}$$

1. **Forward Velocity Reward**:
   $$R_{\text{forward}} = 1.0 \times \frac{x_t - x_{t-1}}{\Delta t} = \dot{x}$$
   - *Behavior*: Linearly incentives forward velocity.
   - *Pitfall*: If weighted too heavily relative to survival, the policy initiates an uncontrolled sprint, leans forward excessively, and faceplants.

2. **Healthy / Survival Bonus**:
   $$R_{\text{healthy}} = +1.0 \quad \text{if } z \in [0.8, 2.0]\text{ m} \text{ and } \theta_y \in [-1.0, 1.0]\text{ rad}$$
   - *Behavior*: Subsidizes staying upright and un-fallen.
   - *Pitfall*: If the forward velocity reward is too weak or control cost is too high, the agent discovers a **degenerative local optimum**: standing completely still like a statue, collecting $+1.0$ per step for 1,000 steps without moving forward!

3. **Control Cost / Actuation Penalty**:
   $$R_{\text{ctrl}} = 10^{-3} \times \sum_{i=1}^6 u_i^2$$
   - *Behavior*: Encourages smooth, energy-efficient joint torques.
   - *Tradeoff*: Suppresses high-frequency bang-bang chatter and joint limit banging.

---

## 3. Training Dynamics & Learning Progression

| Iteration | Timesteps | Mean Return | Episode Length | Forward Speed | Key Behavioral Milestone |
|---|---|---|---|---|---|
| **Init** | 0 | ~15.0 | ~12 steps | 0.05 m/s | Random falling; collapses within 10–15 steps. |
| **Early** | 81k | 246.65 | 132 steps | 0.85 m/s | Learns to keep torso upright; takes 3–4 tentative steps. |
| **Mid** | 491k | 286.01 | 146 steps | 0.98 m/s | Coordinated alternate leg swing; stabilizes pitch oscillation. |
| **Late** | 819k | 350.99 | 166 steps | 1.08 m/s | Smooth continuous heel-to-toe walking cycle. |
| **Final** | 1.0M | **365.44** | **173.6 steps** | **1.11 m/s** | Sustained stable forward walking gait. |

---

## 4. Key Takeaways & Validation for Stage 2
1. **Pipeline Confirmed**: Vectorized PyTorch execution runs at >10,000 SPS on this Mac, meaning the larger Humanoid-v5 (tens of millions of steps) can be trained efficiently.
2. **Deterministic Evaluation**: Policy achieves consistent returns ($365.44 \pm 2.73$) across seeds with zero jitter.
3. **Artifacts Checkpointed**:
   - Best weights: `humanoid_parkour/models/stage1_walker2d_best.pt`
   - Evaluation video: `humanoid_parkour/videos/stage1_eval_best.mp4`
   - TensorBoard runs: `humanoid_parkour/runs/`
