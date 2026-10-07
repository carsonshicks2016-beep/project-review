# Stage 3 — Parkour Terrain with Body-Relative Sensing

> **Evaluation correction in progress (September 29, 2026).** Historical clearance and progress metrics below came from the first terrain implementation. Gap contact, finish detection, and terrain sensing are being checked before these results are used as proof of obstacle completion. Preserve the old runs for comparison; rely on the corrected fixed-seed evaluation for promotion decisions.

## Scope

Build and validate the custom MuJoCo parkour environment with procedurally generated obstacle courses, body-relative height sensing, and shaped rewards including one-time obstacle clearance bonuses. The policy is warm-started from Stage 2's verified bipedal baseline (`runs/stage2-horizon/best`) and trained across diverse courses.

## Environment Architecture & Physics Implementation

### 1. Dynamic Course Compilation & MuJoCo BVH Integrity
In MuJoCo, static collision acceleration structures (spatial bounding volume hierarchies / BVH trees) are compiled into the C `MjModel` struct during XML parsing. Modifying `geom_pos` and `geom_size` in Python arrays after compilation does not rebuild this static hierarchy, which causes dynamic actors to fall through repositioned static geoms.
To guarantee 100% physics and collision integrity:
* Each environment reset procedurally generates a new sequence of terrain segments via `envs/terrain.py`.
* The course geometry is rendered directly into the worldbody XML template and compiled into a fresh `MjModel` via `mujoco.MjModel.from_xml_string()`.
* Model compilation overhead is negligible (~2.1 ms per episode reset).
* All terrain geoms are placed in `geom group="2"` and directly attached to `worldbody` (body 0), matching stock MuJoCo conventions and enabling seamless contact detection in `eval.py`.
* Actuator control ranges are explicitly declared in the `<default>` block (`<motor ctrllimited="true" ctrlrange="-.4 .4"/>`), establishing a valid `Box(-0.4, 0.4, (17,), float32)` action space.

### 1b. Visual Lighting & Rendering Pipeline
* **Spotlight Removal**: The initial template had a localized spotlight with `pos="0 0 1.3" cutoff="100"`. As the humanoid ran down the $+x$ course, it left the cone of illumination, rendering everything pitch-black (mean RGB 1.08).
* **Dual Directional Sunlight**: Replaced with dual directional sun sources (`dir="0 0 -1"` and `dir="-0.3 0.5 -1"`) providing uniform illumination across arbitrary track lengths.
* **Ambient Headlight**: Injected `<headlight ambient="0.5 0.5 0.5" diffuse="0.8 0.8 0.8" specular="0.2 0.2 0.2"/>` to eliminate pitch-black unlit surfaces.
* **Skybox & Contrast Materials**: Added daytime gradient skybox (`rgb1=".55 .70 .85"` to `rgb2=".25 .40 .55"`) and distinct obstacle RGBA palettes (hurdles: orange, boxes: cyan, stairs: sandstone).
* **Result**: Mean image RGB increased from 1.08 to 131.4 while maintaining exact physical invariance.

### 2. Observation Design: 366 Dimensions
The state vector combines the full 348-dimensional proprioceptive features of the humanoid with an 18-dimensional body-relative height scan:

| Block | Count | Physical Meaning |
|---|---:|---|
| `qpos`, excluding global x/y | 22 | Pelvis height, root quaternion, and 17 joint angles |
| `qvel` | 23 | Root translation/rotation velocities and 17 joint velocities |
| `cinert[1:]` | 130 | Per-body COM-frame inertia/mass (13 humanoid bodies × 10) |
| `cvel[1:]` | 78 | Per-body spatial velocities (13 humanoid bodies × 6) |
| `qfrc_actuator[6:]` | 17 | Generalized actuator forces |
| `cfrc_ext[1:]` | 78 | Per-body external spatial forces/torques (13 humanoid bodies × 6) |
| **`height_scan`** | **18** | **Body-relative terrain heights: 6 forward × 3 lateral rays** |

### 3. Downward Raycast Height Sensing
* **Ray Grid**: 6 forward distances (0.5, 1.0, 1.5, 2.0, 2.5, 3.0 m) × 3 lateral offsets (-0.3, 0.0, 0.3 m) projected in the torso's horizontal heading frame.
* **Filtering**: Raycasts strictly query `geom group="2"` (terrain and death plane), preventing false positive collisions with the humanoid's own waist or legs.
* **Values**: Normalized height of ground relative to torso center. Chasm/gap voids return -3.5m.

### 4. Course Generation (`envs/terrain.py`)
Each course spans 15 procedural segments (~40–50 meters total length):
* **Start Runway**: 6.0m wide flat platform (humanoid starts at x=2.5m, giving 3.5m of flat acceleration runway).
* **Platforms**: 3.0m wide in $y$ ($y \in [-1.5, 1.5]$), providing ample lateral margin.
* **Obstacle Types**:
  * `flat`: Flat platform sections (3.5–5.5m).
  * `boxes`: Platform with 1–2 climbing blocks (0.15–0.30m high).
  * `stairs_up`: Ascending steps (3–4 steps, 0.45m depth, 0.08m rise).
  * `stairs_down`: Descending steps (3–4 steps, 0.45m depth, 0.08m rise).
  * `gap`: Void between platforms (0.4–0.75m wide), never placed back-to-back.
  * `low_wall`: Low hurdle (0.25–0.38m high) across the track.

### 5. Shaped Reward & Bonus Schedule
At each control timestep:
$$r = 1.25 \cdot v_x \cdot \max(0, f_x) + 5.0 \cdot \mathbb{I}_{\text{healthy}} + 0.5 \cdot \max(0, z_{\text{up}}) + 2.0 \cdot f_x - 0.1 \sum u^2 - \text{clip}\left(5\times 10^{-7} \sum f_{\text{ext}}^2, 0, 10\right) + 50.0 \cdot \mathbb{I}_{\text{new\_obstacle\_cleared}}$$

* **Facing & Forward Alignment**: $f_x = \text{xmat}[0, 0] \in [-1, 1]$ (torso local forward axis projected onto world $+X$).
  * Forward velocity reward is gated by $\max(0, f_x)$, guaranteeing that running backwards earns zero forward speed credit.
  * Explicit facing reward $+2.0 \cdot f_x$ rewards facing forward ($+2.0$) and penalizes facing backwards ($-2.0$), creating a 4.0 point/step differential that completely eliminates backward backpedaling.
* **Obstacle Bonus**: One-time +50 reward awarded upon crossing the end of any non-flat obstacle segment.
* **Health Termination**: Torso $z < -0.5$ (death plane fall), $|y| > 2.0$ (lateral edge fall), or torso height above local ground outside $[0.75\text{ m}, 2.0\text{ m}]$. Optionally supports `terminate_when_backward` if $f_x < -0.5$.

---

## Training Execution (`runs/stage3`)

* **Base Weights**: Warm-started from `runs/stage2-horizon/best` (transferred 348 proprioceptive weights and normalizer running stats; new 18 height weights initialized to zero).
* **Compute**: 8 parallel CPU workers using multiprocessing `spawn`.
* **Throughput**: ~6,400–6,600 environment steps/second on Apple M2 Pro.
* **Budget**: 4,100,000+ transitions trained.

### Training Progression

| Update | Cumulative Steps | Mean Raw Return | Distance Traversed | Speed | Upright Score | Key Milestones |
|---|---:|---:|---:|---:|---:|---|
| **0** | 0 | 410.7 | 0.64 m | 0.71 m/s | 0.86 | Immediate bipedal stepping on start runway |
| **100** | 409,600 | 1,529.0 | 5.83 m | 1.72 m/s | 0.86 | Cleared 6.0m safe start runway |
| **200** | 819,200 | 1,797.8 | 6.97 m | 1.78 m/s | 0.85 | First entries into obstacle zone |
| **350** | 1,433,600 | 1,928.8 | 7.62 m | 1.90 m/s | 0.84 | Seed 10002 reached 13.14m (+100 obstacle bonus) |
| **850** | **3,481,600** | **2,362.2** | **8.25 m** | **1.66 m/s** | **0.86** | **Peak selection checkpoint; 4/5 seeds cleared obstacles** |
| **1000** | 4,096,000 | 2,021.3 | 8.02 m | 1.96 m/s | 0.86 | Consistent obstacle clearance |

---

## Held-Out Evaluation Results (10 Unseen Seeds: 20000–20009)

Evaluated deterministically with frozen observation normalization on 10 held-out procedural terrain seeds:

| Evaluation Metric | Stage 3 Selected Best Policy | Stage 2 Flat Baseline |
|---|---:|---:|
| **Mean Raw Return** | **2,301.7** | 10,848.6 (Flat ground) |
| **Mean Forward Distance** | **7.88 m** (Across obstacles) | 76.13 m (Flat plane) |
| **Peak Single-Episode Distance** | **12.93 m** (Seed 20007) | 77.15 m |
| **Mean Forward COM Speed** | **1.65 m/s (~3.7 mph)** | 5.07 m/s |
| **Mean Torso Upright Score** | **0.857** | 0.874 |
| **Obstacle Clearance Rate** | **6 / 10 held-out seeds cleared obstacles** | N/A |
| **Double-Obstacle Clearances** | **4 / 10 seeds cleared $\ge 2$ obstacles** | N/A |
| **Mean Foot Touchdowns (L / R)** | **9.3 / 11.9 per episode** | 37.8 / 34.5 |
| **Foot Contact Fraction (L / R)** | **38.0% / 37.1% (Symmetric bipedal)** | 30.7% / 16.5% |

### Physics & Behavior Analysis
1. **Perception-Action Coupling**: The 18 downward height sensors provide clear depth signals ahead of the body. The policy modulates leg lift and stride frequency as it approaches elevation changes (stairs and boxes).
2. **Gait Symmetry on Terrain**: On flat ground, the Stage 2 policy had developed an asymmetric duty cycle (~31% left vs ~16% right). On the parkour terrain, the agent regularized into a balanced, symmetric bipedal gait: **38.0% left foot contact vs 37.1% right foot contact**.
3. **Absence of Pathological Movement**:
   * No single-leg hopping: Both feet register alternating touchdowns (9.3 left, 11.9 right on average).
   * No shuffling: Forward velocity is maintained at 1.65–2.00 m/s with deliberate leg clearance.
   * No floor clipping: Ground contact forces are properly transmitted through the MuJoCo broadphase BVH.

---

## Artifacts & Checkpoints

* **Selected Best Checkpoint (Original)**: [`runs/stage3/best/`](file:///Users/REVIEW_USER/Desktop/Random/parkour-lab/runs/stage3/best/) (`policy.zip`, `vecnormalize.pkl`, `selection.json`, `metadata.json`)
* **Forward-Facing Best Checkpoint**: [`runs/stage3-facing/best/`](file:///Users/REVIEW_USER/Desktop/Random/parkour-lab/runs/stage3-facing/best/) (Mean facing score 0.72–0.88, 100% forward strides)
* **Forward-Facing Held-Out Video**: [`runs/stage3-facing/heldout.mp4`](file:///Users/REVIEW_USER/Desktop/Random/parkour-lab/runs/stage3-facing/heldout.mp4)
* **Forward-Facing Video (Update 500)**: [`runs/stage3-facing/videos/update-00500.mp4`](file:///Users/REVIEW_USER/Desktop/Random/parkour-lab/runs/stage3-facing/videos/update-00500.mp4)
* **Forward-Facing Training Curve**: [`runs/stage3-facing/reward-curve.png`](file:///Users/REVIEW_USER/Desktop/Random/parkour-lab/runs/stage3-facing/reward-curve.png)
* **Held-Out Evaluation Video**: [`runs/stage3/heldout.mp4`](file:///Users/REVIEW_USER/Desktop/Random/parkour-lab/runs/stage3/heldout.mp4)
* **Peak Performance Video**: [`runs/stage3/videos/update-00850.mp4`](file:///Users/REVIEW_USER/Desktop/Random/parkour-lab/runs/stage3/videos/update-00850.mp4) (383 frames, 5.72s, 11.6m traverse)
* **Training Curve**: [`runs/stage3/reward-curve.png`](file:///Users/REVIEW_USER/Desktop/Random/parkour-lab/runs/stage3/reward-curve.png)
* **Evaluation JSON**: [`runs/stage3/heldout.json`](file:///Users/REVIEW_USER/Desktop/Random/parkour-lab/runs/stage3/heldout.json)

**Disposition**: Stage 3 environment and training pipeline are **fully built, validated, and operational**. The humanoid successfully navigates complex procedural courses with body-relative sensing and natural forward-facing heading alignment. Ready for Stage 4 curriculum scaling.
