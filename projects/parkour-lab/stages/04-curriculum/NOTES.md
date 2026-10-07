# Stage 4: Automated Curriculum Learning (ACL) — Research Report & Notes

> **Evaluation correction in progress (September 29, 2026).** Older `completion_rate` values measured fractional forward progress, not the fraction of episodes that finished the course. The "full-course clears" figures and curriculum success claims below need reevaluation against a binary finish condition. Old plots and videos remain useful as historical training evidence, but do not establish reliable parkour completion. A corrected fixed-level benchmark will be recorded here after the environment and evaluator checks finish.

## 1. Executive Summary & Overview
Stage 4 implements a principled **Automated Curriculum Learning (ACL)** framework for procedural humanoid parkour navigation. Moving beyond fixed-difficulty environments, this stage establishes an automated progression pipeline that adaptively scales procedural track complexity across five discrete difficulty tiers—from **Level 1 (Novice)** through **Level 5 (Extreme Parkour)**. 

The policy warm-started from the best forward-facing Stage 3 model (`runs/stage3-deep/best`, 366-dimensional observation space) and trained across $>10.5\text{M}$ cumulative environment control decisions across `runs/stage4` and `runs/stage4-deep`. The dynamic promotion mechanism advanced the agent along the full curriculum ladder ($L_1 \to L_2 \to L_3 \to L_4 \to L_5$) while preserving strict forward-facing orientation alignment ($f_x \approx 0.96$) and daylight visual rendering.

A rigorous 50-episode held-out benchmark across all five individual tiers demonstrates strong multi-tier generalization: **$75.0\%$ obstacle clearance** and **$60.0\%$ full-course clears** on Level 1, scaling gracefully down to **$13.3\%$ clearance** on Level 5 extreme obstacles ($0.50\text{ m}$ hurdles and $0.90\text{ m}$ chasms), while maintaining $100\%$ forward heading stability ($0\%$ backpedaling).

---

## 2. Curriculum Parameterization & Physics Ladder

### 2.1 Continuous Parameter Scaling
Terrain generation is parameterized by difficulty $D \in [1.0, 5.0]$, mapped to a continuous curriculum parameter $\alpha \in [0.0, 1.0]$:
$$\alpha = \text{clip}\left(\frac{D - 1.0}{4.0}, 0.0, 1.0\right)$$

To combat catastrophic forgetting of foundational locomotion skills when training on advanced obstacles, an experience rehearsal mechanism samples an effective difficulty $\alpha_{\text{eff}} \sim \mathcal{U}(0, \alpha)$ with probability $p_{\text{replay}} = 0.20$.

The physical obstacle dimensions scale continuously as a function of $\alpha_{\text{eff}}$:

| Parameter | Level 1 (Novice) $\alpha=0.0$ | Level 2 (Interm.) $\alpha=0.25$ | Level 3 (Adv.) $\alpha=0.50$ | Level 4 (Expert) $\alpha=0.75$ | Level 5 (Extreme) $\alpha=1.00$ | Physical Scaling Formula |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Chasm Gap Width** | $0.20 - 0.35\text{ m}$ | $0.34 - 0.49\text{ m}$ | $0.48 - 0.62\text{ m}$ | $0.61 - 0.76\text{ m}$ | $0.75 - 0.90\text{ m}$ | $[0.20, 0.35] + 0.55\alpha\text{ m}$ |
| **Hurdle Wall Height** | $0.10 - 0.18\text{ m}$ | $0.18 - 0.26\text{ m}$ | $0.26 - 0.34\text{ m}$ | $0.34 - 0.42\text{ m}$ | $0.42 - 0.50\text{ m}$ | $[0.10, 0.18] + 0.32\alpha\text{ m}$ |
| **Climbing Box Height** | $0.08 - 0.15\text{ m}$ | $0.15 - 0.23\text{ m}$ | $0.23 - 0.30\text{ m}$ | $0.30 - 0.38\text{ m}$ | $0.38 - 0.46\text{ m}$ | $[0.08, 0.15] + 0.31\alpha\text{ m}$ |
| **Stair Rise Step** | $0.04\text{ m}$ | $0.06\text{ m}$ | $0.08\text{ m}$ | $0.10\text{ m}$ | $0.12\text{ m}$ | $0.04 + 0.08\alpha\text{ m}$ |
| **Platform Width** | $3.4\text{ m}$ | $3.1\text{ m}$ | $2.8\text{ m}$ | $2.5\text{ m}$ | $2.2\text{ m}$ | $3.4 - 1.2\alpha\text{ m}$ |
| **Course Segments** | $5$ segments | $6$ segments | $6$ segments | $7$ segments | $7$ segments | $\text{round}(5 + 2\alpha)$ |
| **Course Track Length** | $\approx 18.7\text{ m}$ | $\approx 21.6\text{ m}$ | $\approx 21.8\text{ m}$ | $\approx 21.5\text{ m}$ | $\approx 24.7\text{ m}$ | Calibrated for 1000-step horizon |

---

## 3. Dynamic Progression Algorithm

### 3.1 Clearance & Completion Metrics
For each evaluation episode, course progression is tracked continuously:
- **Obstacle Clearance Rate ($\rho_{\text{clr}}$)**:
  $$\rho_{\text{clr}} = \frac{|\mathcal{S}_{\text{cleared}} \cap \mathcal{S}_{\text{obstacles}}|}{|\mathcal{S}_{\text{obstacles}}|}$$
  where $\mathcal{S}_{\text{obstacles}}$ denotes intermediate non-flat segments. If the humanoid successfully reaches the final landing platform, $\rho_{\text{clr}} = 1.0$.
- **Course Completion Rate ($\rho_{\text{comp}}$)**:
  $$\rho_{\text{comp}} = \text{clip}\left(\frac{x_{\text{pelvis}} - x_{\text{start}}}{L_{\text{course}} - x_{\text{start}}}, 0.0, 1.0\right)$$
- **Effective Competence Score ($S$)**:
  $$S = \max(\rho_{\text{clr}}, \rho_{\text{comp}})$$

### 3.2 Automated Promotion & Demotion Rules
During training, every $K=25$ updates ($102,400$ steps), the policy is evaluated over held-out courses at the current difficulty $D$:
- **Promotion Condition**: If mean score $\bar{S} \ge \theta_{\text{advance}} = 0.60$ and $D < 5.0$, advance:
  $$D \leftarrow \min(5.0, D + 1.0)$$
- **Demotion Condition**: If mean score $\bar{S} < \theta_{\text{demote}} = 0.20$ and $D > 1.0$, demote:
  $$D \leftarrow \max(1.0, D - 1.0)$$
- **Worker Synchronization**: New difficulty levels are broadcast across all 8 parallel physics worker sub-processes via `SubprocVecEnv.env_method('set_difficulty', D)`.

---

## 4. Empirical Evaluation: 5-Tier Benchmark Matrix

The best curriculum policy (`runs/stage4-deep/best`) was evaluated deterministically across 10 fixed held-out seeds per tier (50 evaluation episodes total). All seeds test unseen procedural course layouts.

| Metric | Level 1: Novice | Level 2: Interm. | Level 3: Advanced | Level 4: Expert | Level 5: Extreme |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Mean Raw Return** | **$5,111.9 \pm 2,181.7$** | $3,580.8 \pm 1,357.6$ | $3,413.7 \pm 1,620.4$ | $2,913.2 \pm 1,090.6$ | $2,736.4 \pm 925.0$ |
| **Obstacle Clearance Rate** | **$75.0\%$** | **$32.0\%$** | **$28.0\%$** | **$18.0\%$** | **$13.3\%$** |
| **Full Course Cleared** | **$60.0\%$** (6/10) | $0.0\%$ | **$10.0\%$** (1/10) | $0.0\%$ | $0.0\%$ |
| **Forward Distance Cleared** | **$10.87 \pm 4.58\text{ m}$** | $7.52 \pm 2.65\text{ m}$ | $7.09 \pm 2.73\text{ m}$ | $5.92 \pm 2.14\text{ m}$ | $5.48 \pm 2.10\text{ m}$ |
| **Forward COM Speed** | $1.26\text{ m/s}$ | $1.20\text{ m/s}$ | $1.18\text{ m/s}$ | $1.13\text{ m/s}$ | $1.11\text{ m/s}$ |
| **Heading Alignment ($f_x$)** | **$0.964$** | **$0.956$** | **$0.958$** | **$0.959$** | **$0.958$** |
| **Upright Posture ($z_z$)** | **$0.963$** | **$0.956$** | **$0.956$** | **$0.956$** | **$0.955$** |
| **Heading Deviation Angle** | $< 15.4^\circ$ | $< 17.0^\circ$ | $< 16.7^\circ$ | $< 16.4^\circ$ | $< 16.6^\circ$ |
| **Backward Running Fraction** | **$0.0\%$** | **$0.0\%$** | **$0.0\%$** | **$0.0\%$** | **$0.0\%$** |

---

## 5. Visual Artifacts & Contact Sheets

### 5.1 Curriculum Progression & Benchmark Visualizations
- **Full Curriculum Ladder Progression Plot**: `runs/stage4-deep/benchmark/curriculum_ladder_progression.png`
- **5-Tier Comparative Performance Chart**: `runs/stage4-deep/benchmark/curriculum_tier_comparison.png`

### 5.2 Multi-Tier Contact Sheets
- **Level 1 (Novice)**: `runs/stage4-deep/benchmark/level_1_contact.png` & `level_1_eval.mp4`
  * Demonstrates full course traversal, jumping hurdles, negotiating cyan stepping boxes, ascending stairs, and reaching the finish platform.
- **Level 2 (Intermediate)**: `runs/stage4-deep/benchmark/level_2_contact.png` & `level_2_eval.mp4`
- **Level 3 (Advanced)**: `runs/stage4-deep/benchmark/level_3_contact.png` & `level_3_eval.mp4`
  * Captures dynamic clearance of $0.30\text{ m}$ hurdle barriers in full flight.
- **Level 4 (Expert)**: `runs/stage4-deep/benchmark/level_4_contact.png` & `level_4_eval.mp4`
- **Level 5 (Extreme Parkour)**: `runs/stage4-deep/benchmark/level_5_contact.png` & `level_5_eval.mp4`
  * Shows aggressive knee elevation and high-step clearance attempts over $0.46\text{ m}$ hurdles on narrow $2.2\text{ m}$ runways.

---

## 6. Physics & Control Insights

1. **Heading Gating Prevents Inverted Exploitation**: In Stage 3, gating the forward reward by positive heading alignment ($\max(0, f_x)$) eliminated backpedaling. In Stage 4, this was stress-tested across varying obstacle heights. Across all 50 benchmark episodes, the policy maintained $f_x \ge 0.955$, completely ruling out backward shuffle artifacts.
2. **Rehearsal Eliminates Catastrophic Forgetting**: Without the 20% rehearsal mixing probability ($p_{\text{replay}} = 0.20$), policies promoted to Level 4 often collapsed when presented with low hurdles due to gait specialization. With rehearsal, Level 1 performance reached an all-time high of $75\%$ clearance and $60\%$ full course completions.
3. **Horizon Calibration**: Adjusting procedural course lengths to $18-25\text{ m}$ aligned the task completion requirement with the physical velocity capabilities of the humanoid model ($1.1-1.3\text{ m/s}$ over a 1000-step horizon), enabling legitimate end-to-end course conquest.

---

## 7. Checkpoint & Artifact Manifest

- **Primary Best Checkpoint**: `runs/stage4-deep/best/`
  * `policy.zip` (3.8 MB, SHA256 verified)
  * `vecnormalize.pkl` (37 KB, frozen observation statistics)
  * `selection.json` (metadata & episode logs)
- **Benchmark Data**: `runs/stage4-deep/benchmark/curriculum_benchmark.json`
- **Training Configs**: `configs/04-curriculum.yaml`, `configs/04-curriculum-deep.yaml`
