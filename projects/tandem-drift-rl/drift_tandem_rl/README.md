# Multi-Agent Tandem Drift Battles (Leader-Chaser Self-Play) 🚙💨

An advanced reinforcement learning and vehicle dynamics environment extending **`drift_rl`**. Two autonomous agents engage in high-speed gymkhana tandem battles using **Multi-Agent PPO (MAPPO)** with **Centralized Training and Decentralized Execution (CTDE)**. The chase car must shadow the leader door-to-door while braving aerodynamic dirty air, wake turbulence buffeting, and blinding tire smoke.

---

## 🌪️ Key Dynamics & Aerodynamic Phenomena

### 1. Leader Wake Turbulence & Dirty Air Downforce Loss
As the leader drifts at high slip angles $\beta_L$, an aerodynamic wake cone sheds along the opposite direction of the leader's ground velocity vector:
- **Wake Cone Expansion**: $w(d_\parallel) = w_0 + 2 d_\parallel \tan(\theta_{spread})$
- **Dynamic Pressure Deficit**: Up to $55\%$ reduction in dynamic pressure along the centerline:
  $$q_{loss} = q_{max} \cdot \left(1 - \frac{d_\perp}{w}\right)^2 \cdot \exp\left(-\frac{d_\parallel}{L_{decay}}\right)$$
- **Front Axle Downforce Collapse & Washout**: Front tire aero normal load drops by up to $60\%$, inducing high-speed understeer when tucked behind the leader's rear bumper.
- **Vortex-Shedding Buffet Forces**: Stochastic lateral buffeting forces ($F_{buffet,y} \le 380\text{ N}$) and yaw moments ($\tau_{buffet,z} \le 450\text{ Nm}$) simulate turbulent shedding off the leader's rear spoiler and diffuser.

### 2. Molten Rubber & Tire Smoke Degradation
- Severe drifting ($|\beta| > 14^\circ$) emits volumetric tire smoke plumes.
- Follower driving through smoke clouds experiences a local road surface friction reduction ($\mu_{eff} = \mu_{base} \cdot (1 - \Delta \mu_{smoke})$) of up to $30\%$, replicating real-world loss of grip on marbles and molten rubber.

### 3. Separating Axis Theorem (SAT) Rigid-Body Collisions
- Full 2D oriented bounding box collision test.
- Elastic momentum exchange separates door-to-door rubs from catastrophic at-fault collisions.

---

## 🏆 Formula Drift & D1GP Tandem Judging System

| Criterion | Target | Scoring Impact |
|---|---|---|
| **Door-to-Door Proximity** | $0.8\text{ m} \le d_{clearance} \le 2.8\text{ m}$ | High reward in Sweet Spot; bonus for kissing doors ($< 0.8\text{ m}$) |
| **Angle Matching (Mimicry)** | $\cos(\beta_C - \beta_L) \to 1.0$ | Rewards synchronized slip angle and parallel slide |
| **Wake Survival** | High $\beta$ inside wake cone | Multiplier bonus for sustaining drift in dirty air |
| **Clipping Points (Lead)** | Inside/Outside clipping zones | $+1200 - 2500$ points per apex clip |
| **At-Fault Crash** | Penetration $> 0.35\text{ m}$ / T-bone | Instant run forfeit ($-40$ pts, zero score) |

---

## 🧠 MAPPO Architecture (CTDE)

- **Decentralized Actors**:
  - $\pi_{\theta_{lead}}(a_L \mid o_L)$ and $\pi_{\theta_{chase}}(a_C \mid o_C)$
  - Condition strictly on 45-dimensional ego observations (no cheating / god-mode in deployment).
- **Centralized Critics**:
  - $V_{\phi_{lead}}(s_{global})$ and $V_{\phi_{chase}}(s_{global})$
  - Condition on 90-dimensional joint state during training, eliminating multi-agent non-stationarity and stabilizing Generalized Advantage Estimation ($\text{GAE}-\lambda$).

---

## 🚀 Quickstart & Usage

### 1. Interactive 60 FPS Visualizer
```bash
# Watch autonomous AI vs AI tandem battle
python3 -m drift_tandem_rl.visualizer_tandem --mode AI_VS_AI

# Take the wheel as the Chaser against the AI Leader
python3 -m drift_tandem_rl.visualizer_tandem --mode HUMAN_CHASE

# Drive as the Leader trying to gap the AI Chaser
python3 -m drift_tandem_rl.visualizer_tandem --mode HUMAN_LEAD
```

**Controls**:
- `Arrow Keys`: Steer (Left/Right), Throttle (Up), Footbrake (Down)
- `Spacebar`: Handbrake flick
- `M`: Cycle modes (AI vs AI / Human Chase / Human Lead)
- `R`: Reset heat

### 2. Multi-Agent MAPPO Training
```bash
python3 -m drift_tandem_rl.train_tandem --steps 50000 --rollout 512 --track touge
```

### 3. Tournament Evaluation & Telemetry Plotting
```bash
python3 -m drift_tandem_rl.evaluate_tandem --heats 5 --chart tandem_eval_telemetry.png
```
