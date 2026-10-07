> Historical roadmap — superseded by README.md and docs/RESEARCH_PROTOCOL.md. Checked items and numerical claims below describe the original prototype, not validated research conclusions.

# 🔬 Machine Learning Ideas & To-Do Roadmap

This archive tracks the shortlisted physics & biology machine learning concepts, with full documentation preserved in the workspace.

## 🎯 Active Project: Norman Game Day Autonomous Traffic Optimizer ("SoonerGrid")
- [x] **Step 1: Real Network Ingestion & Topological Graph Engine**
  - Ingested 18,813 OSM elements (11,578 nodes, 24,003 directed edges, 991 km centerline).
  - BPR link capacities, LWR jam storage (176k vehicles), 990 contraflow twin pairs.
  - Interactive road visualizer: `visualizer/network_viewer.html`.
- [x] **Step 2: Dynamic Game Day OD Matrix & Surge Demand Engine**
  - Non-homogeneous Poisson arrival curves across 7-hour game-day arc.
  - Multinomial Logit (MNL) discrete choice parking selection with dynamic spillover.
  - Lloyd Noble Center 35-bus shuttle fleet (56,400 passengers / 22,560 car trips saved).
  - Pedestrian crosswalk conflict modeling (82% capacity derating on Lindsey/Jenkins).
  - Interactive surge dashboard: `visualizer/demand_viewer.html`.
- [x] **Step 3: Baseline Simulation Engine (Demonstrating Lindsey & I-35 Gridlock)**
  - Vectorized Daganzo Link Transmission Model (LTM / CTM) over 5,760 timesteps (8 hours).
  - Fixed-time pre-timed signal controllers at 7 key Norman intersections.
  - Measured 269,833 veh-hrs of unmanaged delay (91.3% delay ratio, Lindsey speeds dropping to 4.2 mph).
  - Interactive playback dashboard: `visualizer/simulation_viewer.html`.
- [x] **Step 4: Autonomous Map Resituation & Signal Multi-Agent RL (MARL)**
  - Dynamic tidal contraflow manager (`soonergrid/policy/contraflow_manager.py`) with 10-minute safety buffer (3:1 ingress $\to$ 0:4 outbound flush).
  - Communicating MARL signal coordination (`soonergrid/policy/signal_marl_agent.py`) maintaining 30 mph green wave progression.
  - Perimeter gateway inflow diversion (40% Moore/OKC $\to$ 12th Ave bypass, 50% Noble $\to$ SH-9/LNC) and pulsed pedestrian scramble gating (`soonergrid/policy/perimeter_router.py`).
  - Master hierarchical coordinator (`soonergrid/policy/autonomous_coordinator.py`) unified into LTM engine.
  - Research-Grade Counterfactual Scorecard: -48.3% TSTT, -58.6% delay hours, Lindsey speeds restored from 4.2 mph to 28.4 mph, I-35 spillback eliminated (0 min), 94,871 gallons fuel saved.
  - 40/40 unit tests passing (`python3 -m unittest discover -s tests/ -v`).
  - Interactive Research-Grade Command Center: `visualizer/autonomous_viewer.html` & `visualizer/research_dashboard.html` (dual synchronized split canvas, curtain swipe, live speed profile, policy ablation matrix).
- [x] **Step 5: Microscopic Human Factors, Mixed-Autonomy CAV Platooning & Incident Resilience Stress-Testing**
  - Microscopic human factors & compliance scaling (`soonergrid/policy/human_factors.py`): stochastic driver compliance $\beta \in [0, 1]$, rubbernecking proximity decay, pedestrian jaywalking leakage.
  - Connected & Autonomous Vehicle (CAV) fleet penetration modeling (`soonergrid/physics/cav_platooning.py`): Cooperative Adaptive Cruise Control (CACC) micro-headway scaling ($h_{\text{CAV}}=0.60\text{s}$ vs $h_{\text{HDV}}=1.50\text{s}$), nonlinear capacity expansion $M_C(p) = 1.0 \to 2.5$, jam storage expansion.
  - Real-time dynamic incident generator & crisis stress-testing (`soonergrid/sim/incident_manager.py`): arterial collision at Lindsey & Berry (-67% capacity), severe Oklahoma thunderstorm squall (-40% capacity, -35% speed, 8k vph evacuation), and double overtime egress shift (+45 min).
  - Closed-loop adaptive self-healing coordinator (`soonergrid/policy/self_healing_agent.py`): shockwave anomaly detection, upstream metering, downstream bottleneck flushing, and Network Resilience Index ($R_{\text{net}}$).
  - Resilient Vectorized Simulation Engine (`soonergrid/sim/resilient_traffic_engine.py`): integrated multi-class flow propagation, dynamic incident overrides, and sink/gateway absorption lifecycle.
  - 54/54 unit tests passing cleanly in 1.23s (`tests/test_human_factors.py`, `tests/test_cav_platooning.py`, `tests/test_incident_shocks.py`, `tests/test_self_healing_resilience.py`, `tests/test_resilient_engine.py`).
  - Comprehensive Benchmark Dataset (`data/norman_resilience_benchmark_results.json`): 12 dynamic shock evaluations and 25-cell sensitivity matrix (CAV 0-100% $\times$ Compliance 20-100%).
  - Standalone Research Command Center (`visualizer/resilience_viewer.html`): live interactive incident injection buttons, CAV/compliance sliders, dynamic shockwave heatmap, 5-axis resilience radar chart, and 5x5 sensitivity matrix heatmap.

---

## 📌 Saved Shortlist
- **#1. Active Aerodynamic Ground-Effect & Venturi Stalling RL** (`drift_rl` extension: downforce, porpoising, ride-height control).
- **#2. Multi-Agent Tandem Drift Battles (Leader-Chaser Self-Play)** (`drift_rl` extension: formula drift, trailing wake, close proximity).
- **#9. Hill-Type Muscle-Tendon Actuation for Humanoids** (`humanoid_parkour` extension: biological energy efficiency, compliant gait, MuJoCo).
- **#12. Insect Flapping Flight & Leading-Edge Vortex (LEV) Aerodynamics** (unsteady aerodynamics, dynamic stall, high lift).

*(Full 50-idea catalog saved in system artifacts)*
