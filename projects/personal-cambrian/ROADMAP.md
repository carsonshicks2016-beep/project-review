# PERSONAL CAMBRIAN — ROADMAP.md (detailed work breakdown)

The granular, actionable execution plan. **[PLAN.md](PLAN.md) remains
authoritative for *design rationale*;** this file is the *how*. Every task uses:

- **Do** — the concrete thing to build.
- **Method** — how: library, file/module, function shapes, algorithm.
- **In→Out** — inputs and produced artifacts.
- **Done when** — the check that proves it works (a test, a metric, a render).

Status legend: ✅ done · ⏳ todo · 🔬 research-grade (acceptance = "measurable",
not "solved"). Effort: **S** ≤1 day · **M** 2–4 d · **L** 1–2 wk · **XL** 3 wk+
(solo, rough). **CP** = on the critical path.

Stage map (see PLAN §15): **F** Foundations → **0** Scaffold (done) → **1**
Encoding+Morphogenesis (CP) → **2** Physics niche (CP) → **3** Control (CP) →
**4** Macro-mutation+innovation → **5** Quality-diversity → **6** Budgets/
viability → **7** Multi-niche+speciation → **8** Open-endedness 🔬 → **9** GPU
scale+surrogates → **10** Blender viz → **11** Reality anchor (optional).

---

## Stage F — Foundations & cross-cutting (do alongside Stage 1)

**Goal:** reproducible, inspectable experiment infrastructure so every later
stage is seedable, logged, and resumable. **Depends on:** nothing.

### F.1 Environment & dependency pinning — S, CP
- **Do:** lock a Python env for the heavy stack (mujoco, torch, jax, pyribs).
- **Method:** add `pyproject.toml` (or keep `requirements.txt`) with pinned
  versions; create a `conda`/`venv` env file; document GPU vs CPU installs
  (OpenSim via conda-forge, JAX with CUDA wheels). Keep Stage-0 NumPy-only path
  intact so the scaffold always runs.
- **In→Out:** none → reproducible env, `make install` / documented commands.
- **Done when:** fresh env installs and `python3 tests/test_validation.py` passes.

### F.2 Determinism & seeding utilities — S, CP
- **Do:** one place that seeds Python/NumPy/Torch/JAX and threads a seed through
  genome→morphogenesis→sim→RL.
- **Method:** `personal_cambrian/util/seeding.py` with `set_global_seed(seed)`
  and a `Rng` wrapper passed explicitly (no hidden global RNG in evolution code).
- **In→Out:** seed int → seeded generators.
- **Done when:** two runs with the same seed produce identical genomes, archives,
  and (CPU) rollouts; asserted in a test.

### F.3 Config system — S
- **Do:** declarative experiment configs (no magic constants in code).
- **Method:** Hydra or plain YAML + dataclass configs in `configs/`; one config
  per experiment (encoding params, niche, RL budget, QD descriptors, seeds).
- **In→Out:** `configs/*.yaml` → typed config object.
- **Done when:** an experiment is fully specified by a config file + seed.

### F.4 Experiment logging & artifacts — S
- **Do:** capture metrics, checkpoints, and provenance per run.
- **Method:** Weights & Biases *or* a local `runs/<timestamp>/` with JSONL
  metrics + saved configs + git SHA; `util/logging.py`.
- **In→Out:** run events → durable logs/plots.
- **Done when:** a finished run reproduces its summary plots from saved logs.

### F.5 Persistence store — M
- **Do:** durable store for seeds, genomes, evals, lineage, innovations.
- **Method:** SQLite via `sqlite3`/SQLModel; tables `seed_metrics, genomes
  (json blob + hash), morphologies, evals, lineage(parent,child,mutation_json,
  innovation_flags), niche_tasks, experiments`. Genome stored as canonical JSON +
  content hash for dedup.
- **In→Out:** runtime objects → queryable DB; DB → resume/branch a run.
- **Done when:** a run can be stopped and resumed from the DB; lineage query
  returns a creature's full ancestry.

### F.6 CI & test runner — S
- **Do:** run the test suite automatically.
- **Method:** `pytest` config + a GitHub Actions (or local `make test`) workflow;
  fast tests on every change, slow sim/RL tests behind a `@pytest.mark.slow`.
- **In→Out:** commit → pass/fail signal.
- **Done when:** CI is green on the scaffold and stays gating thereafter.

**Stage F DoD:** any experiment = `config + seed`, runs logged to `runs/` + DB,
reproducible, CI green.

---

## Stage 0 — Scaffold (✅ done) + hardening

**Goal:** keep the NumPy scaffold as a regression baseline and a plug-in surface.

### 0.1 Freeze a regression baseline — S
- **Do:** snapshot current HA/OE archive metrics as a golden file.
- **Method:** script writes coverage/QD-score/best-fitness for fixed seed to
  `tests/golden/stage0_ha.json`; a test asserts future runs match within tol.
- **Done when:** a test fails if scaffold behavior drifts unintentionally.

### 0.2 Namespace the legacy flat genome — S
- **Do:** mark the 14-gene vector as scaffold, not the real encoding.
- **Method:** move `genome.py`/`phenotype.py`/`biomes/` under
  `personal_cambrian/scaffold/` (or leave + docstring "Stage-0; superseded by
  `encoding/`"); update imports in `scripts/`.
- **Done when:** scaffold still runs; new `encoding/` is the obvious home for the
  generative genome.

**Stage 0 DoD:** scaffold runnable, baselined, clearly labeled as scaffold.

---

## Stage 1 — Generative encoding + morphogenesis → MJCF  **(THE CORE, CP)**

**Goal:** a genome of *growth rules* that compiles to a simulatable creature, so
evolution can invent structure, not just resize a human. **Depends on:** F.1–F.2.
**Effort:** XL.

### 1.1 Genome data structures (recursive body graph) — L, CP
- **Do:** define the Sims-style graph genome.
- **Method:** `encoding/genome.py` with dataclasses:
  - `PartNode`: `shape∈{capsule,box,sphere,ellipsoid}`, dimension params,
    `density`, list of `AttachmentSite` (local pos+quat), `recursion_limit`.
  - `ConnectionEdge`: `parent_part`, `child_part`, `site_idx`, child
    `pos/quat/scale`, `Joint(type∈{hinge,ball,slide,fixed}, axis, range,
    stiffness, damping)`, `recursion_count`, `terminal_only`, `symmetry∈{none,
    bilateral,radial(n)}`.
  - `MuscleGene`: `origin(part,site)`, `insertion(part,site)`, `route_sites[]`,
    Hill params (`Fmax` from PCSA, `optimal_fiber_len`, `tendon_slack_len`,
    `pennation`, `vmax`, `fiber_type∈[0,1]`), `activation_cost`.
  - `Genome`: `root: PartNode`, `parts[]`, `edges[]` (directed, may be cyclic for
    recursion), `muscles[]`, global `max_depth`, `symmetry_plane`.
- **In→Out:** none → typed genome object.
- **Done when:** can construct a genome in code; `Genome.validate_shape()` rejects
  malformed graphs (dangling sites, unknown parts).

### 1.2 Canonical (de)serialization + hashing — S, CP
- **Do:** stable JSON form + content hash for the DB and determinism.
- **Method:** `genome.to_json()/from_json()` with sorted keys; `genome.hash()` =
  sha256 of canonical JSON.
- **Done when:** round-trip `from_json(to_json(g)) == g`; equal genomes share a
  hash; test asserts both.

### 1.3 Morphogenesis traversal — L, CP
- **Do:** expand the graph into a concrete body (bodies, joints, sites).
- **Method:** `morphogenesis/develop.py::develop(genome) -> Morphology`.
  Depth-first from root; maintain per-edge recursion counters and a global
  `max_depth` cap to terminate cycles; compute each child's world transform from
  parent site × edge pos/quat/scale; collect `Body, Joint, Site, MuscleRoute`
  objects into a `Morphology` (engine-neutral, not yet MJCF).
- **In→Out:** `Genome` → `Morphology` (lists of bodies/joints/sites/muscles).
- **Done when:** a 3-segment chain genome with `recursion_count=3` yields the
  correct body count and transforms (unit test on known geometry).

### 1.4 Symmetry & limb repetition — M, CP
- **Do:** support bilateral/radial duplication so paired limbs emerge from one
  rule.
- **Method:** in `develop`, when an edge has `symmetry=bilateral`, instantiate the
  child subtree and its mirror across `symmetry_plane`; `radial(n)` rotates n
  copies about an axis. Mirror = negate one axis in transforms + flip joint axes.
- **Done when:** one "leg" rule produces a symmetric pair; a radial rule produces
  n evenly-spaced limbs; geometry verified in a test.

### 1.5 Compile Morphology → MuJoCo model (MJCF) — L, CP
- **Do:** turn a `Morphology` into a loadable physics model.
- **Method:** `morphogenesis/to_mujoco.py` using **`mujoco.MjSpec`** (procedural
  model API, mujoco ≥3.2) — preferred over hand-writing XML; alt: `dm_control`
  PyMJCF. Map each `Body`→`spec.worldbody.add_body`, geoms, `Joint`→`add_joint`,
  `Site`→`add_site`; set inertials from geom density; `spec.compile()` → `mjModel`.
- **In→Out:** `Morphology` → `mjModel` (+ exported `.xml` for inspection).
- **Done when:** `spec.compile()` succeeds, `mj_forward` runs with no NaN, total
  mass > 0, no resting interpenetration beyond tolerance.

### 1.6 Muscles → spatial tendons + actuators — L
- **Do:** attach actuation.
- **Method:** for each `MuscleGene`, add a `spatial` tendon routed through
  origin/route/insertion sites; add an actuator. **v1:** simple `motor`/`position`
  actuators per joint (robust). **v2:** MuJoCo `muscle` actuator with
  `gainprm/biasprm` from Hill params (force-length-velocity). Compute moment arms
  via `mj_tendon` Jacobian to confirm each muscle actually torques its joint.
- **Done when:** actuating a muscle moves the intended joint; computed moment arm
  about the target joint is non-zero (test).

### 1.7 Collision filtering & contact groups — M
- **Do:** stop adjacent connected bodies from self-colliding spuriously.
- **Method:** set `contype/conaffinity` so parent–child pairs don't collide;
  enable collision between non-adjacent parts (for self-intersection detection in
  Stage 6) and with the floor.
- **Done when:** a jointed limb flexes through its full range without phantom
  self-contact; floor contact works.

### 1.8 Hand-authored seed genomes — M, CP
- **Do:** author Agent Zero (human-ish biped) **and** one deliberately non-human
  variant (e.g., biped + tail, or quadruped) by hand in the new encoding.
- **Method:** `data/seeds/agent_zero.genome.json`,
  `data/seeds/tailed.genome.json` constructed via the API in 1.1.
- **Done when:** both compile (1.5), load, and stand/settle under gravity without
  exploding.

### 1.9 Determinism & compile test suite — M, CP
- **Do:** lock correctness of the encoding pipeline.
- **Method:** `tests/test_encoding.py`: serialization round-trip; same genome →
  identical MJCF hash; recursion/symmetry geometry; compile success for both
  seeds; NaN/mass/penetration sanity.
- **Done when:** all pass in CI.

### 1.10 Viewer smoke render — S
- **Do:** eyeball a compiled creature.
- **Method:** `scripts/view_creature.py` using `mujoco.viewer` (interactive) or
  `mujoco.Renderer` → PNG for headless.
- **Done when:** both seeds render to image/video.

**Stage 1 DoD:** an arbitrary valid genome deterministically grows a creature that
loads in MuJoCo, has working actuated muscles, and renders — including at least
one **non-human** body plan.

**Stage 1 risks:** cyclic-graph non-termination (→ hard `max_depth`); MjSpec API
churn (→ pin mujoco, wrap the API); degenerate inertias from tiny geoms (→ min
size + density clamps).

---

## Stage 2 — Physics evaluation of one niche (locomotion)  **(CP)**

**Goal:** score a creature by making it *physically perform* a task. **Depends
on:** Stage 1. **Effort:** M.

### 2.1 Env wrapper (Gymnasium API) — M, CP
- **Do:** wrap a compiled creature + task as a standard RL env.
- **Method:** `sim/env.py::CreatureEnv(gymnasium.Env)`; holds `mjModel/mjData`;
  `reset()` randomizes initial pose slightly; `step(action)` applies controls,
  runs `n_substeps` of `mj_step`, returns `(obs, reward, terminated, truncated,
  info)`.
- **Done when:** env passes `gymnasium.utils.env_checker`.

### 2.2 Observation builder — M, CP
- **Do:** morphology-agnostic proprioception.
- **Method:** `sim/obs.py`: concatenate per-joint `qpos/qvel`, root orientation
  (gravity vector in body frame), root linear/angular velocity, foot/contact
  flags, previous action. Emit a **per-actuator/per-joint structured** obs (list
  of node features) too, for the modular policy in Stage 3.
- **Done when:** obs vector + structured obs have correct, documented dims for
  both seeds.

### 2.3 Action mapping — S, CP
- **Do:** map policy outputs to actuator controls.
- **Method:** clip/scale `[-1,1]` actions to actuator `ctrlrange`; in `sim/env.py`.
- **Done when:** actions drive actuators within range; out-of-range is clipped.

### 2.4 Locomotion reward + termination — M, CP
- **Do:** define the selective pressure.
- **Method:** `sim/tasks/locomotion.py`: reward = `w_v·forward_velocity −
  w_e·energy(Σ act²·dt) − w_ctrl·||Δaction|| + alive_bonus`; terminate on
  fall (root height/orientation threshold) or `t > T`. Energy term ties to the
  metabolic budget (Stage 6).
- **Done when:** reward is finite, deterministic given (seed, policy), and a
  forward-moving rollout scores higher than a still one.

### 2.5 Niche/task config — S
- **Do:** make tasks declarative.
- **Method:** `NicheTask` dataclass (terrain, direction, time limit, reward
  weights, escalation params); locomotion as the first instance.
- **Done when:** task fully specified by config (F.3).

### 2.6 Deterministic stepping & frame-skip — S, CP
- **Do:** fix control rate vs physics rate.
- **Method:** `control_dt = n_substeps · model.opt.timestep`; document; seed
  `mjData`.
- **Done when:** identical (seed, action-sequence) → identical trajectory (test).

### 2.7 Open-loop smoke test — S, CP
- **Do:** prove the reward channel isn't degenerate before adding RL.
- **Method:** drive joints with tuned sinusoids; measure forward progress.
- **Done when:** the biped seed moves forward measurably under open-loop CPG-like
  input.

**Stage 2 DoD:** any compiled creature runs episodes in a configurable locomotion
niche with a sane, deterministic reward; checked with env_checker + open-loop.

**Stage 2 risks:** reward hacking (creature dives/falls forward) → tune
alive_bonus + orientation term; sim instability for odd bodies → clamp actuator
forces, raise solver iterations.

---

## Stage 3 — Learned control (PPO) + morphology-aware policy + warm-start  **(CP)**

**Goal:** each body *learns* to drive its own anatomy; controllers transfer across
kin to keep co-evolution affordable. **Depends on:** Stage 2. **Effort:** L–XL.

### 3.1 Per-body MLP PPO baseline — M, CP
- **Do:** train one creature to locomote.
- **Method:** PPO from **Stable-Baselines3** (fast start) or **CleanRL**
  (hackable). MLP actor-critic over the flat obs (2.2). Vectorized envs
  (`SubprocVecEnv` / Gymnasium vector).
- **In→Out:** env + config → trained policy + return curve.
- **Done when:** return rises and the policy moves the biped seed past a distance
  threshold; curve logged (F.4).

### 3.2 Training infrastructure — M
- **Do:** repeatable training with checkpoints/eval.
- **Method:** `control/train.py` (config-driven), periodic eval rollouts, best-
  checkpoint saving, deterministic eval seeds.
- **Done when:** training resumes from checkpoint; eval is reproducible.

### 3.3 Morphology-aware (modular/GNN) policy — XL, CP
- **Do:** a policy that handles *variable* morphology and shares weights across
  body parts (prereq for transfer/warm-start).
- **Method:** implement **Shared Modular Policies** (Huang et al. 2020) or a
  **NerveNet**-style GNN: one shared module per actuator-node, message passing
  along the body graph (edges from 1.1), per-node action outputs. PyTorch +
  `torch_geometric` (or hand-rolled message passing).
- **In→Out:** structured per-node obs (2.2) + body graph → per-actuator actions.
- **Done when:** the *same* policy network controls both seeds (different actuator
  counts) and matches/beats the per-body MLP on the biped.

### 3.4 Parent→child warm-start — M, CP
- **Do:** initialize a child's controller from its parent's.
- **Method:** because modular weights are shared across nodes, copy parent module
  weights; map nodes by graph correspondence (matched subgraph) and initialize
  new/changed modules from the nearest shared module. `control/warmstart.py`.
- **Done when:** a child with a small mutation starts above-random and reaches
  competence in **far fewer** steps than from scratch (measured).

### 3.5 Fine-tune budget & fairness ablation — M, CP
- **Do:** ensure bodies are judged after adequate adaptation (credit assignment).
- **Method:** fix a per-body fine-tune step budget; ablation sweeping the budget
  to find the knee where a body's score plateaus.
- **Done when:** chosen budget puts ≥90% of bodies past their score knee; a "good
  body, undertrained brain" is not misjudged (documented test).

### 3.6 Transfer test — S, CP
- **Do:** quantify warm-start savings.
- **Method:** compare steps-to-threshold from-scratch vs warm-started across 20
  child morphologies.
- **Done when:** median speedup ≥ target (e.g., ≥3×), logged.

**Stage 3 DoD:** a single modular policy controls arbitrary evolved bodies, learns
locomotion, and warm-starts from parents within a fair, bounded budget.

**Stage 3 risks:** RL instability/cost (→ SB3 defaults first, then tune; PPO
clip/entropy); modular policy underperforms specialists (→ allow short per-body
fine-tune); compute (→ Stage 9 GPU).

---

## Stage 4 — Macro-mutations + innovation log

**Goal:** the operators that *invent* structure, with a record of when each
novelty arose. **Depends on:** Stage 1 (+ Stage 3 to evaluate). **Effort:** L.

### 4.1 Micro-mutation — S
- **Do:** perturb continuous params.
- **Method:** `encoding/mutate.py::micro(genome, rng, sigma)` — Gaussian on
  sizes, joint ranges/stiffness, muscle Hill params; clamp to physical positives.
- **Done when:** output compiles; values stay positive/in-range (test).

### 4.2 Macro-mutation operators — L, CP
- **Do:** structural edits.
- **Method:** each as a pure `genome→genome` op in `encoding/mutate.py`, returning
  a typed `MutationRecord`:
  `add_part`, `delete_subtree` (mass refund), `duplicate_subtree`,
  `change_recursion_count`, `toggle_symmetry`, `add_muscle`, `remove_muscle`,
  `reroute_tendon`, `split_muscle` (→ 2 independently-controlled compartments),
  `fuse_muscles`, `add_joint_dof`, `remove_dof`. Each picks valid targets
  (existing sites/parts) via the rng.
- **Done when:** each operator has a unit test producing a still-developable (or
  intentionally-rejected, Stage 6) genome.

### 4.3 Mutation scheduling — S
- **Do:** control innovation rate over deep time.
- **Method:** `macro_rate = schedule(generation)` (config); mostly micro, rare
  macro; optionally anneal.
- **Done when:** observed macro-event frequency matches the schedule (logged).

### 4.4 Structural diff / innovation detector — M, CP
- **Do:** detect and classify what changed parent→child.
- **Method:** `encoding/innovation.py`: compare graphs (node/edge/muscle sets +
  structural signature / graph-edit-distance); emit flags `{+limb, −limb,
  +muscle, +compartment, topology_change, dof_change}`.
- **Done when:** hand-built before/after pairs are classified correctly (tests).

### 4.5 Phylogeny edge recording — S, CP
- **Do:** persist lineage with mutations + innovations.
- **Method:** write `lineage(parent,child,mutation_json,innovation_flags)` to the
  DB (F.5); in-memory `Phylogeny` tree for queries.
- **Done when:** a creature's ancestry + its innovation timeline is queryable.

**Stage 4 DoD:** mutation operators reliably produce developable, novel genomes;
every parent→child step is logged with classified innovations.

---

## Stage 5 — Quality-diversity archive (pyribs)

**Goal:** stop optimizing one reward; fill and keep evolving a map of distinct
specialists. **Depends on:** Stages 1–4. **Effort:** M–L.

### 5.1 Descriptor definition & measurement — M, CP
- **Do:** choose and compute behavior+morphology descriptors.
- **Method:** `evo/descriptors.py` measuring per creature: strength/mass, max
  speed, explosive power, aerobic economy, **limb count**, **mass-distribution
  PCA / body-plan class**, **morphological distance from seed**, gait symmetry.
  Start with 2–3, scale to 4–6.
- **Done when:** descriptors are deterministic and span a meaningful range across
  sample creatures.

### 5.2 pyribs archive — M, CP
- **Do:** scalable archive replacing the hand-rolled grid.
- **Method:** **pyribs** `GridArchive` (≤3 dims) or **`CVTArchive`** (≥4 dims,
  Voronoi cells). Store genome JSON + controller ref + descriptors + fitness.
- **Done when:** insert/replace works; coverage + QD-score computed; matches
  scaffold semantics (best-per-cell, specialist kept).

### 5.3 Operator-based ask/tell loop — M, CP
- **Do:** drive QD with our *graph* mutation operators (CMA-ES emitters assume
  continuous vectors and don't fit graph genomes).
- **Method:** use the pyribs archive for bookkeeping but a **custom emitter**:
  `ask()` = sample elites + apply micro/macro mutation (Stage 4); `tell()` =
  develop → control (warm-start, bounded fine-tune) → evaluate → insert.
- **Done when:** end-to-end generations run; archive fills over time.

### 5.4 QD metrics & snapshots — S
- **Do:** track progress.
- **Method:** log coverage, QD-score, max-fitness, novelty per generation;
  periodic archive snapshots (heatmaps for ≤2 descriptor dims).
- **Done when:** plots regenerate from logs; reproducible per seed.

### 5.5 Specialist-preservation & reproducibility tests — S
- **Do:** guard QD semantics.
- **Method:** test that a niche elite poor elsewhere isn't overwritten; fixed seed
  reproduces archive.
- **Done when:** both tests pass.

**Stage 5 DoD:** a reproducible QD loop that co-evolves bodies+brains and grows a
diverse archive of specialists, measured by coverage/QD-score/novelty.

---

## Stage 6 — Per-structure biological budgets + viability rejection

**Goal:** the discipline that makes evolved sci-fi *coherent* — reject/penalize
impossible bodies before wasting sim on them. **Depends on:** Stage 1; integrate
into Stage 5. **Effort:** L.

### 6.1 Per-structure cost accounting — M, CP
- **Do:** compute real costs from the actual morphology.
- **Method:** extend `budgets.py`: mass from geom volume×density; metabolic from
  muscle volume×activation (from sim logs); heat = metabolic power vs `BSA`
  dissipation; neural complexity = #independent actuators; recovery from
  load/damage. Returns per-structure + total overflow.
- **Done when:** costs scale correctly with added structures (test: adding a limb
  raises mass/metabolic/neural).

### 6.2 Geometric viability (pre-sim) — M, CP
- **Do:** reject self-intersecting / immobile bodies cheaply.
- **Method:** `evo/viability.py`: at rest pose, broadphase AABB + MuJoCo contact
  count for non-adjacent parts (self-intersection); assert each joint has
  non-degenerate range (ROM preserved); each muscle has non-zero moment arm
  (1.6).
- **Done when:** constructed self-intersecting / zero-moment-muscle bodies are
  rejected; valid ones pass.

### 6.3 Structural-failure check — M
- **Do:** reject bodies that would instantly fail under predicted load.
- **Method:** estimate peak muscle/tendon/bone stress (Hill `Fmax`, tendon
  capacity, bone capacity gene) vs cross-section; flag immediate failure.
- **Done when:** an under-built attachment carrying a huge muscle is flagged
  (test).

### 6.4 Supply/connectivity check — S, CP
- **Do:** every muscle must be reachable (abstract blood/neural supply).
- **Method:** graph connectivity from root to each muscle's innervation; reject
  orphans.
- **Done when:** a disconnected muscle is rejected (test).

### 6.5 Reject-before-sim integration — S, CP
- **Do:** save compute by gating.
- **Method:** in Stage-5 `tell()`, run 6.2–6.4 before develop/sim; log rejections
  with reasons; apply 6.1 overflow as a fitness penalty for the survivors.
- **Done when:** rejected genomes never reach RL; rejection reasons logged.

### 6.6 Invalid-body test battery — S
- **Do:** lock viability behavior.
- **Method:** `tests/test_viability.py` with crafted invalid bodies for each rule.
- **Done when:** all pass.

**Stage 6 DoD:** invalid/over-built bodies are rejected pre-sim or penalized;
every surviving creature pays for its structures.

---

## Stage 7 — Multi-niche selective regimes + speciation/phylogeny

**Goal:** divergent lineages specialize and *stay* distinct (no averaging back to
one body plan). **Depends on:** Stages 2–6. **Effort:** XL.

### 7.1 Implement the niche tasks — XL, CP
- **Do:** the gauntlet as MuJoCo environments, each a persistent pressure.
- **Method:** `sim/tasks/`: `track` (sprint, done as locomotion variant),
  `iron_zone` (push/hold against load → max force), `ballistics` (launch a mass /
  jump → distance/velocity), `endurance` (sustained locomotion with energy
  depletion + heat), `chaos_grid` (random pushes → stay upright), `terrain`
  (slopes/obstacles via heightfield). Each: env + reward + `escalate()` schedule.
- **Done when:** each niche runs, rewards specialization, and escalates difficulty.

### 7.2 Morphological distance metric — M, CP
- **Do:** quantify how different two body plans are.
- **Method:** `evo/distance.py`: graph-edit-distance on genome graphs +
  descriptor-space distance (5.1); combine.
- **Done when:** human-vs-human ≪ human-vs-tailed (sanity test).

### 7.3 Speciation / niching in selection — L, CP
- **Do:** protect divergent lineages from being out-competed by a dominant plan.
- **Method:** cluster genomes into species by 7.2 distance (threshold or k-means
  in descriptor space); restrict replacement/competition within species; optional
  within-species variation bias. Integrate into the Stage-5 emitter.
- **Done when:** ≥N species persist across a long run (species-count-over-time
  plot doesn't collapse to 1).

### 7.4 Phylogeny persistence & export — M
- **Do:** the family tree as a primary artifact.
- **Method:** export DB lineage to **Newick**/JSON; annotate innovation events
  (4.4) on branches; `evo/phylogeny.py`.
- **Done when:** a tree file loads in a tree viewer / Stage-10 Blender with
  innovation markers.

### 7.5 Lineage-divergence metrics — S
- **Do:** measure that niches drive distinct morphologies.
- **Method:** per-niche mean morphology + inter-niche distance over generations.
- **Done when:** niches separate in morphology space (plot).

### 7.6 Per-niche specialization validation — M
- **Do:** confirm specialists beat generalists *in-niche* and pay for it
  elsewhere.
- **Method:** cross-evaluate each niche's elite on all niches (a tradeoff matrix).
- **Done when:** the matrix shows a clear specialist diagonal.

**Stage 7 DoD:** multiple niches yield mechanically-distinct, persistent lineages
with a queryable, innovation-annotated phylogeny.

---

## Stage 8 — Open-endedness: novelty + environment co-evolution  🔬

**Goal:** keep the system *inventing* over deep time instead of plateauing. This
is a research frontier — acceptance is "measurable + attempted", not "solved."
**Depends on:** Stages 5–7. **Effort:** XL, 🔬.

### 8.1 Behavior characterization — M
- **Do:** a vector describing *what a creature does* (for novelty).
- **Method:** `evo/behavior.py`: trajectory features (path, gait spectrum,
  contact pattern) → fixed-length BC; optionally a learned embedding.
- **Done when:** similar behaviors map near each other (sanity test).

### 8.2 Novelty search + local competition — L, 🔬
- **Do:** reward doing something new.
- **Method:** novelty = mean distance to k-NN in BC archive; combine with QD via
  **Novelty Search with Local Competition (NSLC)** / surprise; add a novelty
  emitter to Stage 5.
- **Done when:** novelty-on run explores more of descriptor+BC space than
  novelty-off (ablation).

### 8.3 POET (paired open-ended trailblazer) — XL, 🔬
- **Do:** co-evolve environments with agents to generate an endless curriculum.
- **Method:** `evo/poet.py`: maintain (env, agent) pairs; **mutate environments**
  (terrain/task params); **transfer** agents across envs; gate new envs by
  **minimal-criterion coevolution (MCC)** + novelty; retire trivial/solved envs.
- **Done when:** the active-env set keeps turning over with rising difficulty
  (logged), agents transfer between envs.

### 8.4 Open-endedness metrics — M, 🔬
- **Do:** detect plateau vs sustained innovation.
- **Method:** track innovation-event rate, archive coverage growth, and an
  **ANNECS-style** accumulated-novel-environments-solved curve.
- **Done when:** metrics computed over a long run; plateau (if any) is identified
  and analyzed.

### 8.5 Deep-time experiment — L, 🔬
- **Do:** the headline run.
- **Method:** long multi-niche run with novelty+POET; snapshot phylogeny +
  archive periodically.
- **Done when:** produces a deep, branching tree with clearly non-human lineages +
  an innovation timeline.

**Stage 8 DoD:** open-endedness is *measured*; novelty+POET demonstrably broaden
exploration vs baselines (even if indefinite open-endedness remains unsolved).

---

## Stage 9 — GPU scale (MJX/Brax) + surrogates

**Goal:** make deep-time runs feasible in hours–days, not weeks. **Depends on:**
Stages 2–5 (more valuable after 7). **Effort:** XL.

### 9.1 Port env to MJX/Brax — XL, CP-for-scale
- **Do:** GPU-vectorized physics.
- **Method:** re-express `CreatureEnv` for **MuJoCo MJX** (JAX) or **Brax**;
  thousands of parallel envs; jit/vmap rollouts. Flag MJX contact limitations for
  complex bodies; keep the CPU MuJoCo path as ground-truth fallback.
- **Done when:** identical-ish dynamics vs CPU on a fixed creature (within tol);
  throughput ≥100× parallel envs.

### 9.2 JAX policy & PPO — L
- **Do:** GPU training.
- **Method:** reimplement modular policy + PPO in JAX (Flax/Haiku) or use Brax PPO.
- **Done when:** matches CPU policy quality on the biped; far faster.

### 9.3 QDax archive/emitters — L
- **Do:** GPU quality-diversity.
- **Method:** port the archive + operator emitter to **QDax** (JAX MAP-Elites).
- **Done when:** QD-score parity with pyribs at much higher throughput.

### 9.4 Surrogate-assisted QD — L
- **Do:** skip simulating obviously-bad candidates.
- **Method:** train a small NN/GP mapping genome features → predicted
  fitness/descriptors; pre-filter `ask()` candidates; active-learning: simulate
  uncertain/promising ones, update surrogate (surrogate-assisted MAP-Elites / DDE).
- **Done when:** matches no-surrogate QD-score with materially fewer sims
  (ablation).

### 9.5 Throughput & surrogate benchmarks — S
- **Do:** quantify the speedups.
- **Method:** evals/sec CPU vs GPU; sims-to-QD-score with/without surrogate.
- **Done when:** numbers reported in `runs/`.

**Stage 9 DoD:** deep-time multi-niche runs are GPU-feasible; surrogates cut sim
cost without hurting QD.

**Stage 9 risks:** MJX contact fidelity for exotic bodies (→ CPU fallback for
final eval); JAX reimplementation cost (→ port incrementally, keep CPU oracle).

---

## Stage 10 — Blender visualization (the payoff to look at)

**Goal:** see the creatures, their motion, and the family tree of body plans.
**Depends on:** Stages 1, 7 (+ sim logs). **Effort:** L.

### 10.1 Morphology → Blender — M
- **Do:** build a creature in Blender from its genome/morphology.
- **Method:** `viz/blender_build.py` via `bpy` (this env has a Blender MCP, or run
  `blender --background --python`): create armature + meshes from bodies/joints.
- **Done when:** a seed creature appears correctly in Blender.

### 10.2 Trajectory replay — M, CP-for-viz
- **Do:** animate a simulated rollout.
- **Method:** export `qpos` over time from a MuJoCo eval; keyframe Blender bones;
  render to video.
- **Done when:** a locomotion replay video renders.

### 10.3 Phylogeny tree render — M
- **Do:** the anatomical family tree.
- **Method:** from Newick/JSON (7.4); lay out branches; place innovation markers
  (+limb/+muscle/topology) at the branch where each first appears; render.
- **Done when:** a tree image shows lineages + innovation timeline.

### 10.4 Force-vector / moment-arm overlays — M
- **Do:** show the biomechanics.
- **Method:** from sim logs, draw arrows muscle-origin→insertion scaled by force;
  visualize moment arms about joints.
- **Done when:** an annotated biomechanics clip renders.

### 10.5 Side-by-side lineage comparison — S
- **Do:** show drift across generations.
- **Method:** array N ancestors→descendant morphologies in one scene/render.
- **Done when:** a comparison image renders.

### 10.6 Batch render pipeline — S
- **Do:** automate viz for any run.
- **Method:** `scripts/render_run.py` pulls top elites + lineage from the DB and
  renders all of the above headless.
- **Done when:** one command produces a run's full visual report.

**Stage 10 DoD:** any run yields creature replays, a biomechanics overlay, and an
innovation-annotated phylogeny render.

---

## Stage 11 — Reality anchor (optional, low priority)

**Goal:** keep only the *near-human* region of the tree plausible if real data is
supplied. **Depends on:** Stage 1; independent of deep-time work. **Effort:** M.

### 11.1 Whoop ingest — S
- **Do:** import sleep/recovery/HR.
- **Method:** parse the existing "Sync Whoop" export → `Metric`s with
  status/source/timestamp/CI (`io/whoop.py`).
- **Done when:** metrics populate with provenance; no values invented.

### 11.2 Lifting-app ingest — S
- **Do:** import strength history from `WeightliftingAppData.wld`.
- **Method:** reverse the `.wld` format → 1RM/volume `Metric`s (`io/lifting.py`).
- **Done when:** lifts populate with provenance.

### 11.3 Seed-metric update — S
- **Do:** set the seed from real data where available.
- **Method:** map metrics → seed genome priors; keep unknowns as priors.
- **Done when:** seed reflects measured values; CIs narrow accordingly.

### 11.4 Bayesian near-human update — M
- **Do:** calibrate the near-human region to observed training response.
- **Method:** predict response to a small block; compare; Kalman/linear-Gaussian
  update on a few parameters; **only** within HUMAN-ACHIEVABLE.
- **Done when:** near-human predictions match held-out measurements within CI;
  deep-time branch untouched.

### 11.5 Anchor validation — S
- **Do:** prove the anchor doesn't contaminate the speculative branch.
- **Method:** assert OE lineages are unaffected by the anchor update.
- **Done when:** test passes.

**Stage 11 DoD:** with real data, near-human descendants are calibrated and
labeled; the open-evolution tree remains independent.

---

## Critical path, tracks & milestones

**Critical path:** F.1–F.2 → **1** (encoding+morphogenesis) → **2** (physics
niche) → **3** (control) → **4** (macro-mutation) → **5** (QD) → **6** (viability)
→ **7** (multi-niche+speciation) → **8** (open-endedness). Stages **9** (scale),
**10** (viz), **11** (anchor) are parallelizable side-tracks once their deps land.

**Parallelizable:** F.3–F.6 alongside Stage 1; Stage 10 viz alongside 4+; Stage 9
GPU alongside 5–7; Stage 11 anytime after Stage 1.

**Demo milestones (each is a shareable artifact):**
- **M1 (end Stage 1):** a non-human genome grows + renders in MuJoCo. *Proves
  invention is possible.*
- **M2 (end Stage 3):** that creature *learns to move*. *Proves body+brain works.*
- **M3 (end Stage 5):** a reproducible archive of diverse co-evolved creatures.
- **M4 (end Stage 7):** distinct specialized lineages + an innovation-annotated
  phylogeny. *This is the core product.*
- **M5 (end Stage 8):** a deep-time run with measured open-endedness + a branching
  tree of non-human specialists. *The headline.*
- **M6 (Stage 10):** the rendered family-tree-of-body-plans + creature replays.

## Effort summary (rough, solo)

| Stage | Effort | On CP |
|---|---|---|
| F Foundations | M (spread) | F.1–F.2 yes |
| 0 Hardening | S | no |
| 1 Encoding+morphogenesis | XL | **yes** |
| 2 Physics niche | M | **yes** |
| 3 Control | L–XL | **yes** |
| 4 Macro-mutation | L | **yes** |
| 5 Quality-diversity | M–L | **yes** |
| 6 Budgets/viability | L | **yes** |
| 7 Multi-niche+speciation | XL | **yes** |
| 8 Open-endedness 🔬 | XL | **yes** |
| 9 GPU scale+surrogates | XL | scale-only |
| 10 Blender viz | L | no |
| 11 Reality anchor | M | no |

## Global risks (recap, see PLAN §22)

Open-endedness plateau (8) · simulator exploitation (2,9) · compute cost (3,9) ·
body/brain credit assignment (3) · genome bloat/invalid bodies (1,6) ·
unfalsifiable by design (embraced; validate the engine, label drift). Mitigations
live in the per-stage risk notes above.
