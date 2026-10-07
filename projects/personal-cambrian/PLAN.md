# PERSONAL CAMBRIAN — PLAN.md (authoritative)

An **open-ended artificial-life / evolution engine**. It takes one real human as
the **ancestral seed** ("Agent Zero") and simulates evolution over deep time:
divergent lineages, each under a distinct and escalating selective pressure,
drift away from human and invent genuinely new — eventually non-human, sci-fi —
anatomy and control, each becoming the best possible version of its specialty.

> **North star.** The subject is **evolution**, not the user. Agent Zero is the
> starting body, nothing more. The prize is the **tree of descendants** and the
> novel body plans it grows — sprinters, throwers, climbers, endurance forms,
> and lineages that leave the human envelope entirely.
>
> **What "success" means here:** *mechanically-coherent novelty*. A result is
> good if it is (a) **new** (a body plan/structure not seen before), (b)
> **physically real** (it actually moves in a physics simulator), (c)
> **biologically paid for** (it obeys mass/energy/heat/recovery/viability
> budgets), and (d) **better at its specialty** than its ancestors. It is **not**
> judged by whether it predicts the real user. This project is unfalsifiable by
> design — the rigor is *internal coherence*, not real-world prediction.

> **Honesty tags** stay: `[EVIDENCE]` citable, `[APPROX]` motivated
> approximation, `[SPECULATION]` plausible extrapolation, `[SCI-FI]` open
> evolution. Here `[SCI-FI]` is the **intended output**, not a disclaimer to hide
> it — but it must always be *coherent* sci-fi, never arbitrary.

> **Status (2026-06-22):** A **Stage-0 scaffold** runs (NumPy). It proves the
> bookkeeping (seed schema, biological budgets, quality-diversity archive,
> lineage labels) but it can only *rescale* a human — **it cannot yet invent body
> plans.** The novelty engine (generative encoding + physics + co-evolved
> control + deep-time speciation) is the real build and is **not done**.

---

## 0. What changed from v1 (read this first)

v1 of this plan was mis-framed as a *personal-performance predictor* ("what can
the real Carson achieve"). That is **not** the project. The corrected framing:

| v1 (wrong center of gravity) | This plan (correct) |
|---|---|
| Predict the real user's trainable future | **Evolve a tree of descendants** from the user as seed |
| Reality loop / precise twin is central | Seed only needs to be a *plausible* human; reality loop is optional polish |
| OPEN-EVOLUTION = untrustworthy toy, deprioritized | **OPEN-EVOLUTION is the goal**, the main build target |
| Realism envelopes *clamp* mutations to human | Realism is a **descriptive label**; structural innovation is *unbounded*, filtered by physics + cost, not by human limits |
| Fitness = improvement vs the user | Fitness = **specialty performance − biological cost − viability**, within QD niches; "vs seed" is just a progress readout |
| Analytic biome formulas are fine | Formulas only work for ~human bodies; **novel bodies must be physically simulated** |

## 1. Project definition

> PERSONAL CAMBRIAN is a modular open-ended-evolution platform. From a single
> human seed it co-evolves **morphology** (via a generative developmental
> encoding that can add, remove, duplicate, and reroute structures) and
> **control** (learned neural policies), evaluates each creature in a **general
> physics simulator** across niche-specific selective pressures, charges every
> body its real biological costs, and preserves a **quality-diversity archive +
> phylogeny** of specialists — from near-human athletes to wholly novel
> body plans — labeling each by how far it has drifted from human and explaining
> what each retained innovation does, what it costs, and which ancestor produced
> it.

## 2. The evolutionary thesis

- **Deep time = generations, not literal years.** "Millions of years" is modeled
  as many generations of heritable variation + selection; divergence accumulates.
- **Divergence by niche.** Each biome is a *persistent, escalating selective
  regime*. A lineage committed to one niche keeps climbing that specialty's peak
  and drifts structurally away from generalist-human.
- **Innovation, not just tuning.** Micro-mutations tune parameters; **macro-
  mutations** add/remove/reroute structures (a new limb, a duplicated muscle
  head, a spring tendon, an extra joint). Macro-mutations are how new body plans
  appear. `[SCI-FI]` once they leave the human envelope.
- **Exaptation.** Structures evolved for one purpose get repurposed by descendant
  lineages — the engine should allow and surface this.
- **The hard, unsolved core: open-endedness.** Evolutionary systems tend to find
  one trick and plateau. Sustaining *continuing* invention over deep time is a
  genuine research frontier (novelty search, POET-style environment co-evolution,
  quality-diversity). We treat plateau-avoidance as a first-class objective, not
  an afterthought.
- **Research lineage we build on:** Karl Sims, *Evolving Virtual Creatures*
  (1994); generative/indirect encodings (recursive graphs, CPPN/HyperNEAT,
  L-systems); MAP-Elites quality-diversity; novelty search; POET.

## 3. Scientific assumptions & honest limits

1. Whole-body capability emerges from physics + actuated musculotendon units;
   for **non-human** bodies only a **general simulator** can score it. `[design]`
2. Specific tension σ ≈ 20–35 N/cm²; musculotendon force/velocity/elastic
   behavior follows Hill-type models. `[EVIDENCE]`
3. Every structure costs mass, metabolic energy, oxygen, heat, recovery, and
   neural-control bandwidth; nothing is free. `[APPROX]`
4. Adaptation/development saturates with diminishing returns; overload accrues
   damage. `[APPROX]`
5. **Coefficients are uncalibrated** and **deliberately not** tied to the real
   user. The discipline is coherence (physics conservation, viability, paid
   costs), not predictive accuracy.
6. **Unfalsifiable by design (embraced).** With one seed and speculative deep
   time, outputs cannot be validated against reality. We validate the *engine*
   (does it conserve physics, reject invalid bodies, keep innovating?), not the
   *creatures' realism*.

## 4. System architecture

```
   Agent Zero (human seed) ─┐
                            ▼
   ┌──────────────────────────────────────────────────────────────────────┐
   │  GENERATIVE GENOME  (developmental growth rules: recursive body graph) │  ← §7  THE CORE
   └───────────────┬──────────────────────────────────────────────────────┘
                   │ morphogenesis (genome → segments, joints, muscles)
                   ▼
   ┌──────────────────────────────┐     reject / penalize
   │  VIABILITY + BUDGET CHECK     │────────────────────────►  invalid bodies   ← §12
   └───────────────┬──────────────┘
                   │ valid body → MJCF / model
                   ▼
   ┌──────────────────────────────┐   inner loop (RL): learn to drive THIS body
   │  PHYSICS SUBSTRATE (MuJoCo)   │◀──────────────  CONTROLLER (morphology-aware  ← §8,§9
   │  niche task = selective regime│                 policy, warm-started from parent)
   └───────────────┬──────────────┘
                   │ specialty performance, costs, behavior descriptors
                   ▼
   ┌──────────────────────────────┐   outer loop (evo): mutate genome (micro+macro)
   │  SELECTION: QUALITY-DIVERSITY │   keep best per niche; novelty + POET pressure  ← §10
   │  ARCHIVE  +  PHYLOGENY/LINEAGE│   to avoid plateau; speciate by morph distance  ← §11
   └───────────────┬──────────────┘
                   ▼
   outputs: tree of body plans · novelty timeline · biome dashboards ·
            replayable creature sims · Blender phylogeny of morphologies        ← §24
```

**Two nested loops.** *Inner:* given a body, learn a controller (RL). *Outer:*
evolve bodies (QD over morphology). A new body is judged **after** its controller
adapts, so a good body plan is never discarded for an untrained brain.

## 5. Core data models

- **Metric / AgentZeroProfile** — the human seed with uncertainty (`metrics.py`).
- **GenerativeGenome** — recursive body-graph + growth rules + per-tissue
  parameters (§7). *Replaces the Stage-0 flat 14-gene vector.*
- **Morphology** — instantiated segments, joints, musculotendon units, sensors
  (the phenotype's *structure*); compiles to a physics model (MJCF).
- **Controller** — a morphology-aware policy (graph/modular net) + its weights.
- **Creature = (genome, morphology, controller, phenotype-state, budgets, lineage_id)**.
- **NicheTask** — a persistent selective regime (physics environment + reward).
- **Elite / QDArchive** — niche → best creature; coverage, QD-score, novelty.
- **PhylogenyNode** — parent→child edges tagged with the mutation and any
  **innovation event** (new limb / muscle / topology change).
- Store (DB): SQLite/parquet tables `seed_metrics, genomes, morphologies, evals,
  lineage, innovations, niche_tasks, experiments`.

## 6. Agent Zero — the seed (not the subject)

A **plausible human ancestor**. It does *not* need to be a precise twin. Real
measurements (if supplied) just set a believable starting point; everything
interesting happens downstream of it. Schema in `data/agent_zero.example.json`
keeps the full uncertainty fields (value/unit/CI/status/source/timestamp) and
ships `status=unknown` — we still never invent personal data, but precision here
is **low priority**. Whoop / lifting-app data on this machine can seed plausible
ranges; that's optional polish, not the project.

## 7. Generative / developmental encoding — THE CORE

The flat parameter vector is replaced by a genome that **grows** a body, so
evolution can *invent* structure rather than only resize it.

**Recommended encoding (Sims-style recursive body graph + parameter fields):**
- **Graph genome.** Nodes = body-part *types* (segment with shape/size, set of
  attachment sites, contained musculotendon units). Edges = attachment rules
  (where/orientation/scale) with a **recursion count** so a rule can repeat to
  grow chains/limbs/digits, and **symmetry flags** (bilateral, radial).
- **Developmental rewriting.** Reading the genome runs morphogenesis: expand
  recursive edges, place segments, route muscles between attachment sites,
  innervate them. This yields coherent repeated structures (limbs, spines,
  tails) from compact rules and keeps most mutations valid.
- **Optional parameter fields (CPPN/HyperNEAT) — later.** A function of body
  coordinates sets continuous tissue properties (PCSA, fiber type, tendon
  stiffness) smoothly across the body, giving graded, lifelike variation.

**Musculotendon unit (evolvable struct):** origin region, insertion region,
routing points, per-joint moment arms, belly length, PCSA, pennation, fiber
length, fiber-type mix, tendon length, tendon stiffness/elasticity, max
contraction velocity, max isometric force, metabolic + recovery cost, neural
activation requirement. `[EVIDENCE]` Hill-type parameters.

**Mutation operators.** *Micro* (tune any continuous parameter). *Macro
(innovation):* duplicate a segment/limb, duplicate a muscle head, split a muscle
into independently-controlled compartments, fuse redundant muscles, delete an
unused structure (mass refund), reroute a tendon path, add/remove a joint or
degree of freedom, change a recursion count or symmetry, grow a novel
muscle-tendon path. Macro-mutations are rare per step and are logged as
**innovation events** on the phylogeny.

## 8. Body + brain co-evolution

- **Inner loop (control).** Each new morphology gets a controller via RL (PPO).
  To keep this affordable, use a **morphology-aware policy** (a graph-neural-net
  or modular per-actuator policy) whose I/O adapts to the body's sensors/
  actuators, and **warm-start from the parent's controller** + short fine-tune,
  rather than training every body from scratch.
- **Outer loop (morphology).** Evolve genomes with QD selection (§10).
- **Credit assignment.** Always evaluate after adequate controller adaptation;
  give a fixed fine-tune budget so body plans are compared fairly.

## 9. Evaluation substrate — why physics is now mandatory

Analytic biome formulas (Stage-0) only make sense for roughly-human bodies. **You
cannot write a closed-form sprint equation for a three-legged, tailed creature.**
So real evaluation moves to a **general physics simulator** where the creature
must *physically perform* the niche task. Niches as selective regimes:

- **Track** (locomotion speed/economy) · **Iron Zone** (max force/torque) ·
  **Ballistics** (launch/throw distance/velocity, kinetic-chain transfer) ·
  **Chaos Grid** (balance/agility/perturbation) · **Endurance** (sustained work,
  heat, fatigue) · **Elasticity Field** (rebound/SSC) · **Skill Arena** (accuracy
  under fatigue) · **Durability Trial** (repeated loading over simulated time) ·
  **Real-World Terrain** (slopes/obstacles).

Each niche escalates over generations (POET-style) so lineages never stop being
pushed.

## 10. Selection: quality-diversity + open-endedness

We never optimize one averaged reward. We **fill a map of distinct solutions and
keep evolving it.**
- **MAP-Elites / QD archive** over behavior + morphology descriptors:
  strength-to-mass, max velocity, explosive power, aerobic economy, durability,
  energy cost, **limb count / body-plan class**, **morphological distance from
  human**, **realism level**. (pyribs now; **QDax** for GPU scale.)
- **Novelty pressure.** Reward behavioral/morphological novelty so the search
  keeps entering unexplored body-plan regions (anti-plateau).
- **Environment co-evolution (POET).** Niche tasks get harder as their occupants
  improve, generating an endless curriculum.
- **Specialist protection.** A niche's best is preserved even if it's useless
  elsewhere — that's how specialties survive.

## 11. Deep-time, lineage & speciation

- **Phylogeny log:** every creature records parent, mutation, and any innovation
  event; the tree is a primary output.
- **Speciation:** lineages reproductively/representationally isolate by
  morphological distance, so divergent body plans don't get averaged back
  together.
- **Innovation timeline:** when did the tail / extra limb / spring tendon /
  muscle compartment first appear, in which lineage, and why did it survive
  (what it changed, what it cost).

## 12. Biological budgets & viability — what makes sci-fi *coherent*

These do not constrain creativity; they are what separate believable creatures
from nonsense. A body/structure is rejected or penalized unless it: attaches to
valid skeletal/connective regions; produces a meaningful moment about ≥1 joint;
avoids impossible intersections/routing; preserves needed range of motion; has
abstract blood + neural supply; fits available volume; **pays** mass, metabolic,
heat, recovery, developmental, and control-complexity costs; **withstands** its
predicted forces without immediate failure; and **earns** its keep (net benefit ≥
cost). Budgets implemented in `budgets.py` (mass, oxygen, heat, recovery, neural)
extend to per-structure accounting in the generative engine.

## 13. Fitness & constraint equations

```
specialty(creature, niche) = physics-measured performance at that niche's task
cost   = w_e·energy + w_h·heat_overflow + w_r·recovery_debt + w_n·neural_overflow
viab   = Σ viability_violations (≥0; hard-reject if a body cannot be simulated)
nov    = behavioral/morphological novelty vs the archive

Fitness_in_niche = specialty(creature, niche) − cost − w_v·viab
QD insertion     = keep argmax Fitness per (descriptor cell); also keep high-nov
"progress vs seed" = specialty(creature) / specialty(Agent Zero)   # readout only
```
Realism/implausibility is **not** a penalty here (drifting from human is the
point); it is recorded as a **label/descriptor**. Weights are exposed for
auditing.

## 14. Realism labeling (descriptive) & the optional reality anchor

- **Labels:** HUMAN-ACHIEVABLE ⊂ HUMAN-POSSIBLE ⊂ OPEN-EVOLUTION, applied to
  *describe* how far a descendant has drifted — not to clamp it.
- **Optional reality anchor (low priority):** if real training data is supplied,
  a light Bayesian update can keep the *near-human* region of the tree plausible.
  This is polish; the deep-time branch is the goal and is intentionally
  unanchored.

## 15. Staged roadmap (reordered for open-ended evolution)

> **[ROADMAP.md](ROADMAP.md)** breaks every stage below into granular tasks with
> method-of-action, inputs→outputs, and a "done when" test. This table is the
> overview; ROADMAP.md is the build sheet.

| Stage | Content | State |
|---|---|---|
| 0 | Seed schema, budgets, QD archive, lineage labels (NumPy scaffold) | ✅ built |
| 1 | **Generative encoding + morphogenesis → MJCF** | ⏳ **next, the core** |
| 2 | **MuJoCo** evaluation of one niche (locomotion) | ⏳ |
| 3 | **PPO control**, morphology-aware policy + parent warm-start | ⏳ |
| 4 | Macro-mutations (add/remove/reroute structures) + innovation log | ⏳ |
| 5 | QD archive over morphology descriptors (pyribs) | ⏳ |
| 6 | Per-structure biological budgets + viability rejection | ⏳ |
| 7 | Multi-niche selective regimes + speciation/phylogeny | ⏳ |
| 8 | Open-endedness: novelty pressure + POET environment co-evolution | ⏳ research |
| 9 | GPU scale (MJX/Brax + QDax), surrogates to cut RL cost | ⏳ |
| 10 | **Blender** phylogeny-of-body-plans + creature replays | ⏳ |
| 11 | (optional) reality anchor for the near-human region | ⏳ low-pri |

## 16. What is built now (Stage 0) and how it maps forward

`python3 scripts/run_prototype.py` runs a QD search over a 14-gene **rescaling**
genome on 4 analytic biomes (HA ≈ 67% coverage; OE mode ≈ 92%). Treat it as a
**harness/scaffold**, not the product. Reusable forward: `metrics.py` (seed),
`budgets.py` (cost accounting), `map_elites.py` (QD bookkeeping), `fitness.py`
(multi-objective skeleton), `realism.py` (now descriptive labels), lineage ideas.
To be **replaced**: `genome.py` flat vector → generative encoding (§7); analytic
`biomes/*` → MuJoCo tasks (§9); `phenotype.skill` scalar → learned policy (§8).

## 17. Repository structure

```
personal-cambrian/
  PLAN.md  README.md  requirements.txt
  personal_cambrian/
    metrics.py      # seed schema + uncertainty (KEEP)
    budgets.py      # biological budgets (KEEP, extend per-structure)
    map_elites.py   # QD archive bookkeeping (KEEP, → pyribs/QDax)
    fitness.py      # multi-objective skeleton (KEEP, reframe)
    realism.py      # realism = descriptive label now (KEEP)
    genome.py       # Stage-0 flat vector (REPLACE → generative encoding)
    phenotype.py    # Stage-0 adaptation (REPLACE → morphogenesis + RL)
    biomes/         # Stage-0 analytic tasks (REPLACE → MuJoCo niches)
    # planned:
    morphogenesis/  # genome → segments/joints/muscles → MJCF
    encoding/       # recursive body-graph genome + mutation ops
    control/        # morphology-aware PPO policy + warm-start
    sim/            # MuJoCo niche tasks (selective regimes)
    evo/            # QD + novelty + POET + speciation/phylogeny
    viz/            # Blender phylogeny + creature replays
  data/agent_zero.example.json   # human seed (all-unknown template)
  scripts/  tests/
```

## 18. Technology selections (with reasons)

| Concern | Now | Target | Why |
|---|---|---|---|
| Encoding | flat 14-gene vector | **recursive body-graph (Sims) + CPPN later** | proven generator of *novel* body plans; compact; supports limb repeat/symmetry |
| Physics | analytic formulas | **MuJoCo → MJX/Brax** | only a general simulator can score non-human bodies; GPU-vectorizable |
| Control | skill scalar | **PyTorch PPO, morphology-aware (GNN/modular)** | learns to drive arbitrary bodies; transfers across kin to cut cost |
| Morphology search | Gaussian ES + clamp | **GA/CMA-ES on graph genome** | black-box, handles non-differentiable topology mutation |
| Quality-diversity | hand-rolled grid | **pyribs → QDax (GPU)** | scalable MAP-Elites; novelty archives |
| Open-endedness | none | **novelty search + POET** | the anti-plateau core; escalating niches |
| Phylogeny/store | in-memory | **SQLite/parquet** | lineage, innovations, experiments |
| Viz | ASCII map | **Blender** | morphology growth, force vectors, family tree of body plans |

> Blender is a renderer, never the simulator.

## 19. Pseudocode — the open-ended evolutionary loop

```python
def personal_cambrian(generations, niches):
    seed     = AgentZero()                       # plausible human ancestor
    archive  = QDArchive(descriptors=["morph_class","strength/mass","vmax","novelty"])
    phylo    = Phylogeny(root=seed)

    # found lineages from the seed, one committed to each niche regime
    for niche in niches:
        body = morphogenesis(mutate_genome(seed.genome, macro=True))
        if viable(body):
            ctrl  = learn_controller(body, niche, budget=SHORT)     # inner RL loop
            perf  = simulate(body, ctrl, niche)                     # MuJoCo
            archive.insert(Creature(body, ctrl), fitness(perf, body, niche))

    for g in range(generations):                  # "deep time"
        parent = archive.sample(prefer_novel=True)
        niche  = parent.niche if isolated(parent) else pick_niche(niches)
        genome = mutate_genome(parent.genome,                       # micro + rare macro
                               macro_rate=schedule(g))
        body   = morphogenesis(genome)
        if not viable(body):                       # physics/anatomy/budget reject
            continue
        ctrl   = warm_start(parent.controller, body)
        ctrl   = learn_controller(body, niche, ctrl, budget=SHORT)  # adapt brain to body
        perf   = simulate(body, ctrl, niche)
        child  = Creature(body, ctrl, lineage=parent.lineage)
        archive.insert(child, fitness(perf, body, niche))           # keep best/novel per niche
        phylo.add(parent, child, innovation=detect_innovation(parent, child))
        niche.escalate_if_solved()                                  # POET curriculum

    return archive, phylo    # tree of specialized body plans + when each novelty arose
```

## 20. Validation & testing (coherence, not realism)

The engine — not the creatures' realism — is what we validate:
- **Physics sanity:** energy/momentum conservation within tolerance; no
  exploiting simulator glitches (penalize unphysical force/velocity spikes).
- **Viability gate works:** invalid bodies (self-intersecting, unsupported,
  budget-busting) are rejected, not scored.
- **Morphogenesis determinism:** a genome reproduces the same body (seeded).
- **Control fairness:** ablations showing a body's score stabilizes after the
  fine-tune budget (no premature "inferior body" verdicts).
- **Open-endedness metrics:** archive coverage, QD-score, and **novelty rate**
  keep rising (or detect the plateau we're trying to beat).
- Stage-0 already has 6 passing synthetic checks (`tests/test_validation.py`).

## 21. Computational-cost estimates

- **Stage 0:** ~6k evals in ≈1–2 s (closed-form). Negligible.
- **Body+brain co-evolution (the real cost):** each new body needs controller
  fine-tuning (~1e5–1e6 sim steps with warm-start). Thousands of bodies ⇒
  **hundreds–thousands of GPU-hours** naively. Mitigations are essential:
  morphology-aware/transferable policies, parent warm-start, short fine-tune
  budgets, **surrogate fitness pre-filtering**, GPU-vectorized sim (MJX/Brax) +
  **GPU quality-diversity (QDax)**. With MJX+QDax, thousands of parallel envs make
  a deep-time run feasible in hours–days on one GPU.

## 22. Major risks & limitations

1. **Open-endedness plateau** — the central unsolved problem; mitigated, not
   guaranteed, by novelty search + POET.
2. **Simulator exploitation** — evolved bodies love physics glitches; needs
   realistic actuator/energy limits + physics-sanity penalties.
3. **Compute** — co-evolution is expensive without the GPU/surrogate stack.
4. **Body/brain credit assignment** — a good body with a bad brain looks bad;
   mitigated by warm-start + fair fine-tune budgets.
5. **Genome bloat / invalid bodies** — mitigated by the developmental encoding +
   viability gate.
6. **Unfalsifiable by design** — embraced; we validate the engine, label drift,
   and never present creatures as predictions about the user or as biology fact.

## 23. First concrete implementation task

**Prove the novelty pipeline on the smallest possible slice — one non-human body
that grows from rules, learns to move, and is rendered:**
1. Implement the recursive body-graph genome + `morphogenesis()` → **MJCF** (§7).
2. Hand-author one *non-human* seed variant (e.g., human-ish + a tail or an extra
   leg) to exercise the encoding.
3. Train a PPO controller for a **locomotion** niche in MuJoCo (§8–9).
4. Add the **viability gate** + basic budget rejection (§12).
5. Render the moving creature and confirm it locomotes and is *physically valid*.

This proves encoding → physics → control → eval → viz end-to-end on **one** weird
body, before scaling to macro-mutations, QD, and deep time. (Stage-0 already
covers the seed schema, budgets, and QD bookkeeping it will plug into.)

## 24. Visualization (the actual payoff to look at)

Blender renders: morphology growth/atrophy across a lineage; new limbs/heads;
shifting attachments; changing tendon paths; lost/vestigial structures; animated
force vectors and moment arms; side-by-side lineage comparisons; and the
**anatomical phylogeny** — a family tree of body plans marking when each
innovation first appeared and in which lineage.

## 25. What's real vs speculative (and why that's fine)

- **Real & validated:** the *engine* — generative encoding, morphogenesis,
  physics evaluation, viability/budget gates, QD + phylogeny bookkeeping,
  reproducibility. These must actually work.
- **Coherent speculation (the product):** every evolved creature and body plan.
  They are `[SCI-FI]` by intent. Their credibility comes from **mechanical
  coherence** — they move under real physics and pay real costs — **not** from
  predicting reality. That is the whole point of PERSONAL CAMBRIAN.
```
