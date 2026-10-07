# PERSONAL CAMBRIAN

An **open-ended artificial-life / evolution engine**. It takes one real human as
the **ancestral seed** ("Agent Zero") and simulates evolution over deep time:
divergent lineages, each under a different selective pressure, drift away from
human and invent new — eventually non-human, sci-fi — anatomy and control to
become the best possible version of their specialty.

**The subject is evolution, not the user.** Agent Zero is just the starting body.
The prize is the **tree of descendants** and the novel body plans it grows.
Success = *mechanically-coherent novelty* (new + physically real + biologically
paid-for + better at its specialty), **not** prediction about the real person.

**[PLAN.md](PLAN.md) is the authoritative design doc** (the *why* + architecture).
**[ROADMAP.md](ROADMAP.md) is the detailed build sheet** — every stage broken into
granular tasks with method, inputs→outputs, and a "done when" test. This README is
just a quickstart.

## Status

A **Stage-0 scaffold** runs (NumPy). It proves the bookkeeping — seed schema,
biological budgets, quality-diversity archive, lineage labels — but it can only
*rescale* a human; **it cannot yet invent body plans.** The novelty engine
(generative encoding + physics + co-evolved control + deep-time speciation) is
the real build and is **not done**. See PLAN.md §15 for the roadmap.

## Run it

```bash
python3 tests/run_all.py                  # full test suite (one CI gate)
python3 scripts/build_seeds.py            # write the seed creatures to data/seeds/

# Stage-0 scaffold (the cheap rescaling-genome prototype):
python3 scripts/run_prototype.py          # QD search over a rescaling genome
python3 scripts/run_prototype.py --open   # allow drift past human ranges
python3 scripts/calibrate.py              # archetype bodies vs plausible ranges
```

## Generative encoding (the real engine, ROADMAP Stage 1)

The recursive body-graph genome (`personal_cambrian/encoding/`), morphogenesis to
a concrete creature, and MuJoCo compilation (`personal_cambrian/morphogenesis/`)
are built and tested through Stage 1.9. Two hand-authored seeds live in
`personal_cambrian/seeds.py`: a human-ish biped (Agent Zero) and a non-human
quadruped. See [ROADMAP.md](ROADMAP.md) for status.

## The real build (PLAN.md §15, in order)

1. **Generative encoding + morphogenesis → MJCF** — growth rules that can add/
   remove/reroute/duplicate structures and grow new limbs (the core).
2. **MuJoCo evaluation** — non-human bodies must *physically move* to be scored.
3. **Learned control (PPO)** — a morphology-aware policy, warm-started from the
   parent, so each body learns to drive its own anatomy.
4. **Macro-mutations + innovation log**, then **QD archive** over morphology,
   **per-structure budgets/viability**, **multi-niche speciation/phylogeny**.
5. **Open-endedness** (novelty search + POET), **GPU scale** (MJX/Brax + QDax),
   and **Blender** family-tree-of-body-plans visualization.

## The seed

`data/agent_zero.example.json` is a *plausible human ancestor*, not a precise
twin — precision here is low priority. Fields ship `status: unknown`; the engine
never invents personal values. Whoop / lifting-app data on this machine can seed
plausible ranges (optional polish).

> Not a prediction about any real person, and not biology fact — this is
> coherent speculative evolution. The rigor is internal physical coherence.
