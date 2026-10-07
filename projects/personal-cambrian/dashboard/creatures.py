"""Creature data for the Theater: resolve a genome ref into 3D + anatomy + gauges.

A `ref` is `seed:<name>` (a seed creature) or `node:<id>` (a node in the live
evolution session). Everything returned is plain JSON the Three.js front-end consumes
directly -- no Blender.
"""
from __future__ import annotations

from .session import SESSION


def resolve_genome(ref: str):
    ref = ref or "seed:quadruped"
    if ref.startswith("node:"):
        g = SESSION.genome(ref[5:])
        if g is not None:
            return g, ref
        ref = "seed:" + SESSION.seed_creature
    name = ref.split(":", 1)[1] if ":" in ref else ref
    from personal_cambrian.seeds import SEEDS
    return SEEDS.get(name, SEEDS["quadruped"])(), f"seed:{name}"


def _gauge(label, value, cap, unit):
    ratio = value / cap if cap > 1e-9 else (1.0 if value > 0 else 0.0)
    return {"label": label, "value": round(value, 1), "cap": round(cap, 1),
            "ratio": round(ratio, 3), "unit": unit, "over": ratio >= 0.95}


def budgets_payload(genome, morph) -> dict:
    from personal_cambrian.morphogenesis import compute_budgets
    b = compute_budgets(morph, genome)
    return {
        "heat": _gauge("Heat", b.heat_production_W, b.heat_dissipation_W, "W"),
        "energy": _gauge("Energy", b.recovery_cost, b.recovery_capacity, "W"),
        "neural": _gauge("Neural", b.neural_load, b.neural_capacity, "ch"),
        "metabolic_W": round(b.metabolic_W, 1),
        "penalty": round(b.penalty(), 3),
    }


def creature_payload(ref: str) -> dict:
    from personal_cambrian.morphogenesis import develop
    from personal_cambrian.evo.descriptors import morphology_descriptors
    from personal_cambrian.evo.qd import pre_sim_gate
    from personal_cambrian.viz import plan_from_genome
    g, ref = resolve_genome(ref)
    m = develop(g)
    d = morphology_descriptors(g)
    ok, reasons = pre_sim_gate(g)
    return {
        "ref": ref,
        "plan": plan_from_genome(g, name=ref).to_dict(),
        "budgets": budgets_payload(g, m),
        "stats": {"bodies": m.body_count, "muscles": len(g.muscles),
                  "mass_kg": round(m.total_mass(), 2),
                  "height_m": round(d["height"], 2), "aspect": round(d["aspect"], 2),
                  "symmetry": round(d["symmetry"], 2)},
        "viable": ok, "reject_reasons": reasons,
    }


def trajectory_payload(ref: str, niche: str, steps: int) -> dict:
    from personal_cambrian.viz import record_trajectory
    g, ref = resolve_genome(ref)
    t = record_trajectory(g, niche=niche, n_steps=steps, name=ref)
    d = t.to_dict()
    d["ref"] = ref
    d["root_displacement_m"] = round(t.root_displacement(), 3)
    return d


def biomechanics_payload(ref: str, steps: int) -> dict:
    from personal_cambrian.viz import record_biomechanics
    g, ref = resolve_genome(ref)
    bt = record_biomechanics(g, n_steps=steps, name=ref)
    d = bt.to_dict()
    d["ref"] = ref
    return d
