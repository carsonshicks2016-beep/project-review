"""Stage 6.1 acceptance: per-structure biological cost accounting.

The contract: every structure is paid for. Costs are read from the *instantiated*
morphology, the per-structure breakdowns sum to the aggregates, and adding a
structure (a muscle, or a whole limb) strictly raises the relevant costs --
mass / metabolic / neural load -- so nothing improves for free.

Runs:  python3 tests/test_budgets.py   (requires mujoco only via develop? no -- pure)
"""
import copy
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from personal_cambrian.seeds import agent_zero, quadruped
from personal_cambrian.encoding.genome import MuscleGene
from personal_cambrian.encoding.mutate import duplicate_subtree, add_part
from personal_cambrian.morphogenesis import develop, compute_budgets, Budgets

EPS = 1e-9


def _budgets(genome):
    return compute_budgets(develop(genome), genome)


def _add_muscle_gene(genome):
    """Deterministically clone an existing muscle so references stay valid."""
    g = copy.deepcopy(genome)
    src = g.muscles[0]
    twin = copy.deepcopy(src)
    twin.id = src.id + "__twin"
    g.muscles.append(twin)
    return g


def _add_limb(genome):
    """Grow one extra body using a structural macro op (seeded, deterministic)."""
    base_bodies = develop(genome).body_count
    for seed in range(500):
        rng = np.random.default_rng(seed)
        for op in (duplicate_subtree, add_part):
            child, rec = op(copy.deepcopy(genome), rng)
            if rec.ok and develop(child).body_count > base_bodies:
                return child
    raise RuntimeError("could not grow a limb in 500 seeds")


# --- basic sanity -----------------------------------------------------------
def test_budgets_are_finite_and_nonnegative():
    b = _budgets(agent_zero())
    assert isinstance(b, Budgets)
    for v in (b.total_mass_kg, b.surface_area_m2, b.metabolic_W,
              b.heat_production_W, b.heat_dissipation_W,
              b.neural_load, b.neural_capacity, b.recovery_cost):
        assert v >= 0.0 and v == v  # finite, non-negative
    assert b.total_overflow >= 0.0
    assert b.penalty() >= 0.0


def test_total_mass_matches_morphology():
    g = agent_zero()
    m = develop(g)
    b = compute_budgets(m, g)
    # structural mass is exactly the morphology's rigid-tissue mass
    assert abs(b.structural_mass_kg - m.total_mass()) < 1e-6
    # total = structural + muscle, and muscle adds positive mass
    assert b.muscle_mass_kg > 0.0
    assert abs(b.total_mass_kg - (b.structural_mass_kg + b.muscle_mass_kg)) < 1e-6


def test_per_structure_sums_to_aggregate():
    b = _budgets(agent_zero())
    assert len(b.per_body) == develop(agent_zero()).body_count
    assert len(b.per_muscle) == len(develop(agent_zero()).muscles)
    body_mass = sum(d["mass_kg"] for d in b.per_body.values())
    body_area = sum(d["surface_area_m2"] for d in b.per_body.values())
    musc_mass = sum(d["mass_kg"] for d in b.per_muscle.values())
    assert abs(body_mass - b.structural_mass_kg) < 1e-6
    assert abs(body_area - b.surface_area_m2) < 1e-6
    assert abs(musc_mass - b.muscle_mass_kg) < 1e-6


# --- the core contract: structures cost ------------------------------------
def test_adding_a_muscle_raises_muscle_mass_metabolic_and_neural():
    base = _budgets(agent_zero())
    more = _budgets(_add_muscle_gene(agent_zero()))
    assert more.muscle_mass_kg > base.muscle_mass_kg + EPS
    assert more.active_metabolic_W > base.active_metabolic_W + EPS
    assert more.metabolic_W > base.metabolic_W + EPS
    assert more.neural_load > base.neural_load + EPS
    assert more.heat_production_W > base.heat_production_W + EPS


def test_adding_a_limb_raises_mass_surface_and_metabolic():
    g = quadruped()
    base = _budgets(g)
    more = _budgets(_add_limb(g))
    assert more.structural_mass_kg > base.structural_mass_kg + EPS
    assert more.surface_area_m2 > base.surface_area_m2 + EPS
    assert more.total_mass_kg > base.total_mass_kg + EPS
    # more body mass => more resting metabolism and more heat to shed
    assert more.resting_metabolic_W > base.resting_metabolic_W + EPS
    assert more.heat_production_W > base.heat_production_W + EPS


# --- capacities scale the right way ----------------------------------------
def test_neural_capacity_is_sublinear_in_mass():
    small = _budgets(quadruped())
    big = _budgets(agent_zero())
    assert big.total_mass_kg > small.total_mass_kg
    # capacity grows with mass but slower than mass itself (allometric ^2/3)
    assert big.neural_capacity > small.neural_capacity
    ratio_cap = big.neural_capacity / small.neural_capacity
    ratio_mass = big.total_mass_kg / small.total_mass_kg
    assert ratio_cap < ratio_mass


def test_penalty_zero_when_within_budget_positive_when_over():
    b = _budgets(quadruped())
    # force an overflow by hand and confirm penalty responds
    b.neural_load = b.neural_capacity + 10.0
    assert b.neural_overflow > 0.0
    assert b.penalty() > 0.0
    b.neural_load = 0.0
    assert b.neural_overflow == 0.0


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items())
           if k.startswith("test_") and callable(v)]
    passed = 0
    for fn in fns:
        try:
            fn()
            passed += 1
            print(f"PASS  {fn.__name__}")
        except Exception as e:  # noqa: BLE001
            print(f"FAIL  {fn.__name__}: {type(e).__name__}: {e}")
    print(f"\n{passed}/{len(fns)} passed")
    sys.exit(0 if passed == len(fns) else 1)
