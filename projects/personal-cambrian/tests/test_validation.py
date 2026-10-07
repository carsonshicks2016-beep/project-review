"""Synthetic validation (roadmap step 8): before trusting the engine on bodies we
can't measure, confirm it behaves correctly on bodies whose answer we know.

Checks:
  * monotonicity   — each gene moves its intended performance the right way
  * specificity    — VO2max drives endurance but NOT sprint top speed
  * budgets bite   — an over-built body is penalized, not rewarded
  * QD fills cells — the archive gains coverage with iterations
  * reproducible   — a fixed seed reproduces the archive exactly
Run:  python3 -m pytest -q     (or)     python3 tests/test_validation.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from personal_cambrian.genome import Genotype
from personal_cambrian.phenotype import TrainingProgram
from personal_cambrian.agent import Agent
from personal_cambrian.biomes import evaluate_all
from personal_cambrian.fitness import Baseline, evaluate_agent
from personal_cambrian.evolve import EvolveConfig, run, agent_zero


def _agent(**genes) -> Agent:
    g = Genotype()
    for k, v in genes.items():
        g[k] = v
    return Agent(g, TrainingProgram(strength=0.3, speed=0.3, power=0.2, endurance=0.2))


def test_pcsa_increases_strength():
    lo = evaluate_all(_agent(pcsa_capacity=420))["iron_zone"].score
    hi = evaluate_all(_agent(pcsa_capacity=700))["iron_zone"].score
    assert hi > lo, (lo, hi)


def test_fast_fiber_helps_sprint_not_endurance():
    slow = _agent(fast_fiber_frac=0.30)
    fast = _agent(fast_fiber_frac=0.70)
    s = evaluate_all(slow); f = evaluate_all(fast)
    assert f["track"].score > s["track"].score          # faster top speed
    assert f["endurance"].score < s["endurance"].score  # worse economy


def test_vo2max_drives_endurance_only():
    base = evaluate_all(_agent(vo2max_ceiling=45))
    high = evaluate_all(_agent(vo2max_ceiling=70))
    assert high["endurance"].score > base["endurance"].score
    # sprint top speed must be (near-)independent of VO2max ceiling
    assert abs(high["track"].score - base["track"].score) < 1e-6


def test_tendon_elastic_helps_jump():
    stiff = evaluate_all(_agent(tendon_elastic_ret=0.55, tendon_stiffness=320))
    soft = evaluate_all(_agent(tendon_elastic_ret=0.15, tendon_stiffness=120))
    assert stiff["ballistics"].score > soft["ballistics"].score


def test_budget_overflow_penalizes_overbuilt_body():
    z = agent_zero()
    base = Baseline.from_agent(z)
    monster = Agent(Genotype({"pcsa_capacity": 2000, "heat_dissipation": 45,
                              "recovery_capacity": 0.5, "fast_fiber_frac": 0.95}),
                    TrainingProgram(strength=1.0, volume=1.0, intensity=1.0))
    ev = evaluate_agent(monster, base)
    assert ev.budget_overflow > 0.0
    # raw performance may look high, but fitness must be dragged down by costs
    assert ev.fitness < ev.performance


def test_archive_fills_and_is_reproducible():
    cfg = EvolveConfig(iterations=400, init_population=60, resolution=12, seed=7)
    a1, _ = run(cfg, verbose=False)
    a2, _ = run(cfg, verbose=False)
    assert a1.coverage > 0.05
    assert len(a1.grid) == len(a2.grid)               # determinism
    assert abs(a1.qd_score - a2.qd_score) < 1e-9
    assert a1.best().fitness >= agent_zero_fitness()  # evolution beats baseline


def agent_zero_fitness() -> float:
    z = agent_zero()
    return evaluate_agent(z, Baseline.from_agent(z)).fitness


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    passed = 0
    for fn in fns:
        try:
            fn(); passed += 1
            print(f"PASS  {fn.__name__}")
        except AssertionError as e:
            print(f"FAIL  {fn.__name__}: {e}")
    print(f"\n{passed}/{len(fns)} passed")
    sys.exit(0 if passed == len(fns) else 1)
