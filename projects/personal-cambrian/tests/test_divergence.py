"""Stage 7.5 acceptance: lineage-divergence metrics.

Two lineages bred from the SAME seed under OPPOSITE selective pressures (more vs
fewer limbs -- a stand-in for two niches) pull apart in morphology space: the
mean inter-niche distance starts at 0 and rises over generations. Plus the metric
primitives behave (centroids, separation, per-descriptor spread) and the plots
render.

Runs:  python3 tests/test_divergence.py   (requires mujoco for develop)
"""
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from personal_cambrian.seeds import quadruped, agent_zero
from personal_cambrian.encoding.mutate import mutate, MutationSchedule
from personal_cambrian.evo import morphology_descriptors, pre_sim_gate
from personal_cambrian.evo.divergence import (
    niche_centroid, centroid_distance, inter_niche_separation,
    divergence_trajectory, per_descriptor_spread, plot_divergence,
    plot_niche_space,
)
from personal_cambrian.evo.metrics import has_matplotlib


def _evolve_toward(seed, key, sign, gens, rng):
    """Greedy hill-climb selecting the descriptor `key` toward +/- (a niche proxy)."""
    sched = MutationSchedule(macro_rate=0.8)
    g, snaps = seed, [[seed]]
    for _ in range(gens):
        best, best_s = g, sign * morphology_descriptors(g)[key]
        for _ in range(6):
            child, _ = mutate(g, rng, generation=0, schedule=sched)
            if not pre_sim_gate(child)[0]:
                continue
            s = sign * morphology_descriptors(child)[key]
            if s > best_s:
                best, best_s = child, s
        g = best
        snaps.append([g])
    return snaps


# --- metric primitives ------------------------------------------------------
def test_centroid_and_distance_basics():
    z = agent_zero()
    from personal_cambrian.evo import descriptor_vector
    assert np.allclose(niche_centroid([z, z]), descriptor_vector(z))
    assert centroid_distance(niche_centroid([z]), niche_centroid([z])) == 0.0
    assert niche_centroid([]).shape == (6,)


def test_separation_zero_for_same_population_large_for_different():
    z, q = agent_zero(), quadruped()
    _, same = inter_niche_separation({"a": [z, z], "b": [z]})
    _, diff = inter_niche_separation({"a": [z], "b": [q]})
    assert same == 0.0
    assert diff > 0.3


# --- the headline: niches separate over generations ------------------------
def test_opposite_pressures_diverge_over_generations():
    more = _evolve_toward(quadruped(), "limb_count", +1, 5, np.random.default_rng(1))
    fewer = _evolve_toward(quadruped(), "limb_count", -1, 5, np.random.default_rng(2))
    gens = [{"more": more[i], "fewer": fewer[i]} for i in range(len(more))]
    traj = divergence_trajectory(gens)
    assert traj[0] == 0.0                          # same seed -> no separation
    assert traj[-1] > 0.05                         # they have pulled apart
    assert traj[-1] >= traj[2]                     # separation does not shrink back
    # the niches differ most on the descriptor they were selected on
    spread = per_descriptor_spread(gens[-1])
    assert spread["limb_count"] == max(spread.values())


# --- plots render -----------------------------------------------------------
def test_plots_render():
    if not has_matplotlib():
        print("  (skipped: no matplotlib)")
        return
    gens = [{"a": [quadruped()], "b": [agent_zero()]}]
    traj = [0.0, 0.1, 0.2]
    with tempfile.TemporaryDirectory() as d:
        p1 = plot_divergence(traj, os.path.join(d, "div.png"))
        p2 = plot_niche_space(gens[0], os.path.join(d, "space.png"))
        assert os.path.getsize(p1) > 1000 and os.path.getsize(p2) > 1000


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
