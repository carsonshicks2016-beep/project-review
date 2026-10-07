"""Stage 7.3 acceptance: speciation / niching in selection.

Two checks:
  (1) clustering -- genomes group into species by morphological distance: bodies
      of the same plan share a species, a divergent plan founds its own.
  (2) protection -- species-aware emission gives a rare (singleton) species the
      same reproduction share as a crowded one, and across a multi-seed QD run
      the species count never collapses to 1.

Runs:  python3 tests/test_speciation.py   (clustering: mujoco; run: + gymnasium + torch)
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from personal_cambrian.seeds import quadruped, agent_zero
from personal_cambrian.encoding.mutate import micro, mutate, MutationSchedule
from personal_cambrian.evo import (
    speciate, count_species, group_by_species, sample_species_aware,
    run_qd, QDConfig, Elite,
)

THRESH = 0.15


def _twin(g, seed=0):
    return micro(g, np.random.default_rng(seed), 0.1)


# --- (1) clustering ---------------------------------------------------------
def test_same_plan_clusters_divergent_plan_splits():
    z, q = agent_zero(), quadruped()
    labels, reps = speciate([z, _twin(z, 1), q, _twin(q, 2)], THRESH)
    assert len(reps) == 2                         # humans together, quads together
    assert labels[0] == labels[1]                 # the two humans share a species
    assert labels[2] == labels[3]                 # the two quads share a species
    assert labels[0] != labels[2]                 # human != quad


def test_count_species_and_divergent_descendant():
    z, q = agent_zero(), quadruped()
    assert count_species([z, _twin(z)], THRESH) == 1
    assert count_species([z, q], THRESH) == 2
    # drive a quad far from its origin -> it founds a third species
    rng = np.random.default_rng(3)
    g = q
    for i in range(8):
        g, _ = mutate(g, rng, generation=i, schedule=MutationSchedule(macro_rate=0.8))
    assert count_species([z, q, g], THRESH) >= 3


def test_speciation_is_deterministic():
    genomes = [agent_zero(), _twin(agent_zero()), quadruped()]
    assert speciate(genomes, THRESH)[0] == speciate(genomes, THRESH)[0]


# --- (2) protection: rare species gets equal reproduction ------------------
def _elite(g):
    return Elite(genome=g, fitness=0.0, descriptors={}, meta={})


def test_species_aware_sampling_protects_rare_species():
    z = agent_zero()
    crowded = [_elite(_twin(z, i)) for i in range(5)]   # 5 near-identical humans
    rare = [_elite(quadruped())]                         # 1 lone quadruped
    elites = crowded + rare
    assert len(group_by_species(elites, THRESH)) == 2
    rng = np.random.default_rng(0)
    picks = [sample_species_aware(elites, rng, THRESH) for _ in range(400)]
    quad_share = sum(1 for e in picks if e in rare) / len(picks)
    # equal-per-species would give ~0.5; cell-uniform would give ~1/6 = 0.17
    assert quad_share > 0.35


# --- (2) protection: species persist across a run --------------------------
def test_multi_seed_run_keeps_at_least_two_species():
    cfg = QDConfig(axes=["aspect", "limb_count"], bins=8, iterations=12,
                   seed_steps=256, fine_tune_steps=256, n_envs=2, n_steps=64,
                   ep_steps=40, hidden=32, macro_rate=0.6, seed=0,
                   speciation=True, species_threshold=THRESH)
    logs = []
    run_qd([quadruped(), agent_zero()], cfg, log_fn=logs.append)
    species_series = [r["species"] for r in logs]
    assert all("species" in r for r in logs)
    assert min(species_series) >= 2               # never collapses to one plan
    assert species_series[-1] >= 2


def test_run_species_series_is_reproducible():
    cfg = QDConfig(axes=["aspect", "limb_count"], bins=8, iterations=10,
                   seed_steps=256, fine_tune_steps=256, n_envs=2, n_steps=64,
                   ep_steps=40, hidden=32, macro_rate=0.6, seed=0,
                   speciation=True, species_threshold=THRESH)
    a, b = [], []
    run_qd([quadruped(), agent_zero()], cfg, log_fn=a.append)
    run_qd([quadruped(), agent_zero()], cfg, log_fn=b.append)
    assert [r["species"] for r in a] == [r["species"] for r in b]


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
