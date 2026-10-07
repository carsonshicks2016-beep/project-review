"""Stage 5.1 acceptance: behavior + morphology descriptors are deterministic and
span a meaningful range across creatures.

Runs:  python3 tests/test_descriptors.py   (requires mujoco + gymnasium + torch)
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from personal_cambrian.seeds import agent_zero, quadruped
from personal_cambrian.encoding import micro
from personal_cambrian.encoding.mutate import add_part, duplicate_subtree
from personal_cambrian.evo import (
    morphology_descriptors, distance_from, compute, normalize, normalized,
    DESCRIPTOR_BOUNDS, behavior_descriptors,
)
from personal_cambrian.sim import CreatureEnv, LocomotionTask
from personal_cambrian.control import ActorCritic


def _sample_genomes():
    rng = np.random.default_rng(0)
    g = [agent_zero(), quadruped()]
    g.append(add_part(quadruped(), rng)[0])
    g.append(duplicate_subtree(quadruped(), rng)[0])
    g.append(micro(agent_zero(), rng, sigma=0.3))
    return g


def test_morphology_descriptors_are_deterministic():
    g = quadruped()
    assert morphology_descriptors(g) == morphology_descriptors(g)


def test_descriptors_span_a_range():
    rows = [morphology_descriptors(g) for g in _sample_genomes()]
    for key in ("limb_count", "muscle_count", "mass", "aspect"):
        vals = [r[key] for r in rows]
        assert max(vals) - min(vals) > 0, key            # genuinely varies


def test_biped_vs_quadruped_differ():
    b = morphology_descriptors(agent_zero())
    q = morphology_descriptors(quadruped())
    assert b["limb_count"] != q["limb_count"]
    assert b["mass"] != q["mass"]


def test_morph_distance():
    rng = np.random.default_rng(1)
    seed = quadruped()
    assert distance_from(seed, seed) == 0.0
    child = add_part(seed, rng)[0]
    assert distance_from(child, seed) >= 1.0


def test_normalization_in_unit_range():
    for g in _sample_genomes():
        for k, v in morphology_descriptors(g).items():
            assert 0.0 <= normalize(k, v) <= 1.0
    # out-of-range clamps
    assert normalize("limb_count", 1e6) == 1.0
    assert normalize("mass", -50.0) == 0.0


def test_compute_merges_morphology_and_distance():
    d = compute(quadruped(), seed_genome=agent_zero())
    assert "limb_count" in d and "morph_distance" in d
    assert set(normalized(d)).issubset(set(DESCRIPTOR_BOUNDS))


def test_behavior_speed_is_deterministic():
    env = CreatureEnv(quadruped(), task=LocomotionTask(max_steps=60))
    pi = ActorCritic(env.observation_space.shape[0], env.action_space.shape[0], 32)
    a = behavior_descriptors(env, pi, eval_seed=0, max_steps=60)
    b = behavior_descriptors(env, pi, eval_seed=0, max_steps=60)
    assert a["speed"] == b["speed"] and np.isfinite(a["speed"])
    env.close()


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
