"""Stage 8.1 acceptance: behavior characterization (BC).

The "done-when" is *similar behaviors map near each other*. We make that concrete
with scripted open-loop controllers (no training needed, fully deterministic):

  * the SAME gait under different reset noise  -> a SIMILAR behavior
  * a much faster gait, and a passive ragdoll  -> DIFFERENT behaviors

and assert the same-gait pair is closer in BC space than either different one.
We also check the structural contract: the BC is fixed-length, finite, and
morphology-independent (a 4-limb and a 9-limb creature yield the same DIM), and
that distances behave like a metric (zero on self, symmetric).

Pure helpers (`bc_distance`, `bc_matrix`, `nearest`, `novelty`) are also tested
on synthetic vectors. The rollout tests need mujoco + gymnasium.

Runs:  python3 tests/test_behavior.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from personal_cambrian.evo.behavior import (
    BehaviorChar, characterize, bc_distance, bc_matrix, nearest, novelty,
    LABELS, DIM, SCALES,
)


# --- scripted controllers (deterministic, morphology-agnostic) -------------
class Sine:
    """Open-loop per-actuator sinusoid -- a 'gait' parameterized by frequency.
    A per-actuator phase offset makes a traveling wave so the body actually moves."""
    def __init__(self, nu, dt, freq, amp=0.8, phase=0.0, wave=0.7):
        self.nu, self.dt, self.freq = nu, dt, freq
        self.amp, self.phase, self.wave = amp, phase, wave
        self.t = 0

    def act(self, obs, deterministic=True):
        offs = self.wave * np.arange(self.nu)
        a = self.amp * np.sin(2 * np.pi * self.freq * self.t * self.dt + self.phase + offs)
        self.t += 1
        return a.astype(np.float32)


class Passive:
    """Does nothing -- a ragdoll. Categorically different from any active gait."""
    def __init__(self, nu):
        self.nu = nu

    def act(self, obs, deterministic=True):
        return np.zeros(self.nu, dtype=np.float32)


def _env(seed_name="quadruped"):
    from personal_cambrian import seeds
    from personal_cambrian.sim import CreatureEnv
    return CreatureEnv(getattr(seeds, seed_name)(), obs_mode="flat")


# --- structural contract ----------------------------------------------------
def test_bc_is_fixed_length_and_finite():
    env = _env()
    nu = env.action_space.shape[0]
    bc = characterize(env, Sine(nu, env.control_dt, freq=1.5), eval_seed=0, max_steps=120)
    env.close()
    assert isinstance(bc, BehaviorChar)
    assert bc.vector.shape == (DIM,)
    assert np.all(np.isfinite(bc.vector))
    assert set(bc.raw) == set(LABELS)


def test_bc_is_morphology_independent():
    # different creatures (different limb/actuator counts) -> SAME BC dimension
    e1, e2 = _env("quadruped"), _env("agent_zero")
    assert e1.action_space.shape[0] != e2.action_space.shape[0]
    b1 = characterize(e1, Sine(e1.action_space.shape[0], e1.control_dt, 1.5), max_steps=80)
    b2 = characterize(e2, Sine(e2.action_space.shape[0], e2.control_dt, 1.5), max_steps=80)
    e1.close(); e2.close()
    assert b1.vector.shape == b2.vector.shape == (DIM,)


def test_bc_is_deterministic():
    # same gait, same seed, twice -> identical BC (reproducible characterization)
    env = _env()
    nu, dt = env.action_space.shape[0], env.control_dt
    a = characterize(env, Sine(nu, dt, 1.5), eval_seed=0, max_steps=120)
    b = characterize(env, Sine(nu, dt, 1.5), eval_seed=0, max_steps=120)
    env.close()
    assert np.allclose(a.vector, b.vector)
    assert bc_distance(a, b) == 0.0


# --- the done-when: similar behaviors map near each other -------------------
def test_similar_behaviors_are_nearer_than_different_ones():
    env = _env()
    nu, dt = env.action_space.shape[0], env.control_dt
    walk_a = characterize(env, Sine(nu, dt, 1.5), eval_seed=0, max_steps=200)
    walk_b = characterize(env, Sine(nu, dt, 1.5), eval_seed=7, max_steps=200)   # same gait, new noise
    fast = characterize(env, Sine(nu, dt, 5.0), eval_seed=0, max_steps=200)     # different gait
    passive = characterize(env, Passive(nu), eval_seed=0, max_steps=200)        # different behavior
    env.close()

    d_same = bc_distance(walk_a, walk_b)
    d_fast = bc_distance(walk_a, fast)
    d_pass = bc_distance(walk_a, passive)
    print(f"  d(same-gait)={d_same:.3f}  d(fast)={d_fast:.3f}  d(passive)={d_pass:.3f}")
    assert d_same < d_fast, f"same-gait {d_same:.3f} not < fast-gait {d_fast:.3f}"
    assert d_same < d_pass, f"same-gait {d_same:.3f} not < passive {d_pass:.3f}"
    # and the nearest neighbor of walk_a among the candidates is its own gait
    assert nearest(walk_a, [walk_b, fast, passive]) == 0


# --- pure distance helpers (synthetic) -------------------------------------
def test_bc_distance_is_metric_like():
    a = np.zeros(DIM); b = np.ones(DIM)
    assert bc_distance(a, a) == 0.0
    assert bc_distance(a, b) == bc_distance(b, a) > 0.0
    # one full SCALE step on every axis -> RMS distance of exactly 1.0
    assert abs(bc_distance(a, b) - 1.0) < 1e-12


def test_bc_matrix_shape_and_symmetry():
    vs = [np.zeros(DIM), np.ones(DIM), np.full(DIM, 2.0)]
    M = bc_matrix(vs)
    assert M.shape == (3, 3)
    assert np.allclose(M, M.T)
    assert np.allclose(np.diag(M), 0.0)


def test_nearest_and_novelty():
    q = np.zeros(DIM)
    archive = [np.full(DIM, 3.0), np.full(DIM, 0.1), np.full(DIM, 5.0)]
    assert nearest(q, archive) == 1                      # 0.1 is closest
    assert novelty(q, [], k=3) == float("inf")           # empty archive
    # novelty = mean of k nearest; here the two nearest are 0.1 and 3.0
    expected = np.mean(sorted(bc_distance(q, a) for a in archive)[:2])
    assert abs(novelty(q, archive, k=2) - expected) < 1e-12


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
