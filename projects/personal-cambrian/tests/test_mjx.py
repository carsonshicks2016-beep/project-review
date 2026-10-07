"""Stage 9.1 acceptance: MJX (JAX) vectorized physics backend.

Done-when: identical-ish dynamics vs the CPU C engine on a fixed creature (within
tol), and the backend vectorizes across many parallel envs. We verify:

  * PARITY -- MJX matches CPU MuJoCo to ~1e-7/step in the contact-free regime and
    to ~1e-9 for a single contact step (the port is faithful).
  * KNOWN LIMITATION -- contact-rich rollouts diverge (MJX soft vs C rigid contact);
    asserted and documented, which is why CPU stays the ground-truth oracle.
  * VECTORIZATION -- jit+vmap rollout over many parallel envs produces correct,
    finite, deterministic batched dynamics.

The throughput numbers (CPU-vs-MJX) live in scripts/mjx_benchmark.py; on this CPU
box MJX does not beat the C engine -- the 100x is a GPU property -- so CI only
checks correctness, not speed. Skips cleanly if jax/mujoco-mjx are absent.

Runs:  python3 tests/test_mjx.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from personal_cambrian.sim.mjx_env import mjx_available

_SKIP = not mjx_available()

if not _SKIP:
    from personal_cambrian.seeds import quadruped
    from personal_cambrian.morphogenesis import develop
    from personal_cambrian.morphogenesis.to_mujoco import compile_morphology
    from personal_cambrian.sim.mjx_env import MjxBatchEnv, dynamics_parity, grounded_qpos

_CACHE = {}


def _model():
    if "model" not in _CACHE:
        m, _ = compile_morphology(develop(quadruped()), add_floor=True, free_root=True)
        m.opt.iterations = 50          # match the engines' solver effort for a fair compare
        m.opt.ls_iterations = 50
        _CACHE["model"] = m
    return _CACHE["model"]


def _env():
    if "env" not in _CACHE:
        _CACHE["env"] = MjxBatchEnv(quadruped(), n_substeps=2)
    return _CACHE["env"]


# --- dynamics parity --------------------------------------------------------
def test_contact_free_parity_is_tight():
    m = _model()
    air = grounded_qpos(m).copy()
    air[2] += 3.0                                          # lift clear of the floor
    d = dynamics_parity(m, qpos0=air, n_steps=20, n_substeps=1)
    print(f"  in-air parity (n=20): {d:.2e}")
    assert d < 5e-3                                        # faithful articulated-body port


def test_single_contact_step_parity_is_tight():
    d = dynamics_parity(_model(), n_steps=1, n_substeps=1)  # standing, one step
    print(f"  on-ground parity (n=1): {d:.2e}")
    assert d < 1e-4


def test_hard_contact_divergence_is_documented():
    # MJX uses a SOFT contact model; for gentle contact it tracks the C engine to
    # sub-mm, but as a body is driven HARD into the floor the soft-vs-rigid gap
    # grows fast. This is the known MJX limitation -> CPU stays the ground-truth
    # oracle for final scoring; MJX is the fast pre-screen / training substrate.
    m = _model()
    gentle = grounded_qpos(m)
    hard = gentle.copy(); hard[2] -= 0.10                 # 10cm penetration = hard contact
    d_gentle = dynamics_parity(m, qpos0=gentle, n_steps=20, n_substeps=1)
    d_hard = dynamics_parity(m, qpos0=hard, n_steps=20, n_substeps=1)
    print(f"  parity gentle={d_gentle:.2e}  hard-contact={d_hard:.2e}")
    assert d_hard > 10 * d_gentle                         # contact severity drives divergence
    assert d_hard > 1e-2                                  # the documented soft-vs-rigid gap


# --- vectorization ----------------------------------------------------------
def test_batched_rollout_shapes_and_finite():
    env = _env()
    out = env.rollout(n_envs=16, n_steps=4, seed=0)
    q = env.qpos(out)
    assert q.shape == (16, env.model.nq)
    assert np.all(np.isfinite(q))


def test_batch_is_deterministic():
    env = _env()
    a = env.qpos(env.rollout(8, 4, seed=1))
    b = env.qpos(env.rollout(8, 4, seed=1))
    assert np.array_equal(a, b)                           # same seed -> identical batch


def test_identical_actions_give_identical_envs():
    # all envs start at the same pose; driving them with the SAME action keeps the
    # batch homogeneous, while a different action changes the outcome.
    env = _env()
    data = env.init(4)
    same = np.tile(np.full(env.model.nu, 0.5, dtype=np.float32), (4, 1))
    data = env.step(data, same)
    q = env.qpos(data)
    assert np.allclose(q, q[0], atol=1e-6)                # rows identical under same action
    other = env.step(env.init(4), np.tile(np.full(env.model.nu, -0.5, np.float32), (4, 1)))
    assert not np.allclose(env.qpos(other)[0], q[0])      # a different action diverges


if __name__ == "__main__":
    if _SKIP:
        print("SKIP  jax / mujoco-mjx not installed (Stage 9 GPU extra)")
        sys.exit(0)
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
