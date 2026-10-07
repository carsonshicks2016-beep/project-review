"""Stage 8.3 acceptance: POET (paired open-ended trailblazer).

The "done-when" is a property of the POET loop: the active-environment set keeps
TURNING OVER while DIFFICULTY RISES, and agents TRANSFER between environments. We
prove it RL-free and deterministically on a toy domain (skill scalar vs target
difficulty) driven by the REAL `POET` orchestrator, then unit-test the operators
that make it work -- minimal-criterion (MCC) admission, environment novelty, and
the niche-mutation helpers. A tiny RL smoke checks the creature wiring. The slow
real curriculum lives in `scripts/poet_curriculum.py`.

Runs:  python3 tests/test_poet.py
"""
import os
import sys
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from personal_cambrian.evo.poet import (
    POET, POETConfig, mutate_niche, niche_descriptor, make_creature_poet,
)


# --- a fast deterministic toy: agent "skill" must catch up to env "target" --
@dataclass
class ToyEnv:
    target: float


@dataclass
class ToyAgent:
    skill: float


_LR, _MARGIN = 0.5, 0.5


def t_optimize(agent, env, steps):
    s = agent.skill
    for _ in range(steps):
        s += _LR * max(0.0, (env.target + _MARGIN) - s)   # climb toward solving the env
    return ToyAgent(s)


def t_evaluate(agent, env):
    return agent.skill - env.target                        # >0 = solved with headroom


def t_difficulty(env):
    return env.target


def t_descriptor(env):
    return np.array([env.target], dtype=float)


def t_mutate(env, rng, step=0.5):
    harder = step if rng.random() < 0.8 else -0.25 * step  # mostly ratchet up
    return ToyEnv(max(0.0, env.target + harder))


def _toy_poet(seed=0, **over):
    cfg = POETConfig(max_active=4, reproduce_every=2, transfer_every=2, n_children=4,
                     max_admit=1, optimize_steps=2, mc_low=-0.3, mc_high=0.3,
                     repro_threshold=0.3, novelty_k=3, **over)
    p = POET(cfg, optimize=t_optimize, evaluate=t_evaluate, env_mutate=t_mutate,
             env_descriptor=t_descriptor, env_difficulty=t_difficulty,
             rng=np.random.default_rng(seed))
    p.add_env(ToyEnv(0.0), ToyAgent(0.0))
    return p


# --- the done-when: rising difficulty + turnover + transfer ----------------
def test_poet_curriculum_rises_and_turns_over():
    p = _toy_poet(seed=0)
    p.run(30)
    start_d = p.history[0]["max_difficulty"]
    end_d = p.history[-1]["max_difficulty"]
    print(f"  frontier difficulty {start_d:.2f} -> {end_d:.2f}, "
          f"added={p.n_added} removed={p.n_removed} transfers={len(p.transfers)}")
    assert p.difficulty_rose and end_d > 1.0          # the curriculum got harder
    assert p.turned_over                              # envs were both created and retired
    assert p.n_removed > 0
    assert len(p.transfers) > 0                       # agents transferred between envs
    assert len(p.pairs) <= p.cfg.max_active           # the active set stays capped


def test_poet_is_reproducible():
    a, b = _toy_poet(seed=1), _toy_poet(seed=1)
    a.run(20); b.run(20)
    assert [r["max_difficulty"] for r in a.history] == [r["max_difficulty"] for r in b.history]
    assert a.n_added == b.n_added and len(a.transfers) == len(b.transfers)


# --- transfer in isolation --------------------------------------------------
def test_transfer_adopts_a_stronger_foreign_agent():
    p = POET(POETConfig(transfer_margin=1e-6), optimize=t_optimize, evaluate=t_evaluate,
             env_mutate=t_mutate, env_descriptor=t_descriptor, env_difficulty=t_difficulty,
             rng=np.random.default_rng(0))
    weak = p.add_env(ToyEnv(1.0), ToyAgent(0.4))      # under-skilled incumbent, score -0.6
    p.add_env(ToyEnv(3.0), ToyAgent(3.0))             # a strong trailblazer from a harder env
    assert weak.score < 0.0
    n = p._attempt_transfers()
    assert n >= 1
    assert weak.score > 0.0                           # adopted the stronger agent (3.0 - 1.0 = 2.0)
    assert any(ev["to"] == weak.id for ev in p.transfers)


# --- minimal-criterion coevolution (MCC) admission -------------------------
def _mcc_poet(mutate):
    cfg = POETConfig(reproduce_every=1, transfer_every=10_000, n_children=1, max_admit=5,
                     optimize_steps=2, mc_low=-0.3, mc_high=0.3, repro_threshold=-9.9,
                     max_active=99)
    p = POET(cfg, optimize=t_optimize, evaluate=t_evaluate, env_mutate=mutate,
             env_descriptor=t_descriptor, env_difficulty=t_difficulty,
             rng=np.random.default_rng(0))
    p.add_env(ToyEnv(0.0), ToyAgent(0.0))
    return p


def test_mcc_rejects_too_hard():
    p = _mcc_poet(lambda e, r: ToyEnv(e.target + 100.0))   # nobody can solve it
    p.step()
    assert len(p.pairs) == 1                                # child rejected, only the seed remains


def test_mcc_rejects_too_easy():
    p = _mcc_poet(lambda e, r: ToyEnv(0.0))                 # already solved after optimization
    p.step()
    assert len(p.pairs) == 1


def test_mcc_admits_in_band_child():
    # after 2 optimize steps the seed skill ~0.375; a target ~0.4 child sits in-band
    p = _mcc_poet(lambda e, r: ToyEnv(0.4))
    p.step()
    assert len(p.pairs) == 2                                # solvable-but-unsolved -> admitted


# --- environment novelty ----------------------------------------------------
def test_env_novelty_prefers_far_envs():
    p = _toy_poet(seed=0)
    p.env_archive = [t_descriptor(ToyEnv(x)) for x in (0.0, 0.1, 0.2)]
    near = p._env_novelty(ToyEnv(0.15))
    far = p._env_novelty(ToyEnv(9.0))
    assert far > near > 0.0


# --- niche-mutation helpers (real NicheTasks, no sim needed) ---------------
def test_mutate_niche_mostly_harder_and_serializable():
    from personal_cambrian.sim.tasks import TerrainTask, task_from_dict
    rng = np.random.default_rng(0)
    base = TerrainTask(difficulty=0.5)
    harder = sum(mutate_niche(base, rng).difficulty > 0.5 for _ in range(60))
    assert harder > 36                                      # mostly ratchets difficulty up
    child = mutate_niche(base, np.random.default_rng(2))
    assert type(child) is TerrainTask
    assert task_from_dict(child.to_dict()).difficulty == child.difficulty
    assert child.difficulty >= 0.0


def test_niche_descriptor_encodes_difficulty_and_type():
    from personal_cambrian.sim.tasks import TerrainTask, IronZoneTask
    assert niche_descriptor(TerrainTask(difficulty=0.5))[0] == 0.5
    assert niche_descriptor(TerrainTask(difficulty=1.5))[0] == 1.5
    same_type = niche_descriptor(TerrainTask())[1] == niche_descriptor(TerrainTask(difficulty=2))[1]
    assert same_type
    assert niche_descriptor(IronZoneTask())[1] != niche_descriptor(TerrainTask())[1]


# --- creature wiring smoke (needs mujoco + torch) --------------------------
def test_creature_poet_smoke():
    from personal_cambrian.seeds import quadruped
    from personal_cambrian.sim.tasks import TerrainTask
    cfg = POETConfig(max_active=3, reproduce_every=1, transfer_every=2, n_children=2,
                     max_admit=1, optimize_steps=1, mc_low=-1e9, mc_high=1e9,
                     repro_threshold=-1e9, novelty_k=2)
    poet = make_creature_poet(quadruped(), cfg, train_steps=128, ep_steps=30,
                              hidden=32, n_envs=2, n_steps=64, seed=0,
                              rng=np.random.default_rng(0))
    poet.add_env(TerrainTask(difficulty=0.0, max_steps=30), None)
    poet.run(3)
    assert len(poet.history) == 3
    assert poet.n_added >= 2                                # reproduction admitted new niches
    assert all("max_difficulty" in r and "n_active" in r for r in poet.history)


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
