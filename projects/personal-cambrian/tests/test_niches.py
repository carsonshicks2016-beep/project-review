"""Stage 7.1 acceptance: the niche gauntlet.

For each niche we check the three things the ROADMAP asks:
  (1) it RUNS   -- the env builds and steps with finite reward,
  (2) it rewards SPECIALIZATION -- the reward responds to the behaviour the niche
      is meant to select for (tested on the pure reward function / mechanics),
  (3) it ESCALATES -- escalate() raises difficulty and the concrete pressure
      (load / slope / gravity / energy budget) follows.
Plus every niche is a declarative config that round-trips through to_dict/from_dict.

Runs:  python3 tests/test_niches.py   (requires mujoco + gymnasium)
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import mujoco

from personal_cambrian.seeds import quadruped
from personal_cambrian.morphogenesis import develop
from personal_cambrian.morphogenesis.to_mujoco import compile_morphology
from personal_cambrian.sim import (
    CreatureEnv, NICHES, task_from_dict,
    TrackTask, IronZoneTask, BallisticsTask, EnduranceTask, ChaosGridTask, TerrainTask,
)

G = quadruped()


def _fresh_model():
    return compile_morphology(develop(G), add_floor=True)[0]


# --- (1) every niche runs + round-trips ------------------------------------
def test_all_niches_run_and_serialize():
    assert set(NICHES) == {"locomotion", "track", "iron_zone", "ballistics",
                           "endurance", "chaos_grid", "terrain"}
    for name, cls in NICHES.items():
        task = cls(max_steps=40)
        env = CreatureEnv(G, task=task, obs_mode="structured")
        env.reset(seed=0)
        for _ in range(15):
            _, r, term, trunc, info = env.step(env.action_space.sample())
            assert np.isfinite(r), name
            assert info["reward_terms"], name
            if term or trunc:
                break
        env.close()
        # declarative config survives a round-trip (episode state excluded)
        back = task_from_dict(task.to_dict())
        assert type(back) is cls and back.to_dict() == task.to_dict()


# --- (2) each niche rewards its specialization -----------------------------
def test_track_rewards_speed_and_escalates_headwind():
    t = TrackTask()
    z = np.zeros(4)
    slow, _ = t.reward(forward_vel=0.2, action=z, prev_action=z, dt=0.05)
    fast, _ = t.reward(forward_vel=3.0, action=z, prev_action=z, dt=0.05)
    assert fast > slow
    # headwind grows with difficulty (applied via perturb)
    env = CreatureEnv(G, task=TrackTask(difficulty=1.0))
    env.reset(seed=0)
    env.task.perturb(env.model, env.data, env.np_random, 0)
    assert env.data.xfrc_applied[1, 0] < 0          # opposing +x
    env.close()


def test_iron_zone_rewards_force_and_punishes_drift():
    t = IronZoneTask()
    strong, _ = t.reward(actuator_force=np.full(4, 300.0), displacement=0.0,
                         action=np.zeros(4), dt=0.05)
    weak, _ = t.reward(actuator_force=np.full(4, 10.0), displacement=0.0,
                       action=np.zeros(4), dt=0.05)
    dragged, _ = t.reward(actuator_force=np.full(4, 300.0), displacement=1.0,
                          action=np.zeros(4), dt=0.05)
    assert strong > weak                            # more force -> more reward
    assert strong > dragged                         # being dragged hurts


def test_ballistics_rewards_upward_launch():
    t = BallisticsTask()
    grounded, _ = t.reward(com_vz=0.0, com_z=0.5, stand_height=0.5,
                           action=np.zeros(4), dt=0.05)
    launching, _ = t.reward(com_vz=3.0, com_z=0.9, stand_height=0.5,
                            action=np.zeros(4), dt=0.05)
    assert launching > grounded
    # downward motion earns no launch credit
    falling, _ = t.reward(com_vz=-3.0, com_z=0.5, stand_height=0.5,
                          action=np.zeros(4), dt=0.05)
    assert falling <= grounded + 1e-9


def test_endurance_depletes_and_favors_efficiency():
    # exhaust the reserve: once empty, forward progress earns nothing
    t = EnduranceTask(energy_budget=1.0)
    t.reset_episode(None)
    a = np.full(4, 1.0)
    rewards = [t.reward(forward_vel=1.0, action=a, dt=0.1)[0] for _ in range(50)]
    assert rewards[0] > 0.0
    assert rewards[-1] == 0.0                        # fatigued to zero
    # an efficient (low-action) gait keeps energy longer than a profligate one
    eff = EnduranceTask(energy_budget=5.0); eff.reset_episode(None)
    pro = EnduranceTask(energy_budget=5.0); pro.reset_episode(None)
    for _ in range(20):
        eff.reward(forward_vel=1.0, action=np.full(4, 0.1), dt=0.1)
        pro.reward(forward_vel=1.0, action=np.full(4, 1.0), dt=0.1)
    assert eff._energy > pro._energy


def test_chaos_grid_rewards_stability_and_pushes():
    t = ChaosGridTask()
    steady, _ = t.reward(displacement=0.0, up_proj=1.0, action=np.zeros(4), dt=0.05)
    shoved, _ = t.reward(displacement=1.0, up_proj=1.0, action=np.zeros(4), dt=0.05)
    tipped, _ = t.reward(displacement=0.0, up_proj=0.2, action=np.zeros(4), dt=0.05)
    assert steady > shoved and steady > tipped
    # a push lands on the schedule
    env = CreatureEnv(G, task=ChaosGridTask(difficulty=1.0, push_period=1))
    env.reset(seed=0)
    env.task.perturb(env.model, env.data, env.np_random, 0)
    assert np.linalg.norm(env.data.xfrc_applied[1, :2]) > 0
    env.close()


def test_terrain_tilts_gravity_with_slope():
    flat = _fresh_model(); TerrainTask(difficulty=0.0).configure_model(flat)
    steep = _fresh_model(); TerrainTask(difficulty=1.0).configure_model(steep)
    assert abs(flat.opt.gravity[0]) < 1e-9          # no slope at difficulty 0
    assert steep.opt.gravity[0] < -1.0              # gravity pulls downhill (-x)
    assert abs(np.linalg.norm(steep.opt.gravity) - 9.81) < 1e-3   # |g| preserved (a rotation)


# --- (3) escalation raises the concrete pressure ---------------------------
def test_escalate_increases_difficulty_and_pressure():
    # iron zone: load grows
    iz = IronZoneTask(); d0 = iz.difficulty
    iz.escalate(); assert iz.difficulty > d0
    lo = IronZoneTask(difficulty=0.0)
    hi = IronZoneTask(difficulty=1.0)
    assert (hi.load_base + hi.load_per_difficulty * hi.difficulty) > \
           (lo.load_base + lo.load_per_difficulty * lo.difficulty)
    # ballistics: gravity strengthens
    g_lo = _fresh_model(); BallisticsTask(difficulty=0.0).configure_model(g_lo)
    g_hi = _fresh_model(); BallisticsTask(difficulty=1.0).configure_model(g_hi)
    assert abs(g_hi.opt.gravity[2]) > abs(g_lo.opt.gravity[2])
    # endurance: budget shrinks
    assert EnduranceTask(difficulty=1.0)._budget() < EnduranceTask(difficulty=0.0)._budget()
    # terrain: slope steepens
    assert TerrainTask(difficulty=1.0)._slope() > TerrainTask(difficulty=0.2)._slope()


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
