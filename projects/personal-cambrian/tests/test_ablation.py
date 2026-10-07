"""Stage 3.5 acceptance: the fine-tune-budget fairness tooling is correct.

The knee detector finds where a saturating score plateaus; the budget
recommendation guarantees >=90% of bodies are past their knee (so a good body is
not misjudged for an undertrained brain); and the curve machinery runs end to end.

Runs:  python3 tests/test_ablation.py   (requires mujoco + gymnasium + torch)
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from personal_cambrian.seeds import quadruped
from personal_cambrian.sim import CreatureEnv, LocomotionTask
from personal_cambrian.control import (
    PPOConfig, policy_for_env, find_knee, recommend_budget, budget_curve,
    warmstart_curve, scratch_make_agent, warm_started_make_agent,
)


def test_find_knee_on_saturating_curve():
    steps = np.arange(0, 120_000, 4096)
    scores = 40.0 * (1.0 - np.exp(-steps / 20_000.0))     # saturates at ~40
    knee = find_knee(steps, scores, frac=0.9)
    # reaches 0.9*40 around 20000*ln(10) ~ 46k
    assert 30_000 < knee < 70_000


def test_find_knee_flat_curve_is_early():
    steps = np.arange(0, 50_000, 5_000)
    scores = np.full(steps.shape, 30.0)
    assert find_knee(steps, scores, frac=0.9) == steps[0]   # already plateaued


def test_recommend_budget_guarantees_coverage():
    knees = [10_000, 12_000, 15_000, 20_000, 90_000]        # one slow outlier
    b = recommend_budget(knees, coverage=0.9)
    frac_past = float(np.mean(np.asarray(knees) <= b))
    assert frac_past >= 0.9                                 # >=90% past their knee
    assert b == 90_000                                      # 'higher' quantile covers all


def test_recommend_budget_robust_to_one_outlier_with_many_bodies():
    knees = list(np.full(19, 10_000)) + [200_000]          # 1/20 is slow
    b = recommend_budget(knees, coverage=0.9)
    assert b == 10_000                                     # ignores the 5% outlier
    assert float(np.mean(np.asarray(knees) <= b)) >= 0.9


def test_warmstart_curve_runs_and_is_finite():
    parent = policy_for_env(CreatureEnv(quadruped(),
                            task=LocomotionTask(max_steps=40), obs_mode="structured"))
    cme = lambda s: CreatureEnv(quadruped(), task=LocomotionTask(max_steps=40),
                                obs_mode="structured")
    steps, scores = warmstart_curve(cme, parent, max_steps=256, n_envs=2)
    assert len(steps) == len(scores) >= 1
    assert np.all(np.isfinite(scores))
    assert find_knee(steps, scores) is not None


def test_scratch_make_agent_builds_for_child():
    cme = lambda s: CreatureEnv(quadruped(), task=LocomotionTask(max_steps=40),
                                obs_mode="structured")
    steps, scores = budget_curve(cme, scratch_make_agent(cme), max_steps=256, n_envs=2)
    assert len(steps) >= 1 and np.all(np.isfinite(scores))


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
