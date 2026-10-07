"""Open-loop CPG-style gaits (ROADMAP Stage 2.7).

A central-pattern-generator: each actuator gets a sinusoid
    ctrl_i(t) = amp * sin(2*pi*freq*t + phase_i)
with per-actuator phases. This is a *hand-coded* controller (no learning) used to
smoke-test the evaluation stack before Stage 3 RL: it proves the reward channel
is not degenerate — that control input can produce forward motion the task
rewards. (Stable open-loop walking is easy for a quadruped and effectively
impossible for a biped without feedback, which is itself a useful finding.)
"""
from __future__ import annotations

import numpy as np


def ramp_phases(nu: int) -> np.ndarray:
    return np.linspace(0.0, 2.0 * np.pi, nu, endpoint=False)


def alt_phases(nu: int) -> np.ndarray:
    return np.pi * (np.arange(nu) % 2)


def cpg(t: float, phases: np.ndarray, freq: float, amp: float) -> np.ndarray:
    return amp * np.sin(2.0 * np.pi * freq * t + phases)


def open_loop_rollout(env, *, freq: float = 1.5, amp: float = 1.0,
                      phase: str = "ramp", steps: int = 400, seed: int = 0) -> dict:
    """Drive `env` with a CPG gait; return forward-progress diagnostics."""
    env.reset(seed=seed)
    nu = env.action_space.shape[0]
    phases = ramp_phases(nu) if phase == "ramp" else alt_phases(nu)
    x0 = float(env.data.qpos[env.task.forward_axis])
    t, survived, fwd_reward = 0.0, 0, 0.0
    for _ in range(steps):
        a = cpg(t, phases, freq, amp).astype(np.float32)
        _, _, term, trunc, info = env.step(a)
        fwd_reward += info["reward_terms"]["forward"]
        t += env.control_dt
        survived += 1
        if term or trunc:
            break
    net_x = float(env.data.qpos[env.task.forward_axis]) - x0
    return {"net_x": net_x, "forward_reward": fwd_reward, "survived": survived}


def zero_action_rollout(env, *, steps: int = 400, seed: int = 0) -> dict:
    env.reset(seed=seed)
    x0 = float(env.data.qpos[env.task.forward_axis])
    z = np.zeros(env.action_space.shape, np.float32)
    survived = 0
    for _ in range(steps):
        _, _, term, trunc, _ = env.step(z)
        survived += 1
        if term or trunc:
            break
    return {"net_x": float(env.data.qpos[env.task.forward_axis]) - x0,
            "survived": survived}
