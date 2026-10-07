"""ROLLOUT -> TRAJECTORY (ROADMAP Stage 10.2, bpy-free core).

Runs a MuJoCo rollout and records, per frame, every creature body's WORLD transform
(`data.xpos` + `data.xquat`, keyed by the morphology body id -- which is exactly the
MuJoCo body name and the Blender object name). Bundled with the rest-pose `ScenePlan`
into a JSON-serializable `Trajectory` that `blender_build.animate_trajectory` keyframes
and renders to a locomotion video.

A controller is optional: the default is an open-loop traveling-wave gait, so a seed
creature visibly MOVES without a trained policy -- enough to render a replay. Pass a
policy with `.act(obs, deterministic=True)` to replay a learned gait instead.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

from ..sim import CreatureEnv
from ..sim.tasks import NICHES
from .morphology_plan import plan_from_genome


def _sine_gait(nu: int, dt: float, *, freq=1.5, amp=0.8, wave=0.7):
    """Open-loop per-actuator traveling-wave controller (visible locomotion attempt)."""
    def ctrl(t, obs):
        phase = wave * np.arange(nu)
        return (amp * np.sin(2 * np.pi * freq * t * dt + phase)).astype(np.float32)
    return ctrl


@dataclass
class Trajectory:
    plan: dict                  # rest-pose ScenePlan.to_dict()
    frames: list                # [{body_id: [[x,y,z], [w,x,y,z]]}, ...]
    fps: int

    def to_dict(self) -> dict:
        return {"plan": self.plan, "frames": self.frames, "fps": int(self.fps)}

    @property
    def n_frames(self) -> int:
        return len(self.frames)

    def root_displacement(self) -> float:
        """Horizontal distance the root body travelled over the rollout."""
        root = next((g["name"] for g in self.plan["geoms"] if g["parent"] is None), None)
        if root is None or not self.frames:
            return 0.0
        p0 = np.array(self.frames[0][root][0])
        p1 = np.array(self.frames[-1][root][0])
        return float(np.linalg.norm(p1[:2] - p0[:2]))


def record_trajectory(genome, *, controller: Optional[Callable] = None,
                      policy=None, niche: str = "locomotion", n_steps: int = 120,
                      ep_steps: Optional[int] = None, seed: int = 0, fps: int = 30,
                      name: str = "creature") -> Trajectory:
    """Roll a creature out under `controller` (or `policy`, or the default gait) and
    record per-body world transforms each step."""
    task = NICHES[niche](max_steps=ep_steps or (n_steps + 5))
    env = CreatureEnv(genome, task=task, obs_mode="flat")
    model = env.model
    plan = plan_from_genome(genome, name=name).to_dict()
    body_ids = [g["name"] for g in plan["geoms"]]
    body_idx = {bid: model.body(bid).id for bid in body_ids}

    if controller is None and policy is None:
        controller = _sine_gait(env.action_space.shape[0], env.control_dt)

    def _frame():
        return {bid: [list(map(float, env.data.xpos[i])),
                      list(map(float, env.data.xquat[i]))]
                for bid, i in body_idx.items()}

    obs, _ = env.reset(seed=seed)
    frames = [_frame()]
    for t in range(n_steps):
        if policy is not None:
            action = np.asarray(policy.act(obs, deterministic=True), dtype=np.float32)
        else:
            action = controller(t, obs)
        obs, _, term, trunc, _ = env.step(action)
        frames.append(_frame())
        if term or trunc:
            break
    env.close()
    return Trajectory(plan=plan, frames=frames, fps=fps)
