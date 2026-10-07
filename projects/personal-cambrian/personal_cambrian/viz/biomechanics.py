"""ROLLOUT -> BIOMECHANICS TRAJECTORY (ROADMAP Stage 10.4, bpy-free core).

Extends the Stage-10.2 rollout recording with the per-frame BIOMECHANICS the clip
overlays: for every muscle, its line of action (origin->insertion world sites), the
tension it is pulling with (|actuator_force|), and its MOMENT ARM about the joint it
crosses -- the perpendicular distance from the joint anchor to the muscle's line of
action (so it can be drawn as a lever). Bundled into a JSON-serializable
`BioTrajectory` that `blender_build.animate_biomechanics` turns into a clip with
force-scaled muscle bars + moment-arm levers.

Muscle endpoints come from `data.site_xpos` (site name `{body}:{site}`), force from the
muscle's actuator `{muscle.id}:act`, and the crossed joint's anchor from `data.xanchor`.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

from ..sim import CreatureEnv
from ..sim.tasks import NICHES
from ..morphogenesis import develop
from .morphology_plan import plan_from_genome
from .trajectory import _sine_gait


@dataclass
class BioTrajectory:
    plan: dict
    frames: list                 # [{"bodies": {id:[pos,quat]}, "muscles": [..]}, ...]
    fps: int
    force_max: float             # peak muscle force over the run (for normalization)
    moment_max: float

    def to_dict(self) -> dict:
        return {"plan": self.plan, "frames": self.frames, "fps": int(self.fps),
                "force_max": self.force_max, "moment_max": self.moment_max}

    @property
    def n_frames(self) -> int:
        return len(self.frames)

    def peak_force(self) -> float:
        return self.force_max


def _foot_of_perpendicular(anchor, org, ins):
    """Closest point on segment org->ins to `anchor`, and the moment-arm length."""
    v = ins - org
    L2 = float(v @ v)
    t = 0.0 if L2 < 1e-12 else float(np.clip((anchor - org) @ v / L2, 0.0, 1.0))
    foot = org + t * v
    return foot, float(np.linalg.norm(anchor - foot))


def record_biomechanics(genome, *, controller: Optional[Callable] = None, policy=None,
                        niche: str = "locomotion", n_steps: int = 120,
                        seed: int = 0, fps: int = 30, name: str = "creature") -> BioTrajectory:
    """Roll the creature out and record per-frame body transforms + per-muscle
    force / line-of-action / moment-arm lever."""
    task = NICHES[niche](max_steps=n_steps + 5)
    env = CreatureEnv(genome, task=task, obs_mode="flat")
    model, data = env.model, env.data
    morph = develop(genome)
    byid = {b.id: b for b in morph.bodies}
    plan = plan_from_genome(genome, name=name).to_dict()
    body_idx = {g["name"]: model.body(g["name"]).id for g in plan["geoms"]}

    def _site(wp):
        return model.site(f"{wp.body_id}:{byid[wp.body_id].sites[wp.site_idx].name}").id

    meta = []                                            # (act_id, org_sid, ins_sid, jnt)
    for mus in morph.muscles:
        act_id = model.actuator(f"{mus.id}:act").id
        org_sid, ins_sid = _site(mus.waypoints[0]), _site(mus.waypoints[-1])
        bid = model.body(mus.waypoints[-1].body_id).id
        jnt = int(model.body_jntadr[bid]) if model.body_jntnum[bid] > 0 else -1
        meta.append((act_id, org_sid, ins_sid, jnt))

    def _frame():
        bodies = {bid: [list(map(float, data.xpos[i])), list(map(float, data.xquat[i]))]
                  for bid, i in body_idx.items()}
        muscles = []
        for act_id, org_sid, ins_sid, jnt in meta:
            org = data.site_xpos[org_sid].copy()
            ins = data.site_xpos[ins_sid].copy()
            force = abs(float(data.actuator_force[act_id]))
            if jnt >= 0:
                anchor = data.xanchor[jnt].copy()
                foot, ma = _foot_of_perpendicular(anchor, org, ins)
                anchor_l, foot_l = anchor.tolist(), foot.tolist()
            else:
                anchor_l = foot_l = None
                ma = 0.0
            muscles.append({"org": org.tolist(), "ins": ins.tolist(), "force": force,
                            "anchor": anchor_l, "foot": foot_l, "moment_arm": ma})
        return {"bodies": bodies, "muscles": muscles}

    if controller is None and policy is None:
        controller = _sine_gait(env.action_space.shape[0], env.control_dt)

    obs, _ = env.reset(seed=seed)
    frames = [_frame()]
    for t in range(n_steps):
        action = (np.asarray(policy.act(obs, deterministic=True), dtype=np.float32)
                  if policy is not None else controller(t, obs))
        obs, _, term, trunc, _ = env.step(action)
        frames.append(_frame())
        if term or trunc:
            break
    env.close()

    fmax = max((mu["force"] for fr in frames for mu in fr["muscles"]), default=0.0)
    mmax = max((mu["moment_arm"] for fr in frames for mu in fr["muscles"]), default=0.0)
    return BioTrajectory(plan=plan, frames=frames, fps=fps,
                         force_max=max(fmax, 1e-6), moment_max=max(mmax, 1e-6))
