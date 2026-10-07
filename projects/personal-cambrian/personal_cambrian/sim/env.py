"""CreatureEnv: a Gymnasium environment wrapping a compiled creature + a task.

Stage 2.1 builds the structural wrapper (spaces, reset, step, deterministic
seeding, optional rgb_array render) with a *preliminary* locomotion reward and
fall/timeout termination. Later stages refine: 2.2 richer/structured observations
for the modular policy, 2.3 action mapping, 2.4 reward shaping, 2.5 declarative
task configs.

Observations are proprioceptive and translation-invariant (the root's global x,y
are dropped). Actions are in [-1, 1] per actuator and rescaled to each actuator's
control range.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import gymnasium as gym
from gymnasium import spaces
import mujoco

from ..encoding.genome import Genome
from ..morphogenesis import develop
from ..morphogenesis.to_mujoco import compile_morphology
from .obs import ObservationBuilder
from .tasks.locomotion import LocomotionTask


class CreatureEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 100}

    def __init__(self, genome: Optional[Genome] = None, *, model=None,
                 task: Optional[LocomotionTask] = None, n_substeps: int = 5,
                 obs_mode: str = "flat", render_mode: Optional[str] = None):
        super().__init__()
        self.task = task or LocomotionTask()
        if model is None:
            if genome is None:
                raise ValueError("CreatureEnv needs a genome or a compiled model")
            model, _ = compile_morphology(develop(genome), add_floor=True, free_root=True)
        self.model = model
        self.task.configure_model(self.model)   # Stage 7: terrain/slope/gravity setup
        self.data = mujoco.MjData(model)
        if model.nu == 0:
            raise ValueError("creature has no actuators; cannot form an action space")

        self.n_substeps = int(n_substeps)
        self.control_dt = self.n_substeps * float(model.opt.timestep)
        self.render_mode = render_mode
        self.obs_mode = obs_mode          # "flat" (MLP) | "structured" (modular policy)
        self._adj = None
        self._renderer = None
        self._elapsed = 0

        # actuator control ranges (for action rescaling)
        self._ctrl_lo = model.actuator_ctrlrange[:, 0].copy()
        self._ctrl_hi = model.actuator_ctrlrange[:, 1].copy()
        self._ctrl_limited = self._ctrl_hi > self._ctrl_lo

        # standing initial pose (lift so the lowest creature point ~ floor)
        self._init_qpos = self._grounded_qpos()

        # observation builder (flat + structured per-actuator views)
        self.obs_builder = ObservationBuilder(model)
        self._prev_action = np.zeros(model.nu, dtype=np.float32)

        self.action_space = spaces.Box(-1.0, 1.0, shape=(model.nu,), dtype=np.float32)
        obs = self._get_obs()
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=obs.shape, dtype=np.float32)

    # -- setup helpers ------------------------------------------------------
    def _creature_geoms(self):
        return [i for i in range(self.model.ngeom)
                if self.model.geom(i).name != "floor"]

    def _grounded_qpos(self) -> np.ndarray:
        mujoco.mj_resetData(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)
        cre = self._creature_geoms()
        zmin = min(float(self.data.geom_xpos[i, 2] - self.model.geom_rbound[i]) for i in cre)
        qpos = self.data.qpos.copy()
        qpos[2] += 0.05 - zmin
        return qpos

    @property
    def _root_is_free(self) -> bool:
        return self.model.njnt > 0 and self.model.jnt_type[0] == mujoco.mjtJoint.mjJNT_FREE

    # control-rate contract: each env.step advances physics by `n_substeps`
    # integrator steps (frame-skip), i.e. control_dt = n_substeps * physics_dt.
    @property
    def physics_dt(self) -> float:
        return float(self.model.opt.timestep)

    @property
    def control_hz(self) -> float:
        return 1.0 / self.control_dt

    def _get_obs(self) -> np.ndarray:
        if self.obs_mode == "structured":
            return self.obs_builder.structured(self.data, self._prev_action).reshape(-1)
        return self.obs_builder.flat(self.data, self._prev_action)

    def structured_obs(self) -> np.ndarray:
        """(n_actuators, node_dim) per-muscle node features for the Stage-3 modular
        policy. See ObservationBuilder.structured()."""
        return self.obs_builder.structured(self.data, self._prev_action)

    @property
    def actuator_adjacency(self) -> np.ndarray:
        if self._adj is None:
            self._adj = self.obs_builder.adjacency()
        return self._adj

    def _map_action(self, action: np.ndarray) -> np.ndarray:
        """Map a policy action in [-1, 1]^nu to actuator controls.

        NaN -> 0; values are clipped to [-1, 1] then affinely rescaled to each
        actuator's control range (identity for unlimited actuators), with a final
        defensive clamp so the result is always within range.
        """
        a = np.nan_to_num(np.asarray(action, dtype=np.float64).reshape(-1), nan=0.0)
        a = np.clip(a, -1.0, 1.0)
        ctrl = self._ctrl_lo + (a + 1.0) * 0.5 * (self._ctrl_hi - self._ctrl_lo)
        ctrl = np.where(self._ctrl_limited, ctrl, a)
        return np.where(self._ctrl_limited,
                        np.clip(ctrl, self._ctrl_lo, self._ctrl_hi), ctrl)

    # -- gym API ------------------------------------------------------------
    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:] = self._init_qpos
        # small symmetry-breaking noise on joint angles + velocities
        nqj = self.data.qpos.shape[0] - (7 if self._root_is_free else 0)
        if nqj > 0:
            self.data.qpos[-nqj:] += self.np_random.uniform(-0.02, 0.02, size=nqj)
        self.data.qvel[:] += self.np_random.uniform(-0.02, 0.02, size=self.data.qvel.shape)
        mujoco.mj_forward(self.model, self.data)

        self._elapsed = 0
        self._prev_action[:] = 0.0
        self._stand_height = float(self.data.qpos[2])
        self._prev_x = float(self.data.qpos[self.task.forward_axis])
        com = self.data.subtree_com[0].copy()
        self._prev_com = com
        self._start_xy = com[:2].copy()
        self.task.reset_episode(self.np_random)
        return self._get_obs(), {}

    def step(self, action):
        prior_action = self._prev_action.copy()
        self.data.ctrl[:] = self._map_action(action)
        clipped = np.clip(np.asarray(action, dtype=np.float32).reshape(-1), -1.0, 1.0)
        # Stage 7: the niche may apply an external disturbance/load before stepping
        self.task.perturb(self.model, self.data, self.np_random, self._elapsed)
        for _ in range(self.n_substeps):
            mujoco.mj_step(self.model, self.data)
        self._elapsed += 1
        self._prev_action = clipped

        # physical quantities the task scores
        x = float(self.data.qpos[self.task.forward_axis])
        forward_vel = (x - self._prev_x) / self.control_dt
        self._prev_x = x
        root_z = float(self.data.qpos[2])
        up_proj = float(self.data.xmat[self.obs_builder.root_body].reshape(3, 3)[2, 2])
        com = self.data.subtree_com[0].copy()
        com_vel = (com - self._prev_com) / self.control_dt
        self._prev_com = com
        displacement = float(np.linalg.norm(com[:2] - self._start_xy))

        state = dict(
            forward_vel=forward_vel, action=clipped, prev_action=prior_action,
            dt=self.control_dt, root_z=root_z, stand_height=self._stand_height,
            up_proj=up_proj, com=com, com_vel=com_vel, com_vz=float(com_vel[2]),
            com_z=float(com[2]), displacement=displacement,
            actuator_force=self.data.actuator_force.copy(), elapsed=self._elapsed)

        reward, components = self.task.reward(**state)
        fallen, reason = self.task.is_fallen(**state)
        nan = not np.all(np.isfinite(self.data.qpos))
        terminated = bool(fallen or nan)
        truncated = bool(self.task.truncated(self._elapsed))

        info = {"forward_vel": forward_vel, "x": x, "root_z": root_z,
                "up_proj": up_proj, "com_z": float(com[2]), "com_vz": float(com_vel[2]),
                "displacement": displacement, "fall_reason": reason if fallen else "",
                "reward_terms": components}
        return self._get_obs(), float(reward), terminated, truncated, info

    def render(self):
        if self.render_mode != "rgb_array":
            return None
        if self._renderer is None:
            self._renderer = mujoco.Renderer(self.model, 480, 640)
        cam = mujoco.MjvCamera()
        cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        cam.lookat = self.data.subtree_com[0]
        cam.distance = 3.0
        cam.azimuth, cam.elevation = 140, -20
        self._renderer.update_scene(self.data, camera=cam)
        return self._renderer.render()

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
