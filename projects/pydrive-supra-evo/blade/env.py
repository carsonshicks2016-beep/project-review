"""BLADE environment — stage-driven 3D humanoid duels in MuJoCo.

A `BladeEnv` is configured by a `StageConfig` (world + reward + rules), so the
same code serves every curriculum rung from "stand still" to "armed spar".  It is
weapon-agnostic: the striking surface of whatever the fighter holds is tagged
`strike_*` and its tip is the `weapon_tip` site, so observations, hit detection,
and the aim reward work for fist, sword, mace, axe, katana, or shield alike.

`VecBlade` runs a batch of identical-stage duels for PPO, threading MuJoCo's
GIL-releasing `mj_step` across cores.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import numpy as np
import mujoco

from .model import make_model, JOINT_NAMES, N_JOINT
from .config import StageConfig

CONTROL_DECIM = 5          # 0.005 s * 5 = 40 Hz control
HP0 = 100.0
HIT_DMG = 20.0
HIT_COOLDOWN = 0.35
ENGAGE_RANGE = 1.3
TIP_RANGE = 1.0

# a nearly-upright guard stance (joint -> radians); action 0 holds this
STANCE = {
    "r_hip_y": -0.10, "r_knee": -0.20, "r_ankle": -0.20,
    "l_hip_y": -0.10, "l_knee": -0.20, "l_ankle": -0.20,
    "r_shoulder1": -0.6, "r_shoulder2": -0.3, "r_elbow": -1.0,
    "l_shoulder1": -0.4, "l_shoulder2": 0.4, "l_elbow": -1.0,
}


def _id(m, t, name):
    return mujoco.mj_name2id(m, t, name)


class BladeEnv:
    def __init__(self, stage: StageConfig, seed=0, loadout_b=None):
        self.stage = stage
        self.rng = np.random.default_rng(seed)
        self.model = make_model(loadout=stage.loadout, terrain=stage.terrain,
                                seed=seed, bump=stage.bump, loadout_b=loadout_b)
        self.data = mujoco.MjData(self.model)
        m = self.model
        self.dt = m.opt.timestep * CONTROL_DECIM
        self.max_steps = int(stage.max_seconds / self.dt)

        self.fi = {}
        for f in ("a_", "b_"):
            jq = [m.jnt_qposadr[_id(m, mujoco.mjtObj.mjOBJ_JOINT, f + jn)] for jn in JOINT_NAMES]
            jv = [m.jnt_dofadr[_id(m, mujoco.mjtObj.mjOBJ_JOINT, f + jn)] for jn in JOINT_NAMES]
            root = _id(m, mujoco.mjtObj.mjOBJ_JOINT, f + "root")
            shield = _id(m, mujoco.mjtObj.mjOBJ_BODY, f + "shield")
            self.fi[f] = dict(
                jqpos=np.array(jq), jqvel=np.array(jv),
                rootq=m.jnt_qposadr[root], rootv=m.jnt_dofadr[root],
                torso=_id(m, mujoco.mjtObj.mjOBJ_BODY, f + "torso"),
                tip=_id(m, mujoco.mjtObj.mjOBJ_SITE, f + "weapon_tip"),
                guard=shield if shield >= 0 else _id(m, mujoco.mjtObj.mjOBJ_BODY, f + "l_hand"),
                rfoot=_id(m, mujoco.mjtObj.mjOBJ_BODY, f + "r_foot"),
                lfoot=_id(m, mujoco.mjtObj.mjOBJ_BODY, f + "l_foot"),
            )
        self.act_slc = {"a_": slice(0, N_JOINT), "b_": slice(N_JOINT, 2 * N_JOINT)}
        cr = m.actuator_ctrlrange
        self.ctrl_half = (cr[:, 1] - cr[:, 0]) / 2

        self.geom_cat = {}
        for gi in range(m.ngeom):
            nm = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, gi) or ""
            for f in ("a_", "b_"):
                if nm.startswith(f):
                    part = ("strike" if "strike" in nm else "shield" if "shield" in nm
                            else "weapon" if "wpn" in nm else "body")
                    self.geom_cat[gi] = (f, part)
        self.stance_ctrl = self._stance_ctrl()
        self.obs_dim = self.reset().shape[1]
        self.act_dim = N_JOINT

    def _stance_ctrl(self):
        c = np.zeros(self.model.nu, np.float32)
        cr = self.model.actuator_ctrlrange
        for f in ("a_", "b_"):
            base = self.act_slc[f].start
            for j, jn in enumerate(JOINT_NAMES):
                if jn in STANCE:
                    c[base + j] = np.clip(STANCE[jn], cr[base + j, 0], cr[base + j, 1])
        return c

    # ------------------------------------------------------------------ reset
    def reset(self):
        m, d, s = self.model, self.data, self.stage
        if s.terrain:
            nr, nc = int(m.hfield_nrow[0]), int(m.hfield_ncol[0])
            h = self.rng.standard_normal((nr, nc))
            for _ in range(6):
                h = (h + np.roll(h, 1, 0) + np.roll(h, -1, 0) + np.roll(h, 1, 1) + np.roll(h, -1, 1)) / 5
            h -= h.min(); h /= max(h.max(), 1e-6)
            m.hfield_data[:] = (h * s.bump).ravel()
        mujoco.mj_resetData(m, d)
        hd = s.spawn_dist / 2.0
        for f, (x, qz) in (("a_", (-hd, 0.0)), ("b_", (hd, 1.0))):
            fi = self.fi[f]
            d.qpos[fi["rootq"]:fi["rootq"] + 3] = [x, 0, 1.02]
            d.qpos[fi["rootq"] + 3:fi["rootq"] + 7] = [np.cos(np.pi / 2 * qz), 0, 0, np.sin(np.pi / 2 * qz)]
            for j, jn in enumerate(JOINT_NAMES):
                d.qpos[fi["jqpos"][j]] = STANCE.get(jn, 0.0) + self.rng.uniform(-0.05, 0.05)
        d.ctrl[:] = self.stance_ctrl
        mujoco.mj_forward(m, d)
        self.steps = 0
        self.hp = {"a_": HP0, "b_": HP0}
        self.cool = {"a_": 0.0, "b_": 0.0}
        self.init_dist = self._dist()
        self.prev_dist = self.init_dist
        return self._obs()

    def _dist(self):
        return float(np.linalg.norm(self.data.xpos[self.fi["a_"]["torso"]][:2] -
                                    self.data.xpos[self.fi["b_"]["torso"]][:2]))

    # ------------------------------------------------------------------- obs
    def _fighter_obs(self, f, o):
        d, fi, oi = self.data, self.fi[f], self.fi[o]
        torso = d.xpos[fi["torso"]]
        R = d.xmat[fi["torso"]].reshape(3, 3)
        rel = d.xpos[oi["torso"]] - torso
        return np.concatenate([
            d.qpos[fi["jqpos"]], d.qvel[fi["jqvel"]] * 0.1,
            [torso[2]], R[:, 2], R[:, 0], d.qvel[fi["rootv"]:fi["rootv"] + 6] * 0.3,
            d.site_xpos[fi["tip"]] - torso, d.xpos[fi["guard"]] - torso,
            rel, d.site_xpos[oi["tip"]] - torso, d.xmat[oi["torso"]].reshape(3, 3)[:, 0],
            [np.linalg.norm(rel[:2])],
            [float(d.xpos[fi["rfoot"]][2] < 0.22), float(d.xpos[fi["lfoot"]][2] < 0.22)],
            [self.hp[f] / HP0, self.hp[o] / HP0],
        ]).astype(np.float32)

    def _obs(self):
        return np.stack([self._fighter_obs("a_", "b_"), self._fighter_obs("b_", "a_")])

    # ------------------------------------------------------------------ step
    def step(self, action, auto_reset=True):
        m, d, s = self.model, self.data, self.stage
        a = np.clip(np.asarray(action, np.float32).reshape(2, N_JOINT), -1, 1)
        ctrl = self.stance_ctrl.copy()
        for i, f in enumerate(("a_", "b_")):
            sl = self.act_slc[f]
            ctrl[sl] = np.clip(self.stance_ctrl[sl] + self.ctrl_half[sl] * a[i],
                               m.actuator_ctrlrange[sl, 0], m.actuator_ctrlrange[sl, 1])
        d.ctrl[:] = ctrl
        for _ in range(CONTROL_DECIM):
            mujoco.mj_step(m, d)
        self.steps += 1
        self.cool["a_"] = max(0.0, self.cool["a_"] - self.dt)
        self.cool["b_"] = max(0.0, self.cool["b_"] - self.dt)

        rew = {"a_": 0.0, "b_": 0.0}
        hits = {"a_": 0, "b_": 0}
        if s.combat:
            self._resolve_contacts(rew, hits)

        dist = self._dist()
        upright = {}
        for i, (f, o) in enumerate((("a_", "b_"), ("b_", "a_"))):
            fi = self.fi[f]
            torso = d.xpos[fi["torso"]]
            R = d.xmat[fi["torso"]].reshape(3, 3)
            up_z = R[2, 2]
            rel = d.xpos[self.fi[o]["torso"]] - torso
            rd = np.linalg.norm(rel[:2]) + 1e-6
            to_opp = np.array([rel[0], rel[1], 0.0]) / rd
            rew[f] += s.w_up * up_z + s.w_height * np.clip(torso[2] - 0.9, -0.35, 0.3) + s.w_alive
            rew[f] += s.w_crouch * float(np.clip(-d.qpos[fi["jqpos"][[5, 10]]], 0, 0.5).mean())
            if s.w_forward:
                vel = d.qvel[fi["rootv"]:fi["rootv"] + 2]
                rew[f] += s.w_forward * float(np.clip(np.dot(vel, to_opp[:2]), -1.0, 3.0))
            if s.w_face:
                rew[f] += s.w_face * float(np.dot(R[:, 0], to_opp))
            if s.w_close:
                rew[f] += s.w_close * float(np.clip(ENGAGE_RANGE - dist, 0, ENGAGE_RANGE))
            if s.w_far:
                rew[f] -= s.w_far * float(np.clip(dist - 1.1, 0, 3))
            if s.w_tip:
                tip_d = np.linalg.norm(d.site_xpos[fi["tip"]] - d.xpos[self.fi[o]["torso"]])
                rew[f] += s.w_tip * float(np.clip(TIP_RANGE - tip_d, 0, TIP_RANGE))
            rew[f] -= s.w_ctrl * float(np.sum(a[i] ** 2))
            upright[f] = bool(torso[2] > s.fall_h and up_z > 0.4)
        if s.w_approach:
            for f in ("a_", "b_"):
                rew[f] += s.w_approach * (self.prev_dist - dist)
        self.prev_dist = dist

        fallen = {f: d.xpos[self.fi[f]["torso"]][2] < s.fall_h for f in ("a_", "b_")}
        ko = {f: self.hp[f] <= 0 for f in ("a_", "b_")}
        done = self.steps >= self.max_steps or any(fallen.values()) or any(ko.values())
        winner = -1
        if done:
            for f in ("a_", "b_"):
                if fallen[f] or ko[f]:
                    rew[f] += s.r_fall
            if s.combat:
                aL, bL = fallen["a_"] or ko["a_"], fallen["b_"] or ko["b_"]
                if aL and not bL: winner = 1
                elif bL and not aL: winner = 0
                elif not aL and not bL:
                    winner = 0 if self.hp["a_"] > self.hp["b_"] else (1 if self.hp["b_"] > self.hp["a_"] else -1)
                if winner == 0: rew["a_"] += s.r_win; rew["b_"] -= s.r_win
                elif winner == 1: rew["b_"] += s.r_win; rew["a_"] -= s.r_win

        obs = self._obs()
        reward = np.array([rew["a_"], rew["b_"]], np.float32)
        info = {"winner": winner, "hp": (self.hp["a_"], self.hp["b_"]),
                "upright": np.array([upright["a_"], upright["b_"]]),
                "dist": dist, "init_dist": self.init_dist,
                "hits": np.array([hits["a_"], hits["b_"]])}
        if not np.isfinite(obs).all() or not np.isfinite(reward).all():
            done, reward = True, np.nan_to_num(reward)
        if auto_reset and done:
            obs = self.reset()
        return obs, reward, done, info

    def _resolve_contacts(self, rew, hits):
        d, s = self.data, self.stage
        for c in d.contact[:d.ncon]:
            c1, c2 = self.geom_cat.get(c.geom1), self.geom_cat.get(c.geom2)
            if not c1 or not c2 or c1[0] == c2[0]:
                continue
            for att, dfn in ((c1, c2), (c2, c1)):
                if att[1] == "strike" and dfn[1] == "body":
                    f = att[0]
                    if self.cool[f] <= 0:
                        o = "b_" if f == "a_" else "a_"
                        self.hp[o] -= HIT_DMG
                        rew[f] += s.r_hit; rew[o] -= s.r_hit_taken
                        hits[f] += 1
                        self.cool[f] = HIT_COOLDOWN
                elif att[1] == "strike" and dfn[1] == "shield":
                    rew[dfn[0]] += s.r_block


class VecBlade:
    def __init__(self, stage: StageConfig, n_envs, seed=0, workers=None):
        self.n = n_envs
        self.stage = stage
        self.envs = [BladeEnv(stage, seed=seed + 1000 * i) for i in range(n_envs)]
        self.obs_dim = self.envs[0].obs_dim
        self.act_dim = self.envs[0].act_dim
        self.dt = self.envs[0].dt
        self.max_steps = self.envs[0].max_steps
        self.pool = ThreadPoolExecutor(max_workers=workers or min(n_envs, 16))

    def reset(self):
        return np.stack(list(self.pool.map(lambda e: e.reset(), self.envs)))

    def step(self, actions, auto_reset=True):
        a = np.asarray(actions, np.float32).reshape(self.n, 2, self.act_dim)
        res = list(self.pool.map(lambda i: self.envs[i].step(a[i], auto_reset), range(self.n)))
        obs = np.stack([r[0] for r in res])
        rew = np.stack([r[1] for r in res])
        done = np.array([r[2] for r in res], bool)
        return obs, rew, done, [r[3] for r in res]
