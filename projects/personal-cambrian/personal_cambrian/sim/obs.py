"""Morphology-agnostic observation builder (ROADMAP Stage 2.2).

Produces two views of the same proprioceptive state:

* `flat(data, prev_action)` -> 1-D vector, the standard RL observation:
      [ joint_qpos | joint_qvel | gravity_in_root(3) | root_lin_vel(3)
        | root_ang_vel(3) | floor_contact_per_body | prev_action ]
  with global x,y dropped (translation invariance) by using only 1-DOF joint
  angles + the root's orientation/velocity rather than absolute position.

* `structured(data, prev_action)` -> (n_actuators, node_dim) matrix, one row per
  actuator (muscle) for the Stage-3 modular / graph policy. Each node carries its
  own command + tendon length/velocity, the mean angle/velocity of the joints its
  tendon spans, and a shared global orientation/ang-vel context:
      [ prev_action_i | ten_len_i | ten_vel_i | mean_span_qpos | mean_span_qvel
        | gravity_in_root(3) | root_ang_vel(3) ]

Dims are fixed per creature and documented via the attributes below
(`flat_dim`, `n_nodes`, `node_dim`).
"""
from __future__ import annotations

import numpy as np
import mujoco

_ONE_DOF = (int(mujoco.mjtJoint.mjJNT_HINGE), int(mujoco.mjtJoint.mjJNT_SLIDE))
_GRAV = np.array([0.0, 0.0, -1.0])


class ObservationBuilder:
    def __init__(self, model):
        self.model = model
        self.nu = int(model.nu)

        # free root (if any) -> root body for orientation / velocity
        frees = [j for j in range(model.njnt)
                 if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE]
        self.free_jnt = frees[0] if frees else None
        self.root_body = int(model.jnt_bodyid[self.free_jnt]) if frees else 1

        # 1-DOF joints (hinge/slide): their qpos and dof addresses
        j1 = [j for j in range(model.njnt) if int(model.jnt_type[j]) in _ONE_DOF]
        self.j1_ids = np.array(j1, dtype=int)
        self.jnt_qposadr = np.array([model.jnt_qposadr[j] for j in j1], dtype=int)
        self.jnt_dofadr = np.array([model.jnt_dofadr[j] for j in j1], dtype=int)
        self.n_jnt1 = len(j1)

        # creature bodies (exclude world body 0) -> floor-contact flag slots
        self.n_bodies = model.nbody - 1

        # actuator -> tendon, and which 1-DOF joints each tendon spans (topology,
        # precomputed once at the neutral pose by finite-differencing length)
        self.act_tendon = np.array(
            [int(model.actuator_trnid[a, 0]) for a in range(self.nu)], dtype=int)
        self.act_span = self._precompute_spanned_joints()

        self.node_dim = 1 + 1 + 1 + 1 + 1 + 3 + 3      # = 11
        self.n_nodes = self.nu
        self.flat_dim = (2 * self.n_jnt1 + 3 + 3 + 3 + self.n_bodies + self.nu)

    # -- precompute ---------------------------------------------------------
    def _precompute_spanned_joints(self) -> list[np.ndarray]:
        d = mujoco.MjData(self.model)
        mujoco.mj_resetData(self.model, d)
        mujoco.mj_forward(self.model, d)
        base = d.ten_length.copy()
        spans: list[list[int]] = [[] for _ in range(self.nu)]
        for k, qadr in enumerate(self.jnt_qposadr):
            d.qpos[qadr] += 1e-3
            mujoco.mj_forward(self.model, d)
            dlen = np.abs(d.ten_length - base)
            d.qpos[qadr] -= 1e-3
            for a in range(self.nu):
                if dlen[self.act_tendon[a]] / 1e-3 > 1e-3:
                    spans[a].append(k)
        mujoco.mj_forward(self.model, d)
        return [np.array(s, dtype=int) for s in spans]

    # -- shared pieces ------------------------------------------------------
    def _orientation_and_root_vel(self, data):
        R = data.xmat[self.root_body].reshape(3, 3)
        grav_local = R.T @ _GRAV
        if self.free_jnt is not None:
            lin_local = R.T @ data.qvel[0:3]
            ang = data.qvel[3:6].copy()
        else:
            lin_local = np.zeros(3)
            ang = np.zeros(3)
        return grav_local, lin_local, ang

    def _floor_contacts(self, data) -> np.ndarray:
        flags = np.zeros(self.n_bodies, dtype=np.float32)
        for c in data.contact[: data.ncon]:
            b1 = int(self.model.geom_bodyid[c.geom1])
            b2 = int(self.model.geom_bodyid[c.geom2])
            if b1 == 0 or b2 == 0:                 # contact with the world (floor)
                other = b2 if b1 == 0 else b1
                if other >= 1:
                    flags[other - 1] = 1.0
        return flags

    # -- public views -------------------------------------------------------
    def flat(self, data, prev_action) -> np.ndarray:
        grav, lin, ang = self._orientation_and_root_vel(data)
        obs = np.concatenate([
            data.qpos[self.jnt_qposadr],
            data.qvel[self.jnt_dofadr],
            grav, lin, ang,
            self._floor_contacts(data),
            np.asarray(prev_action, dtype=np.float64).ravel(),
        ]).astype(np.float32)
        return np.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)

    def adjacency(self) -> np.ndarray:
        """(n_actuators, n_actuators) actuator graph for the modular policy.

        Two actuators are connected if the joints their tendons span sit on the
        same body or on parent-child bodies — i.e. the kinematic tree, lifted to
        actuators. Self-loops included. Returns identity if there is one actuator.
        """
        nu = self.nu
        A = np.eye(nu, dtype=np.float32)
        parent = self.model.body_parentid
        act_bodies = []
        for a in range(nu):
            bodies = {int(self.model.jnt_bodyid[self.j1_ids[k]]) for k in self.act_span[a]}
            act_bodies.append(bodies)

        def related(b1, b2):
            return b1 == b2 or parent[b1] == b2 or parent[b2] == b1

        for i in range(nu):
            for j in range(i + 1, nu):
                if any(related(bi, bj) for bi in act_bodies[i] for bj in act_bodies[j]):
                    A[i, j] = A[j, i] = 1.0
        return A

    def structured(self, data, prev_action) -> np.ndarray:
        grav, _, ang = self._orientation_and_root_vel(data)
        qpos = data.qpos[self.jnt_qposadr]
        qvel = data.qvel[self.jnt_dofadr]
        pa = np.asarray(prev_action, dtype=np.float64).ravel()
        rows = np.zeros((self.nu, self.node_dim), dtype=np.float32)
        for a in range(self.nu):
            t = self.act_tendon[a]
            span = self.act_span[a]
            mq = float(qpos[span].mean()) if span.size else 0.0
            mv = float(qvel[span].mean()) if span.size else 0.0
            rows[a, 0] = pa[a] if a < pa.size else 0.0
            rows[a, 1] = data.ten_length[t]
            rows[a, 2] = data.ten_velocity[t]
            rows[a, 3] = mq
            rows[a, 4] = mv
            rows[a, 5:8] = grav
            rows[a, 8:11] = ang
        return np.nan_to_num(rows, nan=0.0, posinf=0.0, neginf=0.0)
