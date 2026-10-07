"""MJX (JAX) VECTORIZED PHYSICS BACKEND (ROADMAP Stage 9.1).

The CPU `CreatureEnv` steps one creature at a time through the C MuJoCo engine -- a
few thousand steps/sec, which makes deep-time runs (Stage 8) a CPU bottleneck. MJX
re-expresses the SAME compiled model in JAX so the physics can be `jit`+`vmap`'d
across THOUSANDS of parallel envs on one accelerator. On a GPU this is the 100x+
throughput that makes big runs feasible; on CPU (where this was developed) it still
vectorizes, just without the GPU multiplier.

Two deliverables for 9.1:

  * `MjxBatchEnv` -- B parallel copies of one creature, advanced by a jit+vmap
    rollout. The control mapping matches the CPU env (clip to [-1,1], rescale to
    each actuator's ctrlrange) so a policy transfers unchanged.
  * `dynamics_parity` -- MJX vs CPU agreement on a fixed creature. The port is
    FAITHFUL in smooth (contact-free) dynamics (~1e-7 / step); it DIVERGES under
    rich contact because MJX uses a soft contact model where the C engine is rigid.
    That is a known MJX limitation: the CPU path stays the ground-truth ORACLE for
    final scoring, MJX is the fast pre-screen / training substrate.

Everything here is import-guarded (`mjx_available()`); the package imports fine on a
machine without jax/mujoco-mjx, and the CPU `CreatureEnv` is untouched.
"""
from __future__ import annotations

import numpy as np
import mujoco

from ..morphogenesis import develop
from ..morphogenesis.to_mujoco import compile_morphology

try:
    import jax
    import jax.numpy as jp
    from mujoco import mjx
    _HAS_MJX = True
except Exception:        # noqa: BLE001 -- jax/mjx are optional (Stage 9 GPU extra)
    _HAS_MJX = False


def mjx_available() -> bool:
    """True if the JAX/MJX backend can be used (jax + mujoco-mjx installed)."""
    return _HAS_MJX


def _require():
    if not _HAS_MJX:
        raise RuntimeError("MJX backend requires jax + mujoco-mjx "
                           "(`pip install jax mujoco-mjx`).")


def _compile(genome, model):
    if model is None:
        model, _ = compile_morphology(develop(genome), add_floor=True, free_root=True)
    return model


def grounded_qpos(model) -> np.ndarray:
    """Standing init pose: lift the body so its lowest point sits ~at the floor
    (mirrors CreatureEnv._grounded_qpos so MJX and CPU start identically)."""
    d = mujoco.MjData(model)
    mujoco.mj_resetData(model, d)
    mujoco.mj_forward(model, d)
    cre = [i for i in range(model.ngeom) if model.geom(i).name != "floor"]
    zmin = min(float(d.geom_xpos[i, 2] - model.geom_rbound[i]) for i in cre)
    q = d.qpos.copy()
    q[2] += 0.05 - zmin
    return q


# --- batched MJX environment -----------------------------------------------
class MjxBatchEnv:
    """`n_envs` parallel copies of one creature, stepped with a jit+vmap MJX rollout.

    Use `init(n_envs)` to build a batched MJX Data (all at the standing pose), then
    `step(data, actions)` with `actions` of shape (n_envs, nu) in [-1, 1]. `qpos(data)`
    pulls the (n_envs, nq) positions back to numpy."""

    def __init__(self, genome=None, *, model=None, n_substeps: int = 5):
        _require()
        self.model = _compile(genome, model)
        if self.model.nu == 0:
            raise ValueError("creature has no actuators; cannot form an action space")
        self.mx = mjx.put_model(self.model)
        self.n_substeps = int(n_substeps)
        self.control_dt = self.n_substeps * float(self.model.opt.timestep)

        lo = self.model.actuator_ctrlrange[:, 0]
        hi = self.model.actuator_ctrlrange[:, 1]
        self._lo = jp.asarray(lo)
        self._hi = jp.asarray(hi)
        self._limited = jp.asarray(hi > lo)
        self._init_qpos = jp.asarray(grounded_qpos(self.model))

        self._init = jax.jit(self._init_impl, static_argnums=0)
        self._step = jax.jit(self._step_impl)

    # control mapping identical to CreatureEnv._map_action (so policies transfer)
    def _map_action(self, a):
        a = jp.clip(jp.nan_to_num(a), -1.0, 1.0)
        ctrl = self._lo + (a + 1.0) * 0.5 * (self._hi - self._lo)
        return jp.where(self._limited, ctrl, a)

    def _init_impl(self, n_envs: int):
        def make(_):
            d = mjx.make_data(self.mx).replace(qpos=self._init_qpos)
            return mjx.forward(self.mx, d)
        return jax.vmap(make)(jp.arange(n_envs))

    def _step_impl(self, data, actions):
        ctrl = jax.vmap(self._map_action)(actions)

        def one(dx, c):
            dx = dx.replace(ctrl=c)
            return jax.lax.fori_loop(0, self.n_substeps,
                                     lambda _i, d: mjx.step(self.mx, d), dx)
        return jax.vmap(one)(data, ctrl)

    # public API
    def init(self, n_envs: int):
        """A batch of `n_envs` reset MJX Datas at the standing pose."""
        return self._init(int(n_envs))

    def step(self, data, actions):
        """Advance every env by `n_substeps` physics steps under `actions` (n_envs, nu)."""
        return self._step(data, jp.asarray(actions))

    def rollout(self, n_envs: int, n_steps: int, *, seed: int = 0):
        """Fully-jit'd scan of `n_steps` random-action control steps over `n_envs`
        parallel envs (for benchmarking). Returns the final batched Data."""
        actions = jax.random.uniform(jax.random.PRNGKey(seed),
                                     (n_steps, n_envs, self.model.nu), minval=-1.0, maxval=1.0)

        @jax.jit
        def run(data):
            def body(d, a):
                return self._step_impl(d, a), None
            final, _ = jax.lax.scan(body, data, actions)
            return final
        out = run(self.init(n_envs))
        jax.block_until_ready(out.qpos)
        return out

    @staticmethod
    def qpos(data) -> np.ndarray:
        return np.asarray(data.qpos)


# --- CPU<->MJX dynamics parity ---------------------------------------------
def _cpu_rollout(model, qpos0, ctrl, n_steps, n_substeps):
    d = mujoco.MjData(model)
    mujoco.mj_resetData(model, d)
    d.qpos[:] = qpos0
    mujoco.mj_forward(model, d)
    for _ in range(n_steps):
        d.ctrl[:] = ctrl
        for _ in range(n_substeps):
            mujoco.mj_step(model, d)
    return d.qpos.copy()


def _mjx_rollout(model, qpos0, ctrl, n_steps, n_substeps):
    mx = mjx.put_model(model)
    dx = mjx.make_data(mx).replace(qpos=jp.asarray(qpos0), ctrl=jp.asarray(ctrl))
    step = jax.jit(mjx.step)
    for _ in range(n_steps * n_substeps):
        dx = step(mx, dx)
    return np.asarray(dx.qpos)


def dynamics_parity(model, *, qpos0=None, ctrl=None, n_steps: int = 20,
                    n_substeps: int = 1) -> float:
    """Max |Δqpos| between the CPU C-engine and MJX after the same rollout on `model`.
    Tight in contact-free regimes (port correctness); grows under contact (known MJX
    soft-contact divergence)."""
    _require()
    if qpos0 is None:
        qpos0 = grounded_qpos(model)
    if ctrl is None:
        ctrl = np.zeros(model.nu)
    cpu = _cpu_rollout(model, qpos0, ctrl, n_steps, n_substeps)
    mjxq = _mjx_rollout(model, qpos0, ctrl, n_steps, n_substeps)
    return float(np.max(np.abs(cpu - mjxq)))
