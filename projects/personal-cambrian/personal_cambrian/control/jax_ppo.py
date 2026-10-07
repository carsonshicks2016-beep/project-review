"""JAX PPO + policy for GPU training (ROADMAP Stage 9.2).

A faithful JAX re-implementation of the CleanRL-style PPO in `control.ppo` (same
hyperparameters, GAE, clipped surrogate, per-minibatch advantage normalization, and
global-norm grad clipping) -- but operating on a VECTORIZED JAX env so the policy
forward/backward and the env step all run under `jit` on whatever accelerator JAX
sees. On a GPU the n_envs-parallel rollout + jit'd update is the "far faster" half of
Stage 9; on CPU it's verified to LEARN to the same quality as the torch PPO (the
done-when's testable half).

Deliberately dependency-light: raw JAX (params as pytrees, hand-rolled Adam) -- no
flax/optax -- so Stage 9 adds only jax/jaxlib/mujoco-mjx. Import-guarded behind
`jax_ppo_available()`.

A "JAX vec env" is any object exposing `n_envs`, `obs_dim`, `act_dim`, and pure
functions `reset(key) -> (state, obs[N,O])` and `step(state, act[N,A], key) ->
(state, obs[N,O], reward[N], done[N])` (auto-resetting finished envs). `MjxVecEnv`
adapts the Stage-9.1 `MjxBatchEnv` (flat proprioceptive obs + forward-velocity
reward) into that protocol.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

try:
    import jax
    import jax.numpy as jp
    _HAS = True
except Exception:        # noqa: BLE001
    _HAS = False


def jax_ppo_available() -> bool:
    return _HAS


_LOG2PI = 1.8378770664093453


# --- policy (MLP actor-critic, state-independent log-std) -------------------
def _init_mlp(key, sizes, *, final_std):
    import jax
    params = []
    for i, (din, dout) in enumerate(zip(sizes[:-1], sizes[1:])):
        key, k = jax.random.split(key)
        std = final_std if i == len(sizes) - 2 else jp.sqrt(2.0)
        W = jax.nn.initializers.orthogonal(std)(k, (din, dout))
        params.append((W, jp.zeros(dout)))
    return params


def init_params(key, obs_dim, act_dim, hidden=64):
    ka, kc = jax.random.split(key)
    return {
        "actor": _init_mlp(ka, [obs_dim, hidden, hidden, act_dim], final_std=0.01),
        "logstd": jp.zeros(act_dim),
        "critic": _init_mlp(kc, [obs_dim, hidden, hidden, 1], final_std=1.0),
    }


def _mlp(layers, x):
    for W, b in layers[:-1]:
        x = jp.tanh(x @ W + b)
    W, b = layers[-1]
    return x @ W + b


def forward(params, obs):
    """Return (action_mean[N,A], logstd[A], value[N])."""
    mean = _mlp(params["actor"], obs)
    value = _mlp(params["critic"], obs)[..., 0]
    return mean, params["logstd"], value


def _logprob(mean, logstd, action):
    var = jp.exp(2.0 * logstd)
    return -0.5 * (((action - mean) ** 2) / var + 2.0 * logstd + _LOG2PI).sum(-1)


def _entropy(logstd):
    return (logstd + 0.5 * (_LOG2PI + 1.0)).sum()


def policy_action(params, obs, key=None, *, deterministic=False):
    """Sample (or take the mean) action + its log-prob for a batch of obs."""
    mean, logstd, _ = forward(params, obs)
    if deterministic or key is None:
        return mean, _logprob(mean, logstd, mean)
    noise = jax.random.normal(key, mean.shape)
    action = mean + jp.exp(logstd) * noise
    return action, _logprob(mean, logstd, action)


# --- hand-rolled Adam over a pytree ----------------------------------------
def _adam_init(params):
    z = lambda p: jax.tree_util.tree_map(jp.zeros_like, p)
    return {"m": z(params), "v": z(params), "t": 0}


def _adam_step(params, grads, st, *, lr, b1=0.9, b2=0.999, eps=1e-5):
    t = st["t"] + 1
    m = jax.tree_util.tree_map(lambda mm, g: b1 * mm + (1 - b1) * g, st["m"], grads)
    v = jax.tree_util.tree_map(lambda vv, g: b2 * vv + (1 - b2) * g * g, st["v"], grads)
    mc = 1 - b1 ** t
    vc = 1 - b2 ** t
    params = jax.tree_util.tree_map(
        lambda p, mm, vv: p - lr * (mm / mc) / (jp.sqrt(vv / vc) + eps), params, m, v)
    return params, {"m": m, "v": v, "t": t}


def _global_norm(grads):
    leaves = jax.tree_util.tree_leaves(grads)
    return jp.sqrt(sum(jp.sum(g * g) for g in leaves))


def _clip_grads(grads, max_norm):
    norm = _global_norm(grads)
    factor = jp.minimum(1.0, max_norm / (norm + 1e-6))
    return jax.tree_util.tree_map(lambda g: g * factor, grads)


# --- GAE (jit) --------------------------------------------------------------
def _gae(rew, val, done, last_val, gamma, lam):
    T = rew.shape[0]

    def body(carry, t):
        lastgae, nextval, nextnonterm = carry
        delta = rew[t] + gamma * nextval * nextnonterm - val[t]
        lastgae = delta + gamma * lam * nextnonterm * lastgae
        return (lastgae, val[t], 1.0 - done[t]), lastgae

    nonterm_T = 1.0 - done[-1]
    init = (jp.zeros_like(rew[0]), last_val, nonterm_T)
    _, adv = jax.lax.scan(body, init, jp.arange(T - 1, -1, -1))
    adv = adv[::-1]
    return adv, adv + val


# --- config -----------------------------------------------------------------
@dataclass
class JaxPPOConfig:
    total_timesteps: int = 200_000
    n_steps: int = 256
    n_epochs: int = 4
    n_minibatches: int = 4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip: float = 0.2
    ent_coef: float = 0.0
    vf_coef: float = 0.5
    lr: float = 3e-4
    max_grad_norm: float = 0.5
    hidden: int = 64
    seed: int = 0


# --- training ---------------------------------------------------------------
def train(env, cfg: JaxPPOConfig = JaxPPOConfig(), *, log_fn=None, verbose=False):
    """Train a PPO policy on a JAX vec env. Returns (params, history).

    The rollout is a Python loop over `n_steps` (so even a slow MJX env works and no
    giant scan is compiled); the policy forward, GAE, and the epoch x minibatch update
    are jit'd. `env` auto-resets finished envs, so episodic returns are tracked from
    its `done` flags."""
    key = jax.random.PRNGKey(cfg.seed)
    key, kp, kr = jax.random.split(key, 3)
    params = init_params(kp, env.obs_dim, env.act_dim, cfg.hidden)
    opt = _adam_init(params)

    fwd = jax.jit(forward)
    act_fn = jax.jit(lambda p, o, k: policy_action(p, o, k))
    gae_fn = jax.jit(lambda r, v, d, lv: _gae(r, v, d, lv, cfg.gamma, cfg.gae_lambda))

    def loss_fn(p, obs, act, old_logp, adv, ret):
        mean, logstd, val = forward(p, obs)
        new_logp = _logprob(mean, logstd, act)
        ratio = jp.exp(new_logp - old_logp)
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        pg = jp.maximum(-adv * ratio,
                        -adv * jp.clip(ratio, 1 - cfg.clip, 1 + cfg.clip)).mean()
        v_loss = 0.5 * ((val - ret) ** 2).mean()
        return pg + cfg.vf_coef * v_loss - cfg.ent_coef * _entropy(logstd)

    @jax.jit
    def update_mb(p, opt, obs, act, old_logp, adv, ret):
        grads = jax.grad(loss_fn)(p, obs, act, old_logp, adv, ret)
        grads = _clip_grads(grads, cfg.max_grad_norm)
        return _adam_step(p, grads, opt, lr=cfg.lr)

    N, T = env.n_envs, cfg.n_steps
    state, obs = env.reset(kr)
    ep_ret = np.zeros(N)
    recent: list = []
    n_iters = max(1, cfg.total_timesteps // (N * T))
    history = []

    for it in range(n_iters):
        O, A, LP, V, R, D = [], [], [], [], [], []
        for _t in range(T):
            key, ka, ks = jax.random.split(key, 3)
            mean, logstd, val = fwd(params, obs)
            action, logp = act_fn(params, obs, ka)
            state, next_obs, rew, done = env.step(state, action, ks)
            O.append(obs); A.append(action); LP.append(logp); V.append(val)
            R.append(rew); D.append(done)
            r_np = np.asarray(rew); d_np = np.asarray(done)
            ep_ret += r_np
            for i in range(N):
                if d_np[i]:
                    recent.append(ep_ret[i]); ep_ret[i] = 0.0
            obs = next_obs

        last_val = fwd(params, obs)[2]
        obs_b = jp.stack(O); act_b = jp.stack(A); logp_b = jp.stack(LP)
        val_b = jp.stack(V); rew_b = jp.stack(R); done_b = jp.stack(D)
        adv, ret = gae_fn(rew_b, val_b, done_b, last_val)

        bs = T * N
        f_obs = obs_b.reshape(bs, env.obs_dim)
        f_act = act_b.reshape(bs, env.act_dim)
        f_logp = logp_b.reshape(bs)
        f_adv = adv.reshape(bs)
        f_ret = ret.reshape(bs)
        mb = bs // cfg.n_minibatches
        for _e in range(cfg.n_epochs):
            key, ksh = jax.random.split(key)
            idx = np.asarray(jax.random.permutation(ksh, bs))
            for s in range(0, bs, mb):
                j = idx[s:s + mb]
                params, opt = update_mb(params, opt, f_obs[j], f_act[j],
                                        f_logp[j], f_adv[j], f_ret[j])

        mean_ret = float(np.mean(recent[-100:])) if recent else float("nan")
        rec = {"iter": it, "global_step": (it + 1) * N * T, "mean_return": mean_ret}
        history.append(rec)
        if log_fn:
            log_fn(rec)
        if verbose:
            print(f"  iter {it:3d}  step {rec['global_step']:7d}  mean_return {mean_ret:8.2f}")
    return params, history


# --- MJX creature env adapter (the GPU training target) --------------------
class MjxVecEnv:
    """Adapts the Stage-9.1 `MjxBatchEnv` into the JAX vec-env protocol: flat
    proprioceptive obs (joint qpos/qvel, root x,y dropped for translation invariance)
    and a forward-velocity reward. A continuing task (no termination) -- enough to
    train locomotion and to prove the GPU training path end-to-end; episodic
    auto-reset is a later refinement."""

    def __init__(self, genome=None, *, model=None, n_envs: int = 64, n_substeps: int = 5):
        from ..sim.mjx_env import MjxBatchEnv
        self.batch = MjxBatchEnv(genome, model=model, n_substeps=n_substeps)
        self.model = self.batch.model
        self.n_envs = int(n_envs)
        self.act_dim = int(self.model.nu)
        self.obs_dim = int((self.model.nq - 2) + self.model.nv)
        self.control_dt = self.batch.control_dt
        self._obs = jax.jit(lambda d: jp.concatenate([d.qpos[:, 2:], d.qvel], axis=1))

    def reset(self, key):
        data = self.batch.init(self.n_envs)
        return (data, data.qpos[:, 0]), self._obs(data)

    def step(self, state, action, key):
        data, prev_x = state
        data = self.batch.step(data, action)
        x = data.qpos[:, 0]
        reward = (x - prev_x) / self.control_dt          # forward velocity
        done = jp.zeros(self.n_envs)                      # continuing task
        return (data, x), self._obs(data), reward, done


def evaluate(params, env, *, n_steps=200, key=None):
    """Mean episodic return of the deterministic policy on a JAX vec env."""
    if key is None:
        key = jax.random.PRNGKey(0)
    key, kr = jax.random.split(key)
    state, obs = env.reset(kr)
    ep_ret = np.zeros(env.n_envs)
    finished: list = []
    for _ in range(n_steps):
        key, ks = jax.random.split(key)
        action, _ = policy_action(params, obs, deterministic=True)
        state, obs, rew, done = env.step(state, action, ks)
        ep_ret += np.asarray(rew)
        for i in range(env.n_envs):
            if np.asarray(done)[i]:
                finished.append(ep_ret[i]); ep_ret[i] = 0.0
    pool = finished if finished else list(ep_ret)
    return float(np.mean(pool))
