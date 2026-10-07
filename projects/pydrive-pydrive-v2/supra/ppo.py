"""
PPO trainer + actor-critic policy (PyTorch).

Standard PPO: GAE advantages, a clipped surrogate objective, value loss and an
entropy bonus. Continuous Gaussian policy with a state-independent log-std.
Rollouts come from an in-process synchronous vector of SupraEnvs (multiprocessing
is a later optimisation). Time-limit truncation is bootstrapped correctly (the
truncated step's value estimate is folded into its reward), separate from real
terminations (crash -> no bootstrap).

A curriculum raises track difficulty once the rolling lap-completion clears a
threshold. The policy is mode-conditioned via the env's observation, so the same
class trains race now and drift / the hybrid later.
"""
from __future__ import annotations

from collections import deque
import copy
from dataclasses import asdict
import hashlib
import os
import sys
import threading
import time

import numpy as np
import torch
import torch.nn as nn

from .config import PPOSpec, RaceReward, SensorSpec, SimSpec
from .ppo_env import RunningNorm, SubprocVecEnv, SupraEnv
from .race_env import RaceEnv

DEVICE = torch.device("cpu")


def _require_finite(name, value):
    """Fail before a numerical incident can reach Adam or a checkpoint."""
    if torch.is_tensor(value):
        ok = bool(torch.isfinite(value).all())
    else:
        ok = bool(np.all(np.isfinite(np.asarray(value))))
    if not ok:
        raise FloatingPointError(f"non-finite {name}")


def state_dict_hash(state_dict) -> str:
    """Stable tensor hash used by checkpoints and evaluation provenance."""
    h = hashlib.sha256()
    for name, tensor in sorted(state_dict.items()):
        if not torch.is_tensor(tensor):
            raise ValueError(f"state_dict entry {name!r} is not a tensor")
        arr = tensor.detach().cpu().contiguous().numpy()
        h.update(name.encode("utf-8"))
        h.update(str(arr.shape).encode("ascii"))
        h.update(arr.tobytes())
    return h.hexdigest()


def _nested_tensors_finite(value) -> bool:
    if torch.is_tensor(value):
        return bool(torch.isfinite(value).all())
    if isinstance(value, dict):
        return all(_nested_tensors_finite(v) for v in value.values())
    if isinstance(value, (list, tuple)):
        return all(_nested_tensors_finite(v) for v in value)
    if isinstance(value, (float, np.floating)):
        return bool(np.isfinite(value))
    return True


def validate_checkpoint_payload(payload: dict, *, require_hash: bool = False) -> None:
    """Reject a checkpoint that could silently poison policy execution."""
    state = payload.get("state_dict")
    if not isinstance(state, dict) or not state:
        raise ValueError("missing state_dict")
    if not _nested_tensors_finite(state):
        raise ValueError("non-finite state_dict")
    actual_hash = state_dict_hash(state)
    stored_hash = payload.get("policy_sha256")
    if require_hash and not stored_hash:
        raise ValueError("missing policy_sha256")
    if stored_hash and stored_hash != actual_hash:
        raise ValueError("policy_sha256 does not match state_dict")

    sdim = int(payload.get("sdim", 0))
    mean = np.asarray(payload.get("norm_mean"), dtype=np.float64)
    var = np.asarray(payload.get("norm_var"), dtype=np.float64)
    count = float(payload.get("norm_count", 0.0))
    if (sdim <= 0 or mean.shape != (sdim,) or var.shape != (sdim,)
            or not np.all(np.isfinite(mean))
            or not np.all(np.isfinite(var)) or not np.all(var > 0.0)
            or not np.isfinite(count) or count <= 0.0):
        raise ValueError("invalid observation normalizer")

    for key, lower_inclusive in (("current_lr", False),
                                 ("current_ent_coef", True),
                                 ("lr_safety_scale", False)):
        if key not in payload:
            continue
        value = float(payload[key])
        if (not np.isfinite(value)
                or (value < 0.0 if lower_inclusive else value <= 0.0)):
            raise ValueError(f"invalid {key}")
    caps = payload.get("action_log_std_max")
    if caps is not None and not np.all(np.isfinite(np.asarray(caps, float))):
        raise ValueError("invalid action_log_std_max")
    if "opt" in payload and not _nested_tensors_finite(payload["opt"]):
        raise ValueError("non-finite optimizer state")


# --------------------------------------------------------------------------- #
class ActorCritic(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden=(128, 128), init_log_std=-0.5,
                 action_bias=None, action_log_std_max=None):
        super().__init__()
        layers, last = [], obs_dim
        for h in hidden:
            layers += [nn.Linear(last, h), nn.Tanh()]
            last = h
        self.trunk = nn.Sequential(*layers)
        self.mean = nn.Linear(last, act_dim)
        self.value = nn.Linear(last, 1)
        self.log_std = nn.Parameter(torch.ones(act_dim) * init_log_std)
        caps = action_log_std_max
        if caps is not None and len(caps) != act_dim:
            raise ValueError(f"action_log_std_max has {len(caps)} entries for "
                             f"{act_dim} actions")
        # Plain attribute (not a state_dict buffer) keeps every historical
        # checkpoint load-compatible. Checkpoints persist the caps separately.
        self.action_log_std_max = (tuple(float(x) for x in caps)
                                   if caps is not None else None)
        # small weights + a forward bias so the UNtrained policy drives forward
        # (else it averages throttle/brake to ~0, stalls, and never discovers
        # that moving earns reward — same trick as the GA's seeded throttle bias).
        self.mean.weight.data.mul_(0.01)
        self.mean.bias.data.zero_()
        self.mean.bias.data[1] = 0.6          # longitudinal -> ~60% throttle
        if act_dim >= 3:
            self.mean.bias.data[2] = -1.0     # handbrake off by default
        if action_bias is not None:           # explicit semantics override
            for i, v in enumerate(action_bias[:act_dim]):
                self.mean.bias.data[i] = float(v)

    def forward(self, obs):
        h = self.trunk(obs)
        return self.mean(h), self.value(h).squeeze(-1)

    # bound the action noise: without this the entropy bonus drives log_std
    # unboundedly upward (entropy exploded to 4+ in long runs), so the policy
    # never sharpens. std in [~0.11, ~1.0].
    LOG_STD_MIN, LOG_STD_MAX = -2.2, 0.0

    def _log_std_upper(self):
        if self.action_log_std_max is None:
            return torch.full_like(self.log_std, self.LOG_STD_MAX)
        return torch.as_tensor(self.action_log_std_max, dtype=self.log_std.dtype,
                               device=self.log_std.device)

    @torch.no_grad()
    def clamp_log_std_(self):
        self.log_std.data.clamp_(min=self.LOG_STD_MIN)
        self.log_std.data.copy_(torch.minimum(self.log_std.data,
                                               self._log_std_upper()))

    def _dist(self, mean):
        bounded = torch.maximum(self.log_std,
                                torch.full_like(self.log_std, self.LOG_STD_MIN))
        bounded = torch.minimum(bounded, self._log_std_upper())
        std = bounded.exp()
        return torch.distributions.Normal(mean, std)

    @torch.no_grad()
    def act(self, obs):
        mean, value = self(obs)
        dist = self._dist(mean)
        a = dist.sample()
        return a, dist.log_prob(a).sum(-1), value

    @torch.no_grad()
    def act_mean(self, obs):
        mean, _ = self(obs)
        return mean

    def evaluate(self, obs, act):
        mean, value = self(obs)
        dist = self._dist(mean)
        return dist.log_prob(act).sum(-1), dist.entropy().sum(-1), value


# --------------------------------------------------------------------------- #
class SyncVecEnv:
    """A list of SupraEnvs stepped together; auto-resets on episode end and
    reports the pre-reset final obs (for truncation bootstrapping)."""
    def __init__(self, n, env_cls=SupraEnv, **kw):
        self.envs = [env_cls(rng_seed=i, **kw) for i in range(n)]
        self.n = n
        self.obs_dim = self.envs[0].obs_dim
        self.act_dim = self.envs[0].action_dim
        self.sensor_dim = self.envs[0].sensor_dim

    def reset(self):
        return np.stack([e.reset() for e in self.envs])

    def step(self, actions):
        nobs, rew, term, trunc, final, infos = [], [], [], [], [], []
        for e, a in zip(self.envs, actions):
            o, r, te, tr, info = e.step(a)
            final.append(o)
            if te or tr:
                o = e.reset()
            nobs.append(o)
            rew.append(r); term.append(te); trunc.append(tr); infos.append(info)
        return (np.stack(nobs), np.array(rew, np.float32),
                np.array(term, bool), np.array(trunc, bool),
                np.stack(final), infos)

    def set_difficulty(self, d):
        for e in self.envs:
            e.difficulty = d

    def set_fixed_track(self, track):
        """Pin every env to one fixed track (used when a generalist graduates to
        fine-tuning on a target track)."""
        for e in self.envs:
            e.fixed_track = track

    def close(self):
        pass        # in-process: nothing to tear down (parallels SubprocVecEnv.close)


# --------------------------------------------------------------------------- #
class PPO:
    def __init__(self, mode="race", car="supra", ppo: PPOSpec | None = None,
                 reward: RaceReward | None = None, sim: SimSpec | None = None,
                 fixed_track=None, track_name=None,
                 target_track=None, target_name=None,
                 opponent_configs: list[dict] | None = None,
                 multiagent=False, n_agents=4,
                 env_cls_override=None, env_kwargs: dict | None = None,
                 eval_callback=None, extra_metadata=None):
        self.cfg = ppo or PPOSpec()
        self.mode = mode
        self.car = car
        # specialist: every env trains on ONE fixed track (curriculum off).
        # generalist: fixed_track is None -> the curriculum track pool.
        self.fixed_track = fixed_track
        self.specialist = fixed_track is not None
        self.track_name = track_name
        # graduate-to-target: a generalist that, once the curriculum maxes out,
        # converts into a specialist on this named track (wide->tight->target).
        self.target_track = target_track
        self.target_name = target_name
        self.graduated = False
        
        self.reward_cfg = reward
        self.sim_cfg = sim
        self.opponent_configs = opponent_configs
        self.multiagent = multiagent
        self.n_agents = n_agents
        self.eval_callback = eval_callback
        self.extra_metadata = extra_metadata or {}
        self.env_kwargs = dict(env_kwargs or {})
        
        vkw = dict(car=car, ppo=self.cfg, reward=reward, sim=sim, fixed_track=fixed_track)
        
        if env_cls_override is not None:
            self.env_cls = env_cls_override
            vkw["mode"] = mode
            vkw.update(self.env_kwargs)
            sync_cls = SyncVecEnv
            subproc_cls = SubprocVecEnv
        elif multiagent:
            from .multi_env import MultiAgentRaceEnv, SyncMultiVecEnv, SubprocMultiVecEnv
            self.env_cls = MultiAgentRaceEnv
            vkw["n_agents"] = n_agents
            sync_cls = SyncMultiVecEnv
            subproc_cls = SubprocMultiVecEnv
        else:
            self.env_cls = RaceEnv if opponent_configs else SupraEnv
            vkw["mode"] = mode
            if opponent_configs:
                vkw["opponent_configs"] = opponent_configs
                vkw.pop("mode", None)
            sync_cls = SyncVecEnv
            subproc_cls = SubprocVecEnv

        nw = max(1, int(getattr(self.cfg, "n_workers", 1)))
        if nw > 1 and self.cfg.n_envs % nw == 0:
            try:
                self.vec = subproc_cls(self.cfg.n_envs, nw, env_cls=self.env_cls, **vkw)
                # the policy net is tiny — 8 torch threads just fight the env
                # workers for cores; 2 is slightly faster AND frees cores for them.
                torch.set_num_threads(2)
                print(f"[workers] parallel rollouts across {nw} subprocesses "
                      f"({self.cfg.n_envs // nw} envs each), torch threads=2", flush=True)
            except Exception as e:      # any spawn/pickle issue -> in-process
                print(f"[workers] subprocess vec failed ({e}); using in-process", flush=True)
                self.vec = sync_cls(self.cfg.n_envs, env_cls=self.env_cls, **vkw)
        else:
            self.vec = sync_cls(self.cfg.n_envs, env_cls=self.env_cls, **vkw)
        self.obs_dim = self.vec.obs_dim
        self.act_dim = self.vec.act_dim
        self.sdim = self.vec.sensor_dim
        self.net = ActorCritic(self.obs_dim, self.act_dim, self.cfg.hidden,
                               self.cfg.init_log_std,
                               action_bias=getattr(self.cfg, "action_bias", None),
                               action_log_std_max=getattr(
                                   self.cfg, "action_log_std_max", None),
                               ).to(DEVICE)
        self.net.clamp_log_std_()
        self.opt = torch.optim.Adam(self.net.parameters(), lr=self.cfg.lr)
        self.norm = RunningNorm(self.sdim)
        # raw-v2 updates from RAW sensor samples only, after the whole PPO update
        # has used one immutable normalizer snapshot. Historical checkpoints used
        # a self-referential normalized-sample update and are frozen on load so a
        # migration cannot silently change the policy's input coordinates.
        self.norm_update_mode = "raw-v2"
        self._pending_norm_obs = None
        # for a specialist the track never changes, so show its inherent
        # difficulty rather than the (meaningless) curriculum level.
        self.difficulty = (float(getattr(fixed_track, "difficulty",
                                         self.cfg.start_difficulty))
                           if self.specialist else self.cfg.start_difficulty)
        self.ep_returns = deque(maxlen=100)
        self.ep_laps = deque(maxlen=100)
        self.ep_drift = deque(maxlen=100)
        self.updates = 0
        self.history = []        # (updates, ret_mean, lap_mean, difficulty)
        self.resumed_metric = -1e9   # set by load_state -> keep-best floor on resume
        self._ent_coef = self.cfg.ent_coef   # current (maybe annealed) entropy weight
        self._restart_it = 0                  # iter the current anneal segment began;
                                              # bumped on a plateau reseed so the LR/
                                              # entropy schedule re-explores from there
        self._schedule_position = 0
        self._resume_lr_cap = None
        self._resume_ent_cap = None
        self._lr_safety_scale = 1.0
        self._kl_rejections = 0
        self._numerical_recoveries = 0
        self._ep_ret_vec = np.zeros(self.cfg.n_envs, dtype=np.float64)
        self._env_error_total = 0
        self._env_error_window = deque(maxlen=32)
        self._env_error_streak_vec = np.zeros(self.cfg.n_envs, dtype=np.int16)

    # ------------------------------------------------------------------ #
    def _norm(self, obs_np):
        """Normalise the sensor block, leave the mode one-hot raw."""
        s = self.norm.normalize(obs_np[..., :self.sdim])
        return np.concatenate([s, obs_np[..., self.sdim:]], axis=-1).astype(np.float32)

    def _t(self, x):
        return torch.as_tensor(x, dtype=torch.float32, device=DEVICE)

    # ------------------------------------------------------------------ #
    def collect(self, obs):
        cfg = self.cfg
        T, N = cfg.rollout, self.vec.n
        b_obs = np.zeros((T, N, self.obs_dim), np.float32)
        b_act = np.zeros((T, N, self.act_dim), np.float32)
        b_logp = np.zeros((T, N), np.float32)
        b_val = np.zeros((T, N), np.float32)
        b_rew = np.zeros((T, N), np.float32)
        b_done = np.zeros((T, N), np.float32)
        b_raw_sensor = np.zeros((T, N, self.sdim), np.float64)
        ep_ret = (self._ep_ret_vec.copy()
                  if getattr(self, "_ep_ret_vec", np.empty(0)).shape == (N,)
                  else np.zeros(N, dtype=np.float64))

        for t in range(T):
            _require_finite("raw observations", obs)
            b_raw_sensor[t] = obs[..., :self.sdim]
            nobs = self._norm(obs)
            _require_finite("normalized observations", nobs)
            a, logp, val = self.net.act(self._t(nobs))
            a_np = a.cpu().numpy()
            _require_finite("sampled actions", a_np)
            _require_finite("action log-probabilities", logp)
            _require_finite("rollout values", val)
            next_obs, rew, term, trunc, final, infos = self.vec.step(a_np)
            _require_finite("environment observations", next_obs)
            _require_finite("environment rewards", rew)

            # A worker deliberately converts an isolated env exception into a
            # terminal transition so one edge case does not kill the night. But
            # a deterministic exception on every reset must become LOUD instead
            # of producing finite zero-reward rollouts/checkpoint heartbeats
            # forever. Three consecutive errors in one slot, or >5% across a
            # short window, escalates to the outer crash supervisor.
            bad = np.asarray([
                isinstance(info, dict)
                and info.get("termination_reason") in ("env_error", "worker_died")
                for info in infos
            ], dtype=bool)
            if getattr(self, "_env_error_streak_vec", np.empty(0)).shape != (N,):
                self._env_error_streak_vec = np.zeros(N, dtype=np.int16)
            self._env_error_streak_vec = np.where(
                bad, self._env_error_streak_vec + 1, 0).astype(np.int16)
            self._env_error_total += int(bad.sum())
            self._env_error_window.append(float(bad.mean()))
            chronic = (int(self._env_error_streak_vec.max(initial=0)) >= 3
                       or (len(self._env_error_window) >= 8
                           and float(np.mean(self._env_error_window)) > 0.05))
            if chronic:
                detail = "; ".join(str(info.get("env_error")
                                               or info.get("termination_reason"))
                                   for info, is_bad in zip(infos, bad)
                                   if is_bad and isinstance(info, dict))
                raise RuntimeError(
                    f"chronic environment failures: {int(bad.sum())}/{N} this "
                    f"step, total {self._env_error_total}; {detail[:500]}")

            done = term | trunc
            # truncation bootstrap: fold gamma * V(final_obs) into reward
            if trunc.any():
                idx = np.where(trunc & ~term)[0]
                if len(idx):
                    with torch.no_grad():
                        _, vf = self.net(self._t(self._norm(final[idx])))
                    rew[idx] = rew[idx] + cfg.gamma * vf.cpu().numpy()

            b_obs[t] = nobs
            b_act[t] = a_np
            b_logp[t] = logp.cpu().numpy()
            b_val[t] = val.cpu().numpy()
            b_rew[t] = rew
            b_done[t] = done.astype(np.float32)

            ep_ret += rew
            for i in range(N):
                if done[i]:
                    self.ep_returns.append(float(ep_ret[i]))
                    self.ep_laps.append(float(min(infos[i]["laps"], 1.0)))
                    if "drift_frac" in infos[i]:
                        self.ep_drift.append(float(infos[i]["drift_frac"]))
                    ep_ret[i] = 0.0
            obs = next_obs

        self._ep_ret_vec = ep_ret
        # Do NOT mutate the normalizer yet: GAE's last-state bootstrap and the
        # optimiser must see exactly the same coordinate system as collection.
        # update() commits these RAW samples only after the PPO transaction.
        self._pending_norm_obs = b_raw_sensor.reshape(-1, self.sdim)
        return obs, (b_obs, b_act, b_logp, b_val, b_rew, b_done)

    # ------------------------------------------------------------------ #
    def gae(self, b_val, b_rew, b_done, last_obs):
        cfg = self.cfg
        T, N = b_rew.shape
        with torch.no_grad():
            _, last_v = self.net(self._t(self._norm(last_obs)))
        last_v = last_v.cpu().numpy()
        adv = np.zeros((T, N), np.float32)
        lastgae = np.zeros(N, np.float32)
        # collect() stores b_done[t]/b_val[t]/b_rew[t] all at the SAME step t
        # (b_done[t] = the episode ended on step t's transition). So the bootstrap
        # mask for step t is (1 - b_done[t]) — NOT b_done[t+1]. The old code masked
        # with b_done[t+1], which shifted every terminal one step early AND double-
        # bootstrapped on truncation (whose reward is already augmented with
        # gamma*V(final) in collect()). Verified against hand-computed GAE.
        for t in reversed(range(T)):
            nextnonterm = 1.0 - b_done[t]
            nextval = last_v if t == T - 1 else b_val[t + 1]
            delta = b_rew[t] + cfg.gamma * nextval * nextnonterm - b_val[t]
            lastgae = delta + cfg.gamma * cfg.gae_lambda * nextnonterm * lastgae
            adv[t] = lastgae
        ret = adv + b_val
        _require_finite("GAE advantages", adv)
        _require_finite("GAE returns", ret)
        return adv, ret

    # ------------------------------------------------------------------ #
    def update(self, batch, adv, ret):
        cfg = self.cfg
        b_obs, b_act, b_logp, b_val, _, _ = batch
        for name, value in (("update observations", b_obs),
                            ("update actions", b_act),
                            ("old log-probabilities", b_logp),
                            ("advantages", adv), ("returns", ret),
                            ("old values", b_val)):
            _require_finite(name, value)
        T, N = adv.shape
        obs = self._t(b_obs.reshape(-1, self.obs_dim))
        act = self._t(b_act.reshape(-1, self.act_dim))
        old_logp = self._t(b_logp.reshape(-1))
        adv_f = self._t(adv.reshape(-1))
        ret_f = self._t(ret.reshape(-1))
        val_old = self._t(b_val.reshape(-1))
        adv_f = (adv_f - adv_f.mean()) / (adv_f.std() + 1e-8)

        n = obs.shape[0]
        mb = max(1, n // cfg.minibatches)
        idx = np.arange(n)
        stats = {"pi": 0.0, "vf": 0.0, "ent": 0.0}
        # trust region (opt-in via cfg.target_kl): approx-KL early stop. The
        # clip bounds each minibatch's OBJECTIVE, but epochs x minibatches of
        # clipped steps can still drag the policy arbitrarily far from the one
        # that collected the rollout — stop the update once it has moved.
        target_kl = getattr(cfg, "target_kl", None)
        # Fable's trust-region update is transactional. The old pre-step check
        # noticed an overshoot only on the NEXT minibatch, after the destructive
        # step had already landed. Snapshot once, measure the full rollout after
        # the update, and reject the whole transaction if it crossed the guard.
        net_before = copy.deepcopy(self.net.state_dict()) if target_kl is not None else None
        opt_before = copy.deepcopy(self.opt.state_dict()) if target_kl is not None else None

        def restore_transaction():
            if net_before is not None:
                self.net.load_state_dict(net_before)
                self.opt.load_state_dict(opt_before)

        kl_last, steps, stop = 0.0, 0, False
        for _ in range(cfg.epochs):
            np.random.shuffle(idx)
            for s in range(0, n, mb):
                j = idx[s:s + mb]
                logp, ent, val = self.net.evaluate(obs[j], act[j])
                logratio = logp - old_logp[j]
                ratio = logratio.exp()
                if target_kl is not None:
                    with torch.no_grad():
                        # low-variance estimator: E[(r-1) - log r] >= 0
                        kl_last = float(((ratio - 1.0) - logratio).mean())
                    if kl_last > 1.5 * target_kl:
                        stop = True
                        break
                a = adv_f[j]
                p1 = ratio * a
                p2 = torch.clamp(ratio, 1 - cfg.clip, 1 + cfg.clip) * a
                pi_loss = -torch.min(p1, p2).mean()
                vf_clip = getattr(cfg, "vf_clip", None)
                if vf_clip is not None:
                    # PPO2 value clip: past vf_clip of movement from the
                    # rollout's prediction, only the (larger) unclipped error
                    # still pulls — a single spiky return can't slingshot V
                    # (and, through the shared trunk, the policy) in one update.
                    v_clip = val_old[j] + torch.clamp(val - val_old[j],
                                                      -vf_clip, vf_clip)
                    vf_loss = torch.max((val - ret_f[j]) ** 2,
                                        (v_clip - ret_f[j]) ** 2).mean()
                else:
                    vf_loss = ((val - ret_f[j]) ** 2).mean()
                ent_loss = ent.mean()
                loss = pi_loss + cfg.vf_coef * vf_loss - self._ent_coef * ent_loss
                _require_finite("PPO loss", loss)
                self.opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), cfg.max_grad_norm)
                for p in self.net.parameters():
                    if p.grad is not None:
                        _require_finite("PPO gradients", p.grad)
                self.opt.step()
                if hasattr(self.net, "clamp_log_std_"):
                    self.net.clamp_log_std_()
                for p in self.net.parameters():
                    _require_finite("policy parameters", p)
                stats["pi"] += pi_loss.item()
                stats["vf"] += vf_loss.item()
                stats["ent"] += ent_loss.item()
                steps += 1
            if stop:
                break

        rejected = False
        if target_kl is not None:
            with torch.no_grad():
                new_logp, _, _ = self.net.evaluate(obs, act)
                full_logratio = new_logp - old_logp
                full_ratio = full_logratio.exp()
                full_kl_t = ((full_ratio - 1.0) - full_logratio).mean()
                kl_last = float(full_kl_t) if torch.isfinite(full_kl_t) else float("inf")
            if not np.isfinite(kl_last) or kl_last > 1.5 * float(target_kl):
                restore_transaction()
                rejected = True
                stop = True
                self._kl_rejections = int(getattr(self, "_kl_rejections", 0)) + 1
                self._lr_safety_scale = max(
                    0.10, float(getattr(self, "_lr_safety_scale", 1.0)) * 0.80)
                for g in self.opt.param_groups:
                    g["lr"] *= 0.80

        pending = getattr(self, "_pending_norm_obs", None)
        if pending is not None and hasattr(self, "norm"):
            # Normalization is part of the effective policy. A rejected trust-
            # region transaction must restore it too, not move the next rollout
            # under a coordinate system the KL guard never measured.
            if (not rejected
                    and getattr(self, "norm_update_mode", "raw-v2") == "raw-v2"):
                self.norm.update(pending)
            self._pending_norm_obs = None

        out = {kk: vv / max(1, steps) for kk, vv in stats.items()}
        out["kl"] = kl_last
        out["kl_stop"] = stop
        out["kl_reject"] = rejected
        out["accepted"] = not rejected
        with torch.no_grad():
            denom = float(torch.var(ret_f))
            out["explained_var"] = (float(1.0 - torch.var(ret_f - val_old) / denom)
                                    if denom > 1e-12 else 0.0)
        return out

    # ------------------------------------------------------------------ #
    def set_schedule(self, it, total):
        """Anneal learning rate + entropy over a run (cosine, down to the floor
        fracs) so the policy explores early and sharpens/refines late. No-op when
        cfg.anneal is False. Called once per iteration by headless + live training."""
        import math
        self._schedule_position = int(it)
        # anneal over the segment SINCE the last plateau reseed, so each restart
        # re-opens exploration (cosine back to 1.0) and re-sharpens over what's left.
        off = getattr(self, "_restart_it", 0)
        if getattr(self.cfg, "anneal", False):
            prog = min(1.0, max(0, it - off) / max(1, total - 1 - off))
            cos = 0.5 * (1.0 + math.cos(math.pi * prog))     # 1 -> 0
        else:
            cos = 1.0
        safety = float(getattr(self, "_lr_safety_scale", 1.0))
        lr = self.cfg.lr * (
            self.cfg.lr_floor_frac + (1 - self.cfg.lr_floor_frac) * cos
        ) * safety
        cap = getattr(self, "_resume_lr_cap", None)
        if cap is not None:
            lr = min(lr, float(cap))
        for g in self.opt.param_groups:
            g["lr"] = lr
        ent = self.cfg.ent_coef * (
            self.cfg.ent_floor_frac + (1 - self.cfg.ent_floor_frac) * cos)
        ent_cap = getattr(self, "_resume_ent_cap", None)
        if ent_cap is not None:
            ent = min(ent, float(ent_cap))
        self._ent_coef = ent

    # ------------------------------------------------------------------ #
    def _plateau_eligible(self):
        """Whether the plateau reseed / early-stop should apply. It kicks in once
        there's no more curriculum left to climb:
          * a specialist (one fixed track — e.g. a multi-self-play run pinned to a
            track), or
          * a generalist / true-multi run that has maxed the difficulty and has no
            pending graduate-to-target phase.
        Mid-curriculum runs keep climbing the difficulty instead of reseeding."""
        if self.specialist:
            return True
        maxed = self.difficulty >= self.cfg.max_difficulty - 1e-6
        pending_target = (self.target_track is not None and not self.graduated)
        return maxed and not pending_target

    def maybe_promote(self):
        # a specialist trains on one fixed track — there is no curriculum to
        # advance, so never promote (the track stays put regardless).
        if self.specialist:
            return False
        # promote on the metric that fits the task: lap completion for racing,
        # time-spent-drifting for drift (a drifter never completes laps fast).
        if self.mode == "drift":
            # advance only when it BOTH drifts enough AND gets around the loop —
            # so a slide-then-spin policy (drift high, laps ~0) never promotes
            # itself onto a harder track it can't complete.
            ready = (len(self.ep_drift) >= 30
                     and np.mean(self.ep_drift) > self.cfg.drift_promote_at
                     and len(self.ep_laps) >= 30
                     and np.mean(self.ep_laps) > self.cfg.drift_promote_lap)
        else:
            ready = (len(self.ep_laps) >= 30
                     and np.mean(self.ep_laps) > self.cfg.promote_at)
        if not ready:
            return False
        if self.difficulty < self.cfg.max_difficulty:
            self.difficulty = min(self.cfg.max_difficulty,
                                  self.difficulty + self.cfg.difficulty_step)
            self.vec.set_difficulty(self.difficulty)
            self.ep_laps.clear()
            self.ep_drift.clear()
            return True
        # curriculum maxed out and mastered -> if a target was set, GRADUATE into
        # a specialist on it (wide->tight->target), else stay at max.
        if self.target_track is not None and not self.graduated:
            self._graduate()
            return True
        return False

    def _graduate(self):
        import os
        import shutil
        # preserve the generalist peak before the target phase takes over _best
        bp = getattr(self, "_best_path", None)
        if bp and os.path.exists(bp):
            try:
                shutil.copy2(bp, bp.replace("_best.pt", "_generalist.pt"))
            except Exception:
                pass
        self.fixed_track = self.target_track
        self.track_name = self.target_name
        self.specialist = True          # curriculum off; early-stop now applies
        self.graduated = True
        self.vec.set_fixed_track(self.target_track)
        self._evalenv = None            # rebuild the eval env on the target track
        self.ep_laps.clear()
        self.ep_drift.clear()
        # CRITICAL: reset the keep-best floor. The generalist's score was earned on
        # easy curriculum tracks; the target (e.g. akina) scores far lower, so a
        # carried-over floor makes keep-best unbeatable and the early-stop fire
        # almost immediately. Judge the target phase on its OWN scale.
        self.best_metric = -1e9
        print(f"  [graduate] curriculum mastered -> fine-tuning on target "
              f"'{self.target_name}' (generalist peak saved as *_generalist.pt; "
              f"keep-best now tracks {self.target_name})", flush=True)

    # ------------------------------------------------------------------ #
    @staticmethod
    def best_path_for(checkpoint):
        """The keep-best sibling path for a run's checkpoint."""
        return (checkpoint[:-3] + "_best.pt" if checkpoint.endswith(".pt")
                else checkpoint + "_best.pt")

    def init_best_metric(self, best_path):
        """Seed the keep-best floor so a run never banks a '_best' worse than (a)
        a peak already saved under this name, or (b) the checkpoint it continued
        FROM. (b) is what makes 'continue from best, repeat' actually monotonic —
        otherwise resuming under a new --out name resets the bar to -inf and the
        first eval saves a worse 'best'."""
        import os
        file_best = -1e9
        if os.path.exists(best_path) and not getattr(self, "ignore_existing_best", False):
            try:
                file_best = float(torch.load(
                    best_path, map_location="cpu", weights_only=False).get("metric", -1e9))
            except Exception:
                file_best = -1e9
        self.best_metric = max(file_best, getattr(self, "resumed_metric", -1e9))
        return self.best_metric

    def eval_and_save(self, checkpoint, best_path):
        """Run the DETERMINISTIC full-episode eval (act_mean, clean start — i.e.
        exactly how you watch it), save the latest checkpoint, and keep-best to
        `best_path` whenever the composite score hits a new high. Prints the
        `[eval]` / `[best]` lines the dashboard parses. Returns True if a new
        best was banked. Shared by headless train() and the live (--live) view,
        so BOTH produce the honest metric and bank a peak."""
        if self.eval_callback is not None:
            result = self.eval_callback(self)
            self._last_eval_updates = int(getattr(self, "updates", 0))
            reported_metric = float(result.get("metric", 0.0))
            # Fable separates stage-objective selection from its human-readable
            # trend metric. A gated/clean-lap proof must never lose to a higher
            # but ungated composite score.
            self.metric = float(result.get("checkpoint_score", reported_metric))
            self.m_drift = float(result.get("drift", 0.0))
            self.m_laps = float(result.get("laps", 0.0))
            # Phase 1: bank the EXACT policy that was evaluated. The callback is
            # not allowed to mutate weights until after this transaction.
            self.save(checkpoint)
            print("  " + str(result.get("log", f"[eval-ring] metric={self.metric:.3f}")),
                  flush=True)
            improved = self.metric > getattr(self, "best_metric", -1e9)
            if improved:
                self.best_metric = self.metric
                self.save(best_path)
                tag = result.get("best_tag") or "[best-ring]"
                print(f"  {tag} metric={reported_metric:.3f} -> {best_path}",
                      flush=True)
                self._lr_safety_scale = min(
                    1.0, float(getattr(self, "_lr_safety_scale", 1.0)) * 1.05)
            # Phase 2: now (and only now) may a pit wall roll back/reseed. If it
            # changes the policy, persist the recovered state as the resumable
            # latest checkpoint; the protected best above remains provenance-true.
            after = getattr(self.eval_callback, "after_eval", None)
            if callable(after):
                post = after(self, result) or {}
                if bool(post.get("policy_changed")):
                    self.save(checkpoint)
            return improved

        ev_dr, ev_lp = self._deterministic_eval()
        self._last_eval_updates = int(getattr(self, "updates", 0))
        self.m_drift, self.m_laps = ev_dr, ev_lp
        if self.mode == "drift":
            self.metric = ev_dr * (0.3 + 0.7 * ev_lp)         # completion-gated drift
            etag = f"eval drift {ev_dr:.2f} x lap {ev_lp:.2f} = {self.metric:.3f}"
        elif self.mode == "hybrid":
            # completion-DOMINANT (must lap fast) + a style multiplier — so a fast
            # clean lap scores 0.6x and a fast DRIFTY lap scores up to 1.0x.
            self.metric = ev_lp * (0.6 + 0.4 * ev_dr)
            etag = f"eval lap {ev_lp:.2f} x style {ev_dr:.2f} = {self.metric:.3f}"
        else:
            # Race scoring is intentionally NOT capped at 1 lap. A clean but slow
            # lap and a superhuman two-lap run should not both look like "1.00" to
            # keep-best / plateau-reseed; cumulative laps over the fixed eval budget
            # is the deterministic lap-time proxy.
            self.metric = ev_lp
            etag = f"eval laps {ev_lp:.2f}"
        if getattr(self, "_ev_jumps", 0) > 0:
            etag += f"  jumps={self._ev_jumps} air={self._ev_air:.1f}s"
        self.save(checkpoint)
        print(f"  [eval] {etag}", flush=True)
        improved = self.metric > getattr(self, "best_metric", -1e9)
        if improved:
            self.best_metric = self.metric
            self.save(best_path)
            print(f"  [best] {etag} -> {best_path}", flush=True)
        return improved

    # ------------------------------------------------------------------ #
    def reseed_from_best(self, best_path, it, calm=False):
        """Plateau kick: reload the best weights and restart training FROM them with
        a fresh optimiser, re-widened policy noise, and re-opened exploration. A plain
        reload+continue would just retrace the same plateau — this injects new
        exploration so the policy can fall into a different (hopefully better) basin.
        The global keep-best floor (self.best_metric) is preserved, so a worse restart
        can never overwrite the saved peak.

        calm=True is the COLLAPSE-RECOVERY variant (Fable pit-wall ROLLBACK):
        restore the banked brain but do NOT re-widen the policy noise and do NOT
        re-open the LR/entropy anneal. Re-widening is right for escaping a
        plateau; it is exactly wrong for a knife-edge race policy that just fell
        off a cliff — wide noise makes the next rollout crash-dominated and the
        next update re-destroys the restored brain (the MANPLEASEWORK fast-stage
        loop: reseed -> one good eval -> dead again, 24x in one night)."""
        import os
        if not best_path or not os.path.exists(best_path):
            return False
        try:
            d = torch.load(best_path, map_location=DEVICE, weights_only=False)
            validate_checkpoint_payload(d)
            self.net.load_state_dict(d["state_dict"])
            if hasattr(self.net, "clamp_log_std_"):
                self.net.clamp_log_std_()
            self._policy_restored_from = os.path.basename(best_path)
            self._policy_anchor_hash = d.get("policy_sha256")
        except Exception as e:
            print(f"  [restart] could not reload best ({e}); skipping reseed", flush=True)
            return False
        # restore the obs normaliser the banked brain was EVALUATED under —
        # weights alone reproduce its behaviour only through the same input
        # scaling (the running norm keeps drifting while the policy wanders).
        try:
            if "norm_mean" in d:
                self.norm.mean = d["norm_mean"]
                self.norm.var = d["norm_var"]
                self.norm.count = d["norm_count"]
        except Exception:
            pass
        if not calm:
            # re-widen any policy dims that sharpened so the restart EXPLORES
            with torch.no_grad():
                self.net.log_std.data.clamp_(min=self.cfg.init_log_std)
                if hasattr(self.net, "clamp_log_std_"):
                    self.net.clamp_log_std_()
            self._ent_coef = self.cfg.ent_coef
            # PitWall historically passed global ppo.updates while set_schedule
            # uses a segment-local iteration, pinning resumed runs at full heat.
            # Anchor to the schedule's own coordinate system instead.
            self._restart_it = int(getattr(self, "_schedule_position", it))
            self._resume_lr_cap = None
            self._resume_ent_cap = None
        # drop stale Adam moments in both modes (they point back off the cliff)
        self.opt = torch.optim.Adam(
            self.net.parameters(),
            lr=self.opt.param_groups[0]["lr"] if calm else self.cfg.lr)
        return True

    # ------------------------------------------------------------------ #
    def train(self, iterations, checkpoint="ppo_race.pt", log_every=1,
              save_every=10, on_update=None):
        import signal
        # keep-best: alongside the latest checkpoint, save a separate "_best" one
        # whenever the eval composite score hits a new high — so a run can plateau
        # or even collapse without losing its peak.
        best_path = self.best_path_for(checkpoint)
        self._best_path = best_path          # so _graduate can preserve the peak
        self.init_best_metric(best_path)    # resuming: don't clobber a better peak
        # Save on all normal service-manager stop signals. SIGTERM/SIGHUP used
        # to bypass the only SIGINT checkpoint path.
        def _on_signal(signum, frame):
            print(f"\n[interrupted] saving checkpoint -> {checkpoint}", flush=True)
            if all(bool(torch.isfinite(p).all()) for p in self.net.parameters()):
                self.save(checkpoint)
            raise SystemExit(0)
        old_signal_handlers = {}
        try:
            for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
                old_signal_handlers[sig] = signal.getsignal(sig)
                signal.signal(sig, _on_signal)
        except (ValueError, OSError):
            pass            # not the main thread (e.g. live window) — skip
        obs = self.vec.reset()
        t0 = time.time()
        iters_since_best = 0
        patience = getattr(self.cfg, "specialist_patience", 0)
        max_restarts = int(getattr(self.cfg, "max_restarts", 0))
        restarts_done = 0        # CONSECUTIVE plateau reseeds since the last new best
        numerical_streak = 0
        eval_failures = 0
        eval_retry = False
        try:
            for it in range(iterations):
                self.set_schedule(it, iterations)
                try:
                    obs, batch = self.collect(obs)
                    adv, ret = self.gae(batch[3], batch[4], batch[5], obs)
                    st = self.update(batch, adv, ret)
                except FloatingPointError as e:
                    self._pending_norm_obs = None
                    numerical_streak += 1
                    self._numerical_recoveries += 1
                    self.cfg.lr = max(1e-6, float(self.cfg.lr) * 0.5)
                    recovered = (numerical_streak <= 3
                                 and self.reseed_from_best(best_path, it, calm=True))
                    print(f"  [health] numerical incident: {e}; "
                          + (f"restored {best_path} calmly, lr -> {self.cfg.lr:.2e}"
                             if recovered else "recovery unavailable"), flush=True)
                    if not recovered:
                        raise
                    obs = self.vec.reset()
                    continue
                numerical_streak = 0
                self.updates += 1
                was_grad = self.graduated
                promoted = self.maybe_promote()
                if self.graduated and not was_grad:
                    iters_since_best = 0
                if (it + 1) % log_every == 0:
                    ret_m = np.mean(self.ep_returns) if self.ep_returns else 0.0
                    steps = (it + 1) * self.cfg.rollout * self.cfg.n_envs
                    if self.mode == "drift":
                        dr = np.mean(self.ep_drift) if self.ep_drift else 0.0
                        metric = f"drift {dr:4.2f}"
                    else:
                        lap_m = np.mean(self.ep_laps) if self.ep_laps else 0.0
                        metric = f"laps {lap_m:4.2f}"
                    kl_s = (f"  kl {st['kl']:.4f}"
                            + ("!" if st.get("kl_reject") else
                               "*" if st.get("kl_stop") else "")
                            if getattr(self.cfg, "target_kl", None) else "")
                    print(f"it {it+1:4d}  ret {ret_m:7.1f}  {metric}  "
                          f"diff {self.difficulty:.2f}  pi {st['pi']:+.3f}  "
                          f"vf {st['vf']:.2f}  ent {st['ent']:.2f}{kl_s}  "
                          f"{steps/ (time.time()-t0):.0f} sps"
                          + ("  [KL-REJECT]" if st.get("kl_reject") else "")
                          + ("  [PROMOTED]" if promoted else ""), flush=True)
                if on_update:
                    on_update(self, it)
                iters_since_best += 1
                self.save(checkpoint)
                eval_due = (it + 1) % save_every == 0 or eval_retry
                if eval_due:
                    try:
                        improved = self.eval_and_save(checkpoint, best_path)
                        eval_failures = 0
                        eval_retry = False
                    except Exception as e:
                        eval_failures += 1
                        eval_retry = True
                        print(f"  [health] deterministic eval failed "
                              f"({eval_failures}/3): {type(e).__name__}: {e}",
                              flush=True)
                        if eval_failures >= 3:
                            raise
                        continue
                    if improved:
                        iters_since_best = 0
                        restarts_done = 0
                    if getattr(self, "request_stop", False):
                        print(f"  [gate] stage objective met — stopping early at "
                              f"it {it+1} (peak banked -> {best_path})", flush=True)
                        break
                    if (self._plateau_eligible() and patience > 0
                            and self.best_metric > -1e8
                            and iters_since_best >= patience):
                        if (restarts_done < max_restarts
                                and self.reseed_from_best(best_path, it)):
                            restarts_done += 1
                            iters_since_best = 0
                            print(f"  [restart] plateau: no new best in {patience} "
                                  f"iters -> reseed {restarts_done}/{max_restarts} "
                                  f"from best; best held at "
                                  f"{self.best_metric:.3f}", flush=True)
                        else:
                            why = (f"no new best after {max_restarts} reseed(s)"
                                   if max_restarts > 0
                                   else f"no eval gain in {patience} iters")
                            print(f"  [converged] specialist plateaued: {why} "
                                  f"(best {self.best_metric:.3f}). stopping at "
                                  f"it {it+1}. peak -> {best_path}", flush=True)
                            break
            self.save(checkpoint)
        finally:
            try:
                self.vec.close()
            except Exception:
                pass
            for sig, handler in old_signal_handlers.items():
                try:
                    signal.signal(sig, handler)
                except (ValueError, OSError):
                    pass
        return self

    # ------------------------------------------------------------------ #
    def _eval_env(self):
        """A dedicated env that mirrors WATCHING: clean start (no exploring
        starts), deterministic. Specialist -> its fixed track; generalist -> a
        fixed-seed curriculum track at the current difficulty (rebuilt on bump)."""
        import copy
        cur_diff = getattr(self, "difficulty", 0.0)
        if (getattr(self, "_evalenv", None) is None
                or (not self.specialist and self._eval_diff != cur_diff)):
            cfg = copy.copy(self.cfg)
            cfg.random_start = False                  # start at the line, like watch
            ft = self.fixed_track
            if ft is None:
                from .track import curriculum_track, drift_curriculum_track
                cur = drift_curriculum_track if self.mode == "drift" else curriculum_track
                ft = cur(cur_diff, seed=12345)
            vkw = dict(car=self.car, ppo=cfg, reward=self.reward_cfg, sim=self.sim_cfg)
            if self.multiagent:
                vkw["n_agents"] = self.n_agents
            elif self.env_cls is not RaceEnv:
                vkw["mode"] = self.mode
            else:
                vkw["opponent_configs"] = self.opponent_configs
            vkw.update(getattr(self, "env_kwargs", {}) or {})
            self._evalenv = self.env_cls(rng_seed=888, fixed_track=ft, **vkw)
            self._eval_diff = cur_diff
        return self._evalenv

    @torch.no_grad()
    def _deterministic_eval(self, n_starts=4, steps_per=None):
        """Deterministic, reproducible drift / lap measure: run act_mean from
        several fixed points around the track at ~racing pace and average. No
        exploration noise, no exploring-start spawns -> matches what you judge
        when WATCHING (cornering drift around the whole loop), so keep-best can
        trust it.

        Budget = a FULL episode (`episode_seconds`) per start. The old fixed
        14 s window (steps_per=420) capped 'laps' at ~0.2-0.4 on 1+ km tracks
        for ANY policy, however good — so every drift policy looked like it
        "couldn't complete the lap" when the eval simply stopped too early
        (a measurement artifact, not a policy failure). Episodes that spin out
        or run off terminate early, so a weak policy is still cheap to score;
        only a genuinely good one runs the whole budget."""
        env = self._eval_env()
        trk = env.trk
        M = len(trk.center)
        if steps_per is None:
            # full-episode budget so a complete lap is actually reachable; the
            # env truncates at episode_seconds regardless, this just doesn't
            # cut the run short before that.
            steps_per = int(round(self.cfg.episode_seconds * self.cfg.control_hz))
        drifts, laps = [], []
        air_time, jumps = 0.0, 0
        for s in range(n_starts):
            idx = int(M * s / n_starts)
            k = abs(float(trk.curvature[idx]))
            nat = min(30.0, (11.0 / k) ** 0.5) if k > 1e-4 else 30.0
            obs = env.reset_at(idx, speed=nat * 0.85)     # drop in at the pace
            info = {}
            # Measure cumulative progress from THIS start, not just the snapshot
            # 'laps' at the final step. For race policies this stays uncapped:
            # 2.20 laps in the fixed eval window is meaningfully better than 1.05.
            # Drift / hybrid still use capped completion as a gate for style.
            best_lap = 0.0
            for _ in range(steps_per):
                nobs = self._norm(obs[None])[0]
                a = self.net.act_mean(self._t(nobs).unsqueeze(0)).squeeze(0).cpu().numpy()
                obs, _, term, trunc, info = env.step(a)
                if self.multiagent:
                    best_lap = max(best_lap, max(i.get("laps", 0.0) for i in info))
                    if term.any() or trunc.any():
                        break
                else:
                    best_lap = max(best_lap, info.get("laps", 0.0))
                    if term or trunc:
                        break
                        
            if self.multiagent:
                drifts.append(np.mean([i.get("drift_frac", 0.0) for i in info]))
                laps.append(best_lap if self.mode == "race" else min(best_lap, 1.0))
                air_time += np.mean([i.get("airtime", 0.0) for i in info])
                jumps += np.mean([i.get("jumps", 0) for i in info])
            else:
                drifts.append(info.get("drift_frac", 0.0))
                laps.append(best_lap if self.mode == "race" else min(best_lap, 1.0))
                air_time += info.get("airtime", 0.0)
                jumps += info.get("jumps", 0)
        # stashed for the [eval] line — hills-era runs report their air
        self._ev_air, self._ev_jumps = air_time, jumps
        return float(np.mean(drifts)), float(np.mean(laps))

    # ------------------------------------------------------------------ #
    def policy_hash(self) -> str:
        """Stable weight hash binding an evaluation to one exact policy."""
        return state_dict_hash(self.net.state_dict())

    # ------------------------------------------------------------------ #
    def save(self, path):
        # write to a temp file then atomically rename, so a concurrent
        # --watch-ppo / --watch-drift can never load a half-written checkpoint
        tmp = f"{path}.tmp.{os.getpid()}.{threading.get_ident()}"
        pace_on = bool(getattr(self.cfg, "sensor_pace_block", False))
        policy_sha = self.policy_hash()
        payload = {
            "state_dict": self.net.state_dict(),
            "opt": self.opt.state_dict(),
            # hills-v1 = 58 sensors + 2 mode; fable-v1 appends the pace block
            "obs_layout": "fable-v1" if pace_on else "hills-v1",
            "sensor_pace_block": pace_on,
            "sensor_pace_distances": (
                list(getattr(self.cfg, "sensor_pace_distances", None) or [])
                or None),
            "obs_dim": self.obs_dim, "act_dim": self.act_dim,
            "hidden": list(self.cfg.hidden), "mode": self.mode, "car": self.car,
            "norm_mean": self.norm.mean, "norm_var": self.norm.var,
            "norm_count": self.norm.count, "sdim": self.sdim,
            "norm_schema": getattr(self, "norm_update_mode", "raw-v2"),
            "difficulty": self.difficulty, "updates": self.updates,
            "track": self.track_name,      # specialist track name, or None
            "track_profile": getattr(self.cfg, "track_profile", None),
            "track_metadata": getattr(self.fixed_track, "metadata", {}) if self.fixed_track is not None else {},
            "sensor_lookahead_distances": list(
                getattr(self.cfg, "sensor_lookahead_distances", None)
                or SensorSpec().lookahead_distances
            ),
            "train_arena": ("true_multi" if self.multiagent else
                            "frozen_opponents" if self.opponent_configs else "solo"),
            "n_agents": int(self.n_agents if self.multiagent else
                            1 + len(self.opponent_configs or [])),
            "opponent_checkpoints": [c.get("checkpoint") for c in (self.opponent_configs or [])],
            "metric": getattr(self, "metric", 0.0),   # composite best-score
            "drift": getattr(self, "m_drift", 0.0),   # raw rolling drift fraction
            "laps": getattr(self, "m_laps", 0.0),     # raw rolling lap completion
            "policy_sha256": policy_sha,
            "policy_restored_from": getattr(self, "_policy_restored_from", None),
            "policy_anchor_sha256": getattr(self, "_policy_anchor_hash", None),
            "ppo_config": asdict(self.cfg),
            "action_log_std_max": (list(getattr(self.cfg, "action_log_std_max", None))
                                   if getattr(self.cfg, "action_log_std_max", None)
                                   is not None else None),
            "current_lr": float(self.opt.param_groups[0]["lr"]),
            "current_ent_coef": float(getattr(self, "_ent_coef", self.cfg.ent_coef)),
            "lr_safety_scale": float(getattr(self, "_lr_safety_scale", 1.0)),
            "kl_rejections": int(getattr(self, "_kl_rejections", 0)),
            "numerical_recoveries": int(getattr(self, "_numerical_recoveries", 0)),
            "env_error_total": int(getattr(self, "_env_error_total", 0)),
            "checkpoint_time_unix": time.time(),
            "runtime_versions": {"python": sys.version.split()[0],
                                 "torch": torch.__version__,
                                 "numpy": np.__version__},
        }
        extra = self.extra_metadata() if callable(self.extra_metadata) else self.extra_metadata
        if extra:
            payload.update(dict(extra))
        torch.save(payload, tmp)
        durable = "_best" in os.path.basename(path) or "_hof_" in os.path.basename(path)
        if durable:
            with open(tmp, "rb") as fh:
                os.fsync(fh.fileno())
        os.replace(tmp, path)
        if durable:
            try:
                fd = os.open(os.path.dirname(os.path.abspath(path)) or ".", os.O_RDONLY)
                try:
                    os.fsync(fd)
                finally:
                    os.close(fd)
            except OSError:
                pass

    def load_state(self, path):
        """Resume a full training state (weights, optimizer, normalizer,
        curriculum) into this trainer. Architecture must match."""
        d = torch.load(path, map_location=DEVICE, weights_only=False)
        validate_checkpoint_payload(d)
        if d["obs_dim"] != self.obs_dim or d["act_dim"] != self.act_dim \
                or list(d["hidden"]) != list(self.cfg.hidden):
            hint = ("  [PRE-HILLS checkpoint — the obs vector grew 42 -> 60 "
                    "when hills/jumps landed; retrain it. See PHYSICS_3D_PLAN.md]"
                    if d.get("obs_layout") != "hills-v1" else "")
            raise ValueError(
                f"checkpoint arch (obs {d['obs_dim']}, act {d['act_dim']}, "
                f"hidden {d['hidden']}) != current "
                f"(obs {self.obs_dim}, act {self.act_dim}, "
                f"hidden {list(self.cfg.hidden)}){hint}")
        self.net.load_state_dict(d["state_dict"])
        # an old checkpoint may carry a blown-up log_std; clamp it into range
        self.net.clamp_log_std_()
        if "opt" in d:
            try:
                self.opt.load_state_dict(d["opt"])
            except Exception as e:
                # Optimizer state is best-effort (a changed arch/param-group
                # layout legitimately can't reload it), but swallowing it
                # silently also hides real checkpoint mismatches — say so.
                print(f"  [ppo] optimizer state not restored from checkpoint "
                      f"({type(e).__name__}: {e}); continuing with a fresh "
                      f"optimizer", flush=True)
        self.norm.mean = d["norm_mean"]
        self.norm.var = d["norm_var"]
        self.norm.count = d["norm_count"]
        schema = d.get("norm_schema")
        if schema == "raw-v2":
            self.norm_update_mode = "raw-v2"
        else:
            # The legacy algorithm trained the policy in a different coordinate
            # system. Freeze it rather than gradually (and destructively) morphing
            # tens of millions of historical samples into new statistics.
            self.norm_update_mode = "legacy-frozen-v1"
            print("  [ppo] legacy observation normalizer frozen; new checkpoints "
                  "use raw-v2 for fresh policies", flush=True)
        self.difficulty = float(d.get("difficulty", self.difficulty))
        self.vec.set_difficulty(self.difficulty)
        self.updates = int(d.get("updates", 0))
        self._resume_lr_cap = float(d.get(
            "current_lr", self.opt.param_groups[0].get("lr", self.cfg.lr)))
        self._resume_ent_cap = float(d.get("current_ent_coef", self.cfg.ent_coef))
        self._ent_coef = self._resume_ent_cap
        self._lr_safety_scale = float(d.get("lr_safety_scale", 1.0))
        self._kl_rejections = int(d.get("kl_rejections", 0))
        self._numerical_recoveries = int(d.get("numerical_recoveries", 0))
        self._env_error_total = int(d.get("env_error_total", 0))
        self._env_error_window.clear()
        self._env_error_streak_vec.fill(0)
        self._policy_restored_from = d.get("policy_restored_from")
        self._policy_anchor_hash = d.get("policy_anchor_sha256")
        # remember how good the checkpoint we're continuing FROM was, so keep-best
        # uses it as a floor — a continued run never banks a "_best" worse than the
        # one you continued from (true best-of-best chaining; see init_best_metric).
        # ONLY when the track matches: a generalist's score isn't comparable once
        # you warm-start it onto a different (harder) specialist track, so there we
        # leave the floor open and let the run bank its own bests from scratch.
        self._resumed_mode = d.get("mode")
        # only carry the keep-best floor when it's COMPARABLE: same track AND same
        # mode (a drift score isn't comparable to a hybrid score when you warm-start
        # a drift policy into a hybrid run).
        self.resumed_metric = (float(d.get("metric", -1e9) or -1e9)
                               if (d.get("track") == self.track_name
                                   and d.get("mode") == self.mode) else -1e9)
        return d

    @staticmethod
    def load_policy(path):
        d = torch.load(path, map_location=DEVICE, weights_only=False)
        validate_checkpoint_payload(d)
        # loud pre-hills guard: a 42-obs net fed today's 60-dim obs would die
        # in a cryptic matmul mid-drive — refuse up front with the real reason.
        # Layout-aware: reconstruct the sensor spec the checkpoint trained with
        # (fable-v1 pace block changes obs size) and compare against THAT.
        from .config import SensorSpec
        from .sensors import SensorSuite
        spec = SensorSpec()
        if d.get("sensor_pace_block"):
            spec.pace_block = True
            if d.get("sensor_pace_distances"):
                spec.pace_distances = tuple(d["sensor_pace_distances"])
        cur = SensorSuite(spec).obs_size + SupraEnv.MODE_DIM
        if d["obs_dim"] != cur:
            raise ValueError(
                f"checkpoint '{path}' has obs_dim {d['obs_dim']} but the "
                f"current sensor layout is {cur}"
                + ("  [PRE-HILLS checkpoint — the obs vector grew 42 -> 60 "
                   "when hills/jumps landed; retrain it. See PHYSICS_3D_PLAN.md]"
                   if d.get("obs_layout") != "hills-v1" else ""))
        net = ActorCritic(d["obs_dim"], d["act_dim"], tuple(d["hidden"]),
                          action_log_std_max=d.get("action_log_std_max"))
        net.load_state_dict(d["state_dict"])
        net.clamp_log_std_()
        net.eval()
        norm = RunningNorm(d["sdim"])
        norm.mean, norm.var, norm.count = d["norm_mean"], d["norm_var"], d["norm_count"]
        return net, norm, d
