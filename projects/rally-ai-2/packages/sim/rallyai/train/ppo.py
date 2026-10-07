"""PPO update with KL rollback, value clip, and minibatch advantage norm.

Hyperparameters are a starting point to measure, not a derivation. The KL
guard restores both parameters and optimiser state — one bad update must not
destroy a competent policy after a long run.
"""

from __future__ import annotations

import copy
import math
import signal
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from rallyai.env import EnvConfig, RewardConfig
from rallyai.sense import SensorSpec
from rallyai.train.checkpoint import (
    HallOfFame,
    build_payload,
    load_checkpoint,
    require_finite,
    save_checkpoint,
    state_dict_hash,
)
from rallyai.train.curriculum import Curriculum, CurriculumConfig
from rallyai.train.metrics import MetricsWriter
from rallyai.train.normalise import ObservationNormaliser
from rallyai.train.policy import ACT_DIM, OBS_DIM, ActorCritic
from rallyai.train.rollout import RolloutBuffer
from rallyai.train.stages import stage_reward_config, stage_tier_bounds
from rallyai.train.vec_env import AsyncVecEnv, SyncVecEnv


@dataclass
class PPOConfig:
    n_envs: int = 8
    n_steps: int = 512
    epochs: int = 10
    minibatch: int = 512
    gamma: float = 0.995
    gae_lambda: float = 0.95
    clip: float = 0.2
    ent_coef: float = 0.005
    ent_coef_final: float = 0.0
    lr: float = 3e-4
    lr_final: float = 0.0
    vf_coef: float = 0.5
    vf_clip: float = 10.0
    target_kl: float = 0.02
    kl_early_stop: bool = False
    """Stop the epoch loop at target_kl instead of relying on rollback alone.

    The guard as written is all-or-nothing: exceed 1.5 * target_kl on the full
    batch and the whole update is discarded, rollout included. foundation_01
    lost 51.1% of its updates that way. Early stopping lands a smaller update
    instead of none. Off by default — this changes what training does, so it is
    opt-in until a measurement says it should not be.
    """
    max_grad_norm: float = 0.5
    total_timesteps: int = 200_000_000
    seed: int = 0
    device: str = "cpu"


@dataclass
class PPOUpdateStats:
    policy_loss: float
    value_loss: float
    entropy: float
    approx_kl: float
    clip_frac: float
    explained_var: float
    lr: float
    grad_norm: float
    rejected: bool
    early_stopped: bool = False
    """True when ``kl_early_stop`` cut the epoch loop short."""
    n_minibatches: int = 0
    """Minibatch steps actually applied — ``epochs * ceil(n / minibatch)`` unless
    early stopping cut it short. Worth watching on its own: an update that keeps
    getting truncated is a learning rate that is too high for the batch."""
    log_std: tuple[float, ...] = ()
    """Per-dim exploration scale, live (post-rollback) at the end of the update.

    Load-bearing, not a curiosity: ``foundation_01`` spent most of 10.9M steps
    with throttle, brake and handbrake flat against ``log_std_max`` at sigma
    0.98 while steering annealed to 0.42, and the only way to see it was to
    open the checkpoints afterwards. A dim pinned at its cap is exploration
    that never converges, and it belongs in the stream while the run is alive.
    """


class PPO:
    """Clipped PPO with transactional KL guard."""

    def __init__(
        self,
        policy: ActorCritic,
        *,
        config: PPOConfig | None = None,
        optimizer: torch.optim.Optimizer | None = None,
    ) -> None:
        self.cfg = config or PPOConfig()
        self.device = torch.device(self.cfg.device)
        self.policy = policy.to(self.device)
        self.policy.clamp_log_std_()
        self.opt = optimizer or torch.optim.Adam(
            self.policy.parameters(), lr=self.cfg.lr
        )
        self._ent_coef = float(self.cfg.ent_coef)
        self._kl_rejections = 0

    def set_schedules(self, progress: float) -> None:
        """``progress`` in ``[0, 1]`` — cosine anneal lr and entropy."""
        progress = float(np.clip(progress, 0.0, 1.0))
        cos = 0.5 * (1.0 + math.cos(math.pi * progress))
        lr = self.cfg.lr_final + (self.cfg.lr - self.cfg.lr_final) * cos
        self._ent_coef = (
            self.cfg.ent_coef_final
            + (self.cfg.ent_coef - self.cfg.ent_coef_final) * cos
        )
        for group in self.opt.param_groups:
            group["lr"] = lr

    def update(
        self,
        obs: np.ndarray,
        actions: np.ndarray,
        old_log_probs: np.ndarray,
        advantages: np.ndarray,
        returns: np.ndarray,
        old_values: np.ndarray,
    ) -> PPOUpdateStats:
        cfg = self.cfg
        for name, arr in (
            ("obs", obs),
            ("actions", actions),
            ("old_log_probs", old_log_probs),
            ("advantages", advantages),
            ("returns", returns),
            ("old_values", old_values),
        ):
            require_finite(name, arr)

        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        act_t = torch.as_tensor(actions, dtype=torch.float32, device=self.device)
        old_logp = torch.as_tensor(
            old_log_probs, dtype=torch.float32, device=self.device
        )
        adv_t = torch.as_tensor(advantages, dtype=torch.float32, device=self.device)
        ret_t = torch.as_tensor(returns, dtype=torch.float32, device=self.device)
        val_old = torch.as_tensor(old_values, dtype=torch.float32, device=self.device)

        n = obs_t.shape[0]
        mb_size = max(1, min(int(cfg.minibatch), n))
        idx = np.arange(n)

        net_before = copy.deepcopy(self.policy.state_dict())
        opt_before = copy.deepcopy(self.opt.state_dict())

        pi_losses: list[float] = []
        vf_losses: list[float] = []
        ents: list[float] = []
        clip_fracs: list[float] = []
        grad_norms: list[float] = []
        kl_last = 0.0
        early_stopped = False

        for _ in range(cfg.epochs):
            if early_stopped:
                break
            np.random.shuffle(idx)
            for start in range(0, n, mb_size):
                if cfg.kl_early_stop and kl_last > cfg.target_kl:
                    early_stopped = True
                    break
                j = idx[start : start + mb_size]
                # Advantage normalisation per minibatch (not whole batch).
                a = adv_t[j]
                a = (a - a.mean()) / (a.std() + 1e-8)

                logp, ent, val = self.policy.evaluate(obs_t[j], act_t[j])
                logratio = logp - old_logp[j]
                ratio = logratio.exp()

                with torch.no_grad():
                    kl_last = float(((ratio - 1.0) - logratio).mean())
                    clip_fracs.append(
                        float(
                            ((ratio - 1.0).abs() > cfg.clip).float().mean().item()
                        )
                    )

                p1 = ratio * a
                p2 = torch.clamp(ratio, 1.0 - cfg.clip, 1.0 + cfg.clip) * a
                pi_loss = -torch.min(p1, p2).mean()

                if cfg.vf_clip is not None and cfg.vf_clip > 0:
                    v_clip = val_old[j] + torch.clamp(
                        val - val_old[j], -cfg.vf_clip, cfg.vf_clip
                    )
                    vf_unclipped = ((val - ret_t[j]) ** 2).mean()
                    vf_clipped = ((v_clip - ret_t[j]) ** 2).mean()
                    vf_loss = torch.max(vf_unclipped, vf_clipped)
                else:
                    vf_loss = ((val - ret_t[j]) ** 2).mean()

                ent_loss = ent.mean()
                loss = pi_loss + cfg.vf_coef * vf_loss - self._ent_coef * ent_loss
                require_finite("ppo_loss", loss)

                self.opt.zero_grad()
                loss.backward()
                grad_norm = float(
                    nn.utils.clip_grad_norm_(
                        self.policy.parameters(), cfg.max_grad_norm
                    )
                )
                for p in self.policy.parameters():
                    if p.grad is not None:
                        require_finite("grad", p.grad)
                self.opt.step()
                self.policy.clamp_log_std_()

                pi_losses.append(float(pi_loss.item()))
                vf_losses.append(float(vf_loss.item()))
                ents.append(float(ent_loss.item()))
                grad_norms.append(grad_norm)

        # Full-batch KL after the transaction — rollback if over the guard.
        rejected = False
        with torch.no_grad():
            new_logp, _, _ = self.policy.evaluate(obs_t, act_t)
            full_logratio = new_logp - old_logp
            full_ratio = full_logratio.exp()
            full_kl = float(((full_ratio - 1.0) - full_logratio).mean())
            if not np.isfinite(full_kl):
                full_kl = float("inf")
            kl_last = full_kl
            if full_kl > 1.5 * float(cfg.target_kl):
                self.policy.load_state_dict(net_before)
                self.opt.load_state_dict(opt_before)
                rejected = True
                self._kl_rejections += 1

        with torch.no_grad():
            denom = float(torch.var(ret_t))
            explained = (
                float(1.0 - torch.var(ret_t - val_old) / denom)
                if denom > 1e-12
                else 0.0
            )

        lr = float(self.opt.param_groups[0]["lr"])
        return PPOUpdateStats(
            policy_loss=float(np.mean(pi_losses) if pi_losses else 0.0),
            value_loss=float(np.mean(vf_losses) if vf_losses else 0.0),
            entropy=float(np.mean(ents) if ents else 0.0),
            approx_kl=float(kl_last),
            clip_frac=float(np.mean(clip_fracs) if clip_fracs else 0.0),
            explained_var=explained,
            lr=lr,
            grad_norm=float(np.mean(grad_norms) if grad_norms else 0.0),
            rejected=rejected,
            early_stopped=early_stopped,
            n_minibatches=len(grad_norms),
            log_std=tuple(
                float(x) for x in self.policy._bounded_log_std().detach().cpu()
            ),
        )


@dataclass
class EpisodeWindow:
    returns: deque[float] = field(default_factory=lambda: deque(maxlen=200))
    cleans: deque[float] = field(default_factory=lambda: deque(maxlen=200))
    finish_times: deque[float] = field(default_factory=lambda: deque(maxlen=200))
    progresses: deque[float] = field(default_factory=lambda: deque(maxlen=200))
    term_counts: dict[str, int] = field(
        default_factory=lambda: defaultdict(int)
    )
    term_totals: dict[str, float] = field(
        default_factory=lambda: defaultdict(float)
    )
    n_episodes: int = 0

    def add(
        self,
        ep_return: float,
        info: dict[str, Any],
    ) -> None:
        self.n_episodes += 1
        self.returns.append(float(ep_return))
        clean = bool(info.get("clean", False))
        self.cleans.append(1.0 if clean else 0.0)
        term = str(info.get("termination") or "unknown")
        self.term_counts[term] += 1
        if term == "finish":
            self.finish_times.append(float(info.get("time_s", 0.0)))
        self.progresses.append(float(info.get("progress", 0.0)))
        terms = info.get("reward_terms") or {}
        if isinstance(terms, dict):
            for k, v in terms.items():
                self.term_totals[str(k)] += float(v)

    def consume_stats(self) -> dict[str, Any]:
        """Snapshot since the last consume, then clear the window."""
        n = int(self.n_episodes)
        rets = np.asarray(list(self.returns)[-n:], dtype=np.float64) if n else np.array([])
        cleans = np.asarray(list(self.cleans)[-n:], dtype=np.float64) if n else np.array([])
        finishes = list(self.finish_times)
        progresses = list(self.progresses)[-n:] if n else []

        reward = None
        if rets.size:
            reward = {
                "mean": float(rets.mean()),
                "std": float(rets.std()) if rets.size > 1 else 0.0,
                "min": float(rets.min()),
                "max": float(rets.max()),
                "n": int(rets.size),
            }
        completion = {
            "rate": float(cleans.mean()) if cleans.size else 0.0,
            "finish": int(self.term_counts.get("finish", 0)),
            "crash": int(self.term_counts.get("crash", 0)),
            "off_course": int(self.term_counts.get("off_course", 0)),
            "stuck": int(self.term_counts.get("stuck", 0)),
            "spun": int(self.term_counts.get("spun", 0)),
            "timeout": int(self.term_counts.get("timeout", 0)),
        }
        denom = max(1, n)
        terms = (
            {k: v / denom for k, v in self.term_totals.items()}
            if self.term_totals
            else None
        )
        mean_finish = float(np.mean(finishes)) if finishes else None
        max_prog = float(np.max(progresses)) if progresses else None
        mean_ret = float(rets.mean()) if rets.size else None
        clean_rate = float(cleans.mean()) if cleans.size else None

        self.returns.clear()
        self.cleans.clear()
        self.finish_times.clear()
        self.progresses.clear()
        self.term_counts = defaultdict(int)
        self.term_totals = defaultdict(float)
        self.n_episodes = 0
        return {
            "reward": reward,
            "completion": completion,
            "terms": terms,
            "mean_return": mean_ret,
            "clean_rate": clean_rate,
            "mean_finish_time": mean_finish,
            "max_progress": max_prog,
        }


class Trainer:
    """End-to-end PPO training loop with curriculum, metrics, and checkpoints."""

    def __init__(
        self,
        *,
        run_id: str,
        stage: str = "foundation",
        workers: int = 4,
        timesteps: int = 100_000,
        tier_start: int = 0,
        config: PPOConfig | None = None,
        out_dir: str | Path = "runs",
        resume: str | Path | None = None,
        sync: bool = False,
        supervise: bool = False,
        ntfy_topic: str | None = None,
        log_std_max: float | tuple[float, ...] | None = None,
    ) -> None:
        self.run_id = str(run_id)
        self.stage = str(stage)
        self.workers = int(workers)
        self.timesteps_target = int(timesteps)
        self.cfg = config or PPOConfig()
        self.cfg.total_timesteps = self.timesteps_target
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self._stop = False

        from rallyai.train.notify import Notifier
        self.notifier = Notifier(ntfy_topic, run_id=self.run_id)
        self.supervisor = None
        if supervise:
            from rallyai.train.supervisor import Supervisor
            self.supervisor = Supervisor(trainer=self, ntfy_topic=ntfy_topic)

        self.sensor_spec = SensorSpec()
        self.reward_config = stage_reward_config(self.stage)
        self.env_config = EnvConfig()
        min_tier, max_tier = stage_tier_bounds(self.stage)
        self.tier = int(np.clip(tier_start, min_tier, max_tier))

        metrics_path = self.out_dir / f"{self.run_id}.jsonl"
        self.metrics = MetricsWriter(metrics_path, self.run_id)

        n_envs = self.workers if sync or self.workers <= 1 else self.workers
        self.cfg.n_envs = n_envs
        if sync or self.workers <= 1:
            self.vec: SyncVecEnv | AsyncVecEnv = SyncVecEnv(
                n=max(1, self.workers),
                env_config=self.env_config,
                sensor_spec=self.sensor_spec,
                reward_config=self.reward_config,
                tier=self.tier,
            )
        else:
            self.vec = AsyncVecEnv(
                n_workers=self.workers,
                envs_per_worker=1,
                env_config=self.env_config,
                sensor_spec=self.sensor_spec,
                reward_config=self.reward_config,
                tier=self.tier,
                metrics_path=metrics_path,
                run_id=self.run_id,
            )

        self.obs_dim = int(self.vec.obs_dim)
        self.act_dim = int(self.vec.act_dim)
        assert self.obs_dim == OBS_DIM or self.obs_dim == self.sensor_spec.size

        self.normaliser = ObservationNormaliser(dim=self.obs_dim)
        caps: tuple[float, ...] | None = None
        if log_std_max is not None:
            caps = (
                (float(log_std_max),) * self.act_dim
                if isinstance(log_std_max, int | float)
                else tuple(float(x) for x in log_std_max)
            )
        self.policy = ActorCritic(
            obs_dim=self.obs_dim, act_dim=self.act_dim, log_std_max=caps
        )
        self.ppo = PPO(self.policy, config=self.cfg)
        self.curriculum = Curriculum(
            tier=self.tier,
            config=CurriculumConfig(min_tier=min_tier, max_tier=max_tier),
        )
        self.hof = HallOfFame(self.out_dir, self.run_id)
        self.timesteps = 0
        self.window = EpisodeWindow()
        self._ep_ret = np.zeros(self.vec.n, dtype=np.float64)

        if resume is not None:
            self._load(resume)

    def request_stop(self, *_args: Any) -> None:
        self._stop = True

    def _load(self, path: str | Path) -> None:
        payload = load_checkpoint(path)
        self.policy.load_state_dict(payload["state_dict"])
        if payload.get("opt") is not None:
            self.ppo.opt.load_state_dict(payload["opt"])
        self.normaliser.load_state_dict(payload["normaliser"])
        self.timesteps = int(payload.get("timesteps", 0))
        self.tier = int(payload.get("tier", self.tier))
        self.curriculum.tier = self.tier
        if payload.get("curriculum"):
            self.curriculum.load_state_dict(payload["curriculum"])
        self.vec.set_tier(self.tier)
        self.policy.clamp_log_std_()

    def _normalize_obs(self, obs: np.ndarray, *, update: bool) -> np.ndarray:
        if update:
            self.normaliser.update(obs)
        return self.normaliser.normalize(obs)

    def _values(self, norm_obs: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            t = torch.as_tensor(norm_obs, dtype=torch.float32, device=self.ppo.device)
            return self.policy.value_only(t).cpu().numpy().astype(np.float32)

    def _collect(self, obs: np.ndarray) -> tuple[np.ndarray, RolloutBuffer]:
        buf = RolloutBuffer(
            self.cfg.n_steps, self.vec.n, self.obs_dim, self.act_dim
        )
        raw_obs = obs
        for _ in range(self.cfg.n_steps):
            norm = self._normalize_obs(raw_obs, update=True)
            with torch.no_grad():
                t = torch.as_tensor(norm, dtype=torch.float32, device=self.ppo.device)
                result = self.policy.act(t)
            actions = result.actions.cpu().numpy().astype(np.float32)
            logp = result.log_prob.cpu().numpy().astype(np.float32)
            values = result.value.cpu().numpy().astype(np.float32)

            next_obs, rewards, terminated, truncated, infos = self.vec.step(actions)

            bootstrap = np.zeros(self.vec.n, dtype=np.float32)
            trunc_idx = np.where(truncated & ~terminated)[0]
            if len(trunc_idx):
                finals = []
                for i in trunc_idx:
                    fo = infos[i].get("final_observation")
                    if fo is None:
                        finals.append(raw_obs[i])
                    else:
                        finals.append(np.asarray(fo, dtype=np.float32))
                final_arr = np.stack(finals, axis=0)
                final_norm = self.normaliser.normalize(final_arr)
                bootstrap[trunc_idx] = self._values(final_norm)

            buf.add(
                obs=norm,
                actions=actions,
                log_probs=logp,
                rewards=rewards.astype(np.float32),
                values=values,
                terminated=terminated,
                truncated=truncated,
                bootstrap_values=bootstrap,
            )

            self._ep_ret += rewards.astype(np.float64)
            for i in range(self.vec.n):
                if terminated[i] or truncated[i]:
                    info = infos[i]
                    # Prefer final-episode fields from the ending transition.
                    self.window.add(float(self._ep_ret[i]), info)
                    event = self.curriculum.observe(bool(info.get("clean", False)))
                    if event is not None:
                        self.tier = event.new_tier
                        self.vec.set_tier(self.tier)
                        self.metrics.write(
                            "curriculum",
                            timesteps=self.timesteps,
                            tier=self.tier,
                            completion={
                                "rate": event.completion_rate,
                                "finish": 0,
                                "crash": 0,
                                "off_course": 0,
                                "stuck": 0,
                                "spun": 0,
                                "timeout": 0,
                            },
                            msg=f"{event.reason} {event.old_tier}->{event.new_tier}",
                        )
                    self._ep_ret[i] = 0.0

            raw_obs = next_obs
            self.timesteps += self.vec.n

        last_norm = self._normalize_obs(raw_obs, update=False)
        last_values = self._values(last_norm)
        self._last_batch = buf.compute_gae(
            last_values, gamma=self.cfg.gamma, gae_lambda=self.cfg.gae_lambda
        )
        return raw_obs, buf

    def _payload(self) -> dict[str, Any]:
        return build_payload(
            policy=self.policy,
            optimizer=self.ppo.opt,
            normaliser=self.normaliser,
            sensor_spec=self.sensor_spec,
            reward_config=self.reward_config,
            timesteps=self.timesteps,
            tier=self.tier,
            stage=self.stage,
            run_id=self.run_id,
            curriculum=self.curriculum.state_dict(),
        )

    def train(self) -> dict[str, Any]:
        torch.manual_seed(self.cfg.seed)
        np.random.seed(self.cfg.seed)

        prev_handler = signal.signal(signal.SIGINT, self.request_stop)
        self.metrics.write(
            "run_start",
            timesteps=self.timesteps,
            tier=self.tier,
            msg=f"stage={self.stage} workers={self.workers}",
        )
        self.notifier.training_started(self.stage, self.workers, self.timesteps_target)

        obs = self.vec.reset(seed=self.cfg.seed)
        updates = 0
        t0 = time.perf_counter()
        summary: dict[str, Any] = {}

        try:
            while self.timesteps < self.timesteps_target and not self._stop:
                progress = self.timesteps / max(1, self.timesteps_target)
                self.ppo.set_schedules(progress)

                rollout_t0 = time.perf_counter()
                obs, _buf = self._collect(obs)
                batch = self._last_batch
                rollout_s = time.perf_counter() - rollout_t0

                flat_obs = batch.obs.reshape(-1, self.obs_dim)
                flat_act = batch.actions.reshape(-1, self.act_dim)
                flat_logp = batch.log_probs.reshape(-1)
                flat_adv = batch.advantages.reshape(-1)
                flat_ret = batch.returns.reshape(-1)
                flat_val = batch.values.reshape(-1)

                self.normaliser.update_returns(flat_ret)

                update_t0 = time.perf_counter()
                stats = self.ppo.update(
                    flat_obs, flat_act, flat_logp, flat_adv, flat_ret, flat_val
                )
                update_s = time.perf_counter() - update_t0
                updates += 1

                window = self.window.consume_stats()
                steps_per_s = (self.cfg.n_steps * self.vec.n) / max(
                    rollout_s + update_s, 1e-9
                )

                self.metrics.write(
                    "update",
                    timesteps=self.timesteps,
                    tier=self.tier,
                    reward=window["reward"],
                    terms=window["terms"],
                    completion=window["completion"],
                    ppo={
                        "policy_loss": stats.policy_loss,
                        "value_loss": stats.value_loss,
                        "entropy": stats.entropy,
                        "approx_kl": stats.approx_kl,
                        "clip_frac": stats.clip_frac,
                        "explained_var": stats.explained_var,
                        "lr": stats.lr,
                        "grad_norm": stats.grad_norm,
                        "rejected": stats.rejected,
                        "early_stopped": stats.early_stopped,
                        "n_minibatches": stats.n_minibatches,
                        "log_std": list(stats.log_std),
                    },
                    throughput={
                        "steps_per_s": float(steps_per_s),
                        "n_workers": int(self.workers),
                        "rollout_s": float(rollout_s),
                        "update_s": float(update_s),
                    },
                )

                payload = self._payload()
                written = self.hof.update(
                    payload,
                    mean_return=window["mean_return"],
                    clean_rate=window["clean_rate"],
                    mean_finish_time=window["mean_finish_time"],
                    max_progress=window["max_progress"],
                )
                for slot in written:
                    self.metrics.write(
                        "checkpoint",
                        timesteps=self.timesteps,
                        tier=self.tier,
                        checkpoint={
                            "path": str(self.hof.path_for(slot)),
                            "slot": slot,
                            "policy_hash": state_dict_hash(payload["state_dict"]),
                            "has_normaliser": True,
                        },
                    )

                if "best" in written or "cleanest" in written:
                    import json
                    replay_path = self.out_dir / f"replay_{self.timesteps}.json"
                    with open(replay_path, "w") as f:
                        json.dump({"msg": "Auto-replay placeholder"}, f)

                summary = {
                    "timesteps": self.timesteps,
                    "updates": updates,
                    "steps_per_s": steps_per_s,
                    "approx_kl": stats.approx_kl,
                    "rejected": stats.rejected,
                    "tier": self.tier,
                    "completion_rate": (window["completion"] or {}).get("rate"),
                    "mean_return": window["mean_return"],
                    "elapsed_s": time.perf_counter() - t0,
                    "checkpoint": str(self.hof.path_for("latest")),
                }
                if self.supervisor is not None:
                    try:
                        decisions = self.supervisor.after_update(
                            self.timesteps, window, stats
                        )
                        for d in decisions:
                            if d.kind == "rollback":
                                # Supervisor loaded the best checkpoint into our policy/optimizer/normaliser
                                obs = self.vec.reset(seed=self.cfg.seed + self.timesteps)
                            print(f"[{self.run_id}] supervisor: {d.kind} — {d.detail}", flush=True)
                    except Exception as exc:
                        print(f"[{self.run_id}] supervisor error (non-fatal): {exc}", flush=True)
                print(
                    f"[{self.run_id}] t={self.timesteps} "
                    f"sps={steps_per_s:.0f} "
                    f"R={window['mean_return']} "
                    f"kl={stats.approx_kl:.4f}"
                    f"{' REJECT' if stats.rejected else ''}",
                    flush=True,
                )
        except Exception as exc:
            self.notifier.training_crashed(str(exc))
            raise
        finally:
            signal.signal(signal.SIGINT, prev_handler)
            # Graceful stop: finish current update already done; write checkpoint.
            payload = self._payload()
            save_checkpoint(self.hof.path_for("latest"), payload)
            self.metrics.write(
                "checkpoint",
                timesteps=self.timesteps,
                tier=self.tier,
                checkpoint={
                    "path": str(self.hof.path_for("latest")),
                    "slot": "latest",
                    "policy_hash": state_dict_hash(payload["state_dict"]),
                    "has_normaliser": True,
                },
            )
            self.metrics.write(
                "run_end",
                timesteps=self.timesteps,
                tier=self.tier,
                msg="interrupted" if self._stop else "complete",
            )
            if self._stop:
                self.notifier.send("Training interrupted", f"t={self.timesteps}")
            else:
                self.notifier.training_complete(self.timesteps, time.perf_counter() - t0)
            self.metrics.close()
            self.vec.close()

        return summary
