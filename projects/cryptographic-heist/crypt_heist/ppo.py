"""Small PPO trainers for early learned-driving curriculum stages."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .env import CryptHeistParallelEnv
from .imitation import require_torch
from .observations import OBS_SIZE, build_observation
from .rewards import RewardWeights
from .sim import HeistSim


class PPOActorCritic(require_torch()[1].Module):
    """Gaussian actor-critic over one vehicle's continuous control vector."""

    def __init__(self, obs_dim: int = OBS_SIZE, hidden: int = 128, action_dim: int = 3):
        torch, nn = require_torch()
        super().__init__()
        self.obs_dim = int(obs_dim)
        self.hidden = int(hidden)
        self.action_dim = int(action_dim)
        self.encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
        )
        self.actor_mean = nn.Sequential(
            nn.Linear(hidden, action_dim),
            nn.Tanh(),
        )
        self.value_head = nn.Linear(hidden, 1)
        self.log_std = nn.Parameter(torch.full((action_dim,), -0.45))

    def forward(self, obs):
        features = self.encoder(obs)
        mean = self.actor_mean(features)
        value = self.value_head(features).squeeze(-1)
        return mean, value

    def distribution(self, obs):
        torch, _ = require_torch()
        mean, value = self(obs)
        std = torch.exp(torch.clamp(self.log_std, -5.0, 1.0)).expand_as(mean)
        return torch.distributions.Normal(mean, std), value

    def act(self, obs, deterministic: bool = False):
        torch, _ = require_torch()
        dist, value = self.distribution(obs)
        raw = dist.mean if deterministic else dist.sample()
        log_prob = dist.log_prob(raw).sum(dim=-1)
        action = torch.clamp(raw, -1.0, 1.0)
        return raw, action, log_prob, value

    def evaluate_actions(self, obs, raw_actions):
        dist, value = self.distribution(obs)
        log_prob = dist.log_prob(raw_actions).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        return log_prob, entropy, value

    def deterministic_action(self, obs):
        _, action, _, _ = self.act(obs, deterministic=True)
        return action


@dataclass
class PPOConfig:
    agent_name: str = "evader_0"
    agent_names: list[str] | None = None
    checkpoint_kind: str = "evader_ppo"
    seed: int = 11
    updates: int = 8
    steps_per_update: int = 512
    max_cycles: int = 900
    control_repeat: int = 4
    gamma: float = 0.992
    gae_lambda: float = 0.94
    clip_ratio: float = 0.18
    lr: float = 3e-4
    train_epochs: int = 4
    minibatch_size: int = 128
    entropy_coef: float = 0.012
    value_coef: float = 0.5
    max_grad_norm: float = 0.7
    hidden: int = 128
    decoder_accuracy_alpha: float = 0.08
    spoof_susceptibility_gamma: float = 0.50
    evader_spoof_reward: float = 0.35
    prediction_horizon_steps: int = 8
    counterfactual_interval: int = 16
    counterfactual_horizon_steps: int = 24
    counterfactual_evader_weight: float = 0.20
    counterfactual_pursuer_weight: float = 0.20
    opponent_evader_checkpoint: str | None = None
    opponent_pursuer_checkpoint: str | None = None
    opponent_pursuer_team_checkpoint: str | None = None
    init_checkpoint: str | None = None
    validation_steps: int = 0
    validation_seed: int | None = None
    device: str = "cpu"


@dataclass
class PPOTrainingResult:
    checkpoint: str
    updates: int
    steps: int
    episodes: int
    final_mean_episode_return: float
    final_loss: float
    stats: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "checkpoint": self.checkpoint,
            "updates": self.updates,
            "steps": self.steps,
            "episodes": self.episodes,
            "final_mean_episode_return": self.final_mean_episode_return,
            "final_loss": self.final_loss,
            "stats": self.stats,
        }


def train_evader_ppo(
    out: str | Path,
    *,
    seed: int = 11,
    updates: int = 8,
    steps_per_update: int = 512,
    max_cycles: int = 900,
    control_repeat: int = 4,
    gamma: float = 0.992,
    gae_lambda: float = 0.94,
    clip_ratio: float = 0.18,
    lr: float = 3e-4,
    train_epochs: int = 4,
    minibatch_size: int = 128,
    entropy_coef: float = 0.012,
    value_coef: float = 0.5,
    max_grad_norm: float = 0.7,
    hidden: int = 128,
    decoder_accuracy_alpha: float = 0.08,
    spoof_susceptibility_gamma: float = 0.50,
    evader_spoof_reward: float = 0.35,
    prediction_horizon_steps: int = 8,
    counterfactual_interval: int = 16,
    counterfactual_horizon_steps: int = 24,
    counterfactual_evader_weight: float = 0.20,
    counterfactual_pursuer_weight: float = 0.20,
    opponent_pursuer_checkpoint: str | Path | None = None,
    opponent_pursuer_team_checkpoint: str | Path | None = None,
    init_checkpoint: str | Path | None = None,
    validation_steps: int = 0,
    validation_seed: int | None = None,
    device: str = "cpu",
) -> PPOTrainingResult:
    """Train the evader's continuous controls against scripted pursuers."""
    return _train_single_agent_ppo(
        out,
        agent_name="evader_0",
        checkpoint_kind="evader_ppo",
        action_builder=_evader_action,
        seed=seed,
        updates=updates,
        steps_per_update=steps_per_update,
        max_cycles=max_cycles,
        control_repeat=control_repeat,
        gamma=gamma,
        gae_lambda=gae_lambda,
        clip_ratio=clip_ratio,
        lr=lr,
        train_epochs=train_epochs,
        minibatch_size=minibatch_size,
        entropy_coef=entropy_coef,
        value_coef=value_coef,
        max_grad_norm=max_grad_norm,
        hidden=hidden,
        decoder_accuracy_alpha=decoder_accuracy_alpha,
        spoof_susceptibility_gamma=spoof_susceptibility_gamma,
        evader_spoof_reward=evader_spoof_reward,
        prediction_horizon_steps=prediction_horizon_steps,
        counterfactual_interval=counterfactual_interval,
        counterfactual_horizon_steps=counterfactual_horizon_steps,
        counterfactual_evader_weight=counterfactual_evader_weight,
        counterfactual_pursuer_weight=counterfactual_pursuer_weight,
        opponent_pursuer_checkpoint=opponent_pursuer_checkpoint,
        opponent_pursuer_team_checkpoint=opponent_pursuer_team_checkpoint,
        init_checkpoint=init_checkpoint,
        validation_steps=validation_steps,
        validation_seed=validation_seed,
        device=device,
    )


def train_pursuer_ppo(
    out: str | Path,
    *,
    agent_name: str = "pursuer_0",
    seed: int = 11,
    updates: int = 8,
    steps_per_update: int = 512,
    max_cycles: int = 900,
    control_repeat: int = 4,
    gamma: float = 0.992,
    gae_lambda: float = 0.94,
    clip_ratio: float = 0.18,
    lr: float = 3e-4,
    train_epochs: int = 4,
    minibatch_size: int = 128,
    entropy_coef: float = 0.012,
    value_coef: float = 0.5,
    max_grad_norm: float = 0.7,
    hidden: int = 128,
    decoder_accuracy_alpha: float = 0.08,
    spoof_susceptibility_gamma: float = 0.50,
    evader_spoof_reward: float = 0.35,
    prediction_horizon_steps: int = 8,
    counterfactual_interval: int = 16,
    counterfactual_horizon_steps: int = 24,
    counterfactual_evader_weight: float = 0.20,
    counterfactual_pursuer_weight: float = 0.20,
    opponent_evader_checkpoint: str | Path | None = None,
    opponent_pursuer_team_checkpoint: str | Path | None = None,
    init_checkpoint: str | Path | None = None,
    validation_steps: int = 0,
    validation_seed: int | None = None,
    device: str = "cpu",
) -> PPOTrainingResult:
    """Train one pursuer's driving controls while the rest of the chase is scripted."""
    if agent_name not in {f"pursuer_{i}" for i in range(5)}:
        raise ValueError(f"agent_name must be pursuer_0 through pursuer_4, got {agent_name!r}")
    return _train_single_agent_ppo(
        out,
        agent_name=agent_name,
        checkpoint_kind="pursuer_ppo",
        action_builder=_pursuer_action,
        seed=seed,
        updates=updates,
        steps_per_update=steps_per_update,
        max_cycles=max_cycles,
        control_repeat=control_repeat,
        gamma=gamma,
        gae_lambda=gae_lambda,
        clip_ratio=clip_ratio,
        lr=lr,
        train_epochs=train_epochs,
        minibatch_size=minibatch_size,
        entropy_coef=entropy_coef,
        value_coef=value_coef,
        max_grad_norm=max_grad_norm,
        hidden=hidden,
        decoder_accuracy_alpha=decoder_accuracy_alpha,
        spoof_susceptibility_gamma=spoof_susceptibility_gamma,
        evader_spoof_reward=evader_spoof_reward,
        prediction_horizon_steps=prediction_horizon_steps,
        counterfactual_interval=counterfactual_interval,
        counterfactual_horizon_steps=counterfactual_horizon_steps,
        counterfactual_evader_weight=counterfactual_evader_weight,
        counterfactual_pursuer_weight=counterfactual_pursuer_weight,
        opponent_evader_checkpoint=opponent_evader_checkpoint,
        opponent_pursuer_team_checkpoint=opponent_pursuer_team_checkpoint,
        init_checkpoint=init_checkpoint,
        validation_steps=validation_steps,
        validation_seed=validation_seed,
        device=device,
    )


def train_pursuer_team_ppo(
    out: str | Path,
    *,
    agent_names: list[str] | None = None,
    seed: int = 11,
    updates: int = 8,
    steps_per_update: int = 512,
    max_cycles: int = 900,
    control_repeat: int = 4,
    gamma: float = 0.992,
    gae_lambda: float = 0.94,
    clip_ratio: float = 0.18,
    lr: float = 3e-4,
    train_epochs: int = 4,
    minibatch_size: int = 256,
    entropy_coef: float = 0.012,
    value_coef: float = 0.5,
    max_grad_norm: float = 0.7,
    hidden: int = 128,
    decoder_accuracy_alpha: float = 0.08,
    spoof_susceptibility_gamma: float = 0.50,
    evader_spoof_reward: float = 0.35,
    prediction_horizon_steps: int = 8,
    counterfactual_interval: int = 16,
    counterfactual_horizon_steps: int = 24,
    counterfactual_evader_weight: float = 0.20,
    counterfactual_pursuer_weight: float = 0.20,
    opponent_evader_checkpoint: str | Path | None = None,
    device: str = "cpu",
) -> PPOTrainingResult:
    """Train a parameter-shared control policy for all five pursuers."""
    names = agent_names or _pursuer_names()
    invalid = [name for name in names if name not in set(_pursuer_names())]
    if invalid:
        raise ValueError(f"invalid pursuer agent names: {invalid}")
    return _train_multi_agent_ppo(
        out,
        agent_names=names,
        checkpoint_kind="pursuer_team_ppo",
        action_builder=_pursuer_action,
        seed=seed,
        updates=updates,
        steps_per_update=steps_per_update,
        max_cycles=max_cycles,
        control_repeat=control_repeat,
        gamma=gamma,
        gae_lambda=gae_lambda,
        clip_ratio=clip_ratio,
        lr=lr,
        train_epochs=train_epochs,
        minibatch_size=minibatch_size,
        entropy_coef=entropy_coef,
        value_coef=value_coef,
        max_grad_norm=max_grad_norm,
        hidden=hidden,
        decoder_accuracy_alpha=decoder_accuracy_alpha,
        spoof_susceptibility_gamma=spoof_susceptibility_gamma,
        evader_spoof_reward=evader_spoof_reward,
        prediction_horizon_steps=prediction_horizon_steps,
        counterfactual_interval=counterfactual_interval,
        counterfactual_horizon_steps=counterfactual_horizon_steps,
        counterfactual_evader_weight=counterfactual_evader_weight,
        counterfactual_pursuer_weight=counterfactual_pursuer_weight,
        opponent_evader_checkpoint=opponent_evader_checkpoint,
        device=device,
    )


def _train_single_agent_ppo(
    out: str | Path,
    *,
    agent_name: str,
    checkpoint_kind: str,
    action_builder,
    seed: int,
    updates: int,
    steps_per_update: int,
    max_cycles: int,
    control_repeat: int,
    gamma: float,
    gae_lambda: float,
    clip_ratio: float,
    lr: float,
    train_epochs: int,
    minibatch_size: int,
    entropy_coef: float,
    value_coef: float,
    max_grad_norm: float,
    hidden: int,
    decoder_accuracy_alpha: float,
    spoof_susceptibility_gamma: float,
    evader_spoof_reward: float,
    prediction_horizon_steps: int,
    counterfactual_interval: int,
    counterfactual_horizon_steps: int,
    counterfactual_evader_weight: float,
    counterfactual_pursuer_weight: float,
    opponent_evader_checkpoint: str | Path | None = None,
    opponent_pursuer_checkpoint: str | Path | None = None,
    opponent_pursuer_team_checkpoint: str | Path | None = None,
    init_checkpoint: str | Path | None = None,
    validation_steps: int = 0,
    validation_seed: int | None = None,
    device: str,
) -> PPOTrainingResult:
    """Train one named agent through the PettingZoo-style chase environment."""

    torch, _ = require_torch()
    config = PPOConfig(
        agent_name=agent_name,
        checkpoint_kind=checkpoint_kind,
        seed=seed,
        updates=updates,
        steps_per_update=steps_per_update,
        max_cycles=max_cycles,
        control_repeat=control_repeat,
        gamma=gamma,
        gae_lambda=gae_lambda,
        clip_ratio=clip_ratio,
        lr=lr,
        train_epochs=train_epochs,
        minibatch_size=minibatch_size,
        entropy_coef=entropy_coef,
        value_coef=value_coef,
        max_grad_norm=max_grad_norm,
        hidden=hidden,
        decoder_accuracy_alpha=decoder_accuracy_alpha,
        spoof_susceptibility_gamma=spoof_susceptibility_gamma,
        evader_spoof_reward=evader_spoof_reward,
        prediction_horizon_steps=prediction_horizon_steps,
        counterfactual_interval=counterfactual_interval,
        counterfactual_horizon_steps=counterfactual_horizon_steps,
        counterfactual_evader_weight=counterfactual_evader_weight,
        counterfactual_pursuer_weight=counterfactual_pursuer_weight,
        opponent_evader_checkpoint=str(opponent_evader_checkpoint) if opponent_evader_checkpoint else None,
        opponent_pursuer_checkpoint=str(opponent_pursuer_checkpoint) if opponent_pursuer_checkpoint else None,
        opponent_pursuer_team_checkpoint=str(opponent_pursuer_team_checkpoint) if opponent_pursuer_team_checkpoint else None,
        init_checkpoint=str(init_checkpoint) if init_checkpoint else None,
        validation_steps=int(validation_steps),
        validation_seed=validation_seed,
        device=device,
    )

    torch.manual_seed(seed)
    np.random.seed(seed)
    env = CryptHeistParallelEnv(
        seed=seed,
        max_cycles=max_cycles,
        control_repeat=control_repeat,
        reward_weights=RewardWeights(
            decoder_accuracy_alpha=decoder_accuracy_alpha,
            spoof_susceptibility_gamma=spoof_susceptibility_gamma,
            evader_spoof_reward=evader_spoof_reward,
        ),
        prediction_horizon_steps=prediction_horizon_steps,
        counterfactual_interval=counterfactual_interval,
        counterfactual_horizon_steps=counterfactual_horizon_steps,
        counterfactual_evader_weight=counterfactual_evader_weight,
        counterfactual_pursuer_weight=counterfactual_pursuer_weight,
    )
    obs, _ = env.reset(seed=seed)
    current_obs = obs[agent_name].astype(np.float32)

    model = PPOActorCritic(obs_dim=OBS_SIZE, hidden=hidden, action_dim=3).to(device)
    init_info = _load_initial_policy(model, init_checkpoint, device=device) if init_checkpoint else None
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    rng = np.random.default_rng(seed)
    opponent_controllers = _load_opponent_controllers(
        evader_checkpoint=opponent_evader_checkpoint,
        pursuer_checkpoint=opponent_pursuer_checkpoint,
        pursuer_team_checkpoint=opponent_pursuer_team_checkpoint,
    )
    validation_seed_value = seed if validation_seed is None else int(validation_seed)
    best_validation: dict[str, Any] | None = None
    best_state: dict[str, Any] | None = None
    if validation_steps > 0:
        best_validation = _validate_single_agent_model(
            model,
            agent_name=agent_name,
            action_builder=action_builder,
            seed=validation_seed_value,
            steps=validation_steps,
            device=device,
            update=0,
        )
        best_validation["selected"] = True
        best_state = _clone_model_state(model)

    stats: list[dict[str, Any]] = []
    total_steps = 0
    completed_returns: list[float] = []
    completed_lengths: list[int] = []
    episode_return = 0.0
    episode_length = 0
    final_loss = 0.0
    last_info: dict[str, Any] = {}

    for update in range(1, updates + 1):
        obs_buf: list[np.ndarray] = []
        raw_action_buf: list[np.ndarray] = []
        logprob_buf: list[float] = []
        value_buf: list[float] = []
        reward_buf: list[float] = []
        done_buf: list[float] = []

        for _ in range(steps_per_update):
            obs_t = torch.as_tensor(current_obs[None, :], dtype=torch.float32, device=device)
            with torch.no_grad():
                raw, action, log_prob, value = model.act(obs_t, deterministic=False)

            raw_np = raw[0].detach().cpu().numpy().astype(np.float32)
            action_np = action[0].detach().cpu().numpy().astype(np.float32)
            learner_actions = {agent_name: action_builder(action_np)}
            actions = _merge_action_dicts(_opponent_actions(opponent_controllers, env.sim), learner_actions)
            step_obs, rewards, terms, truncs, infos = env.step(actions)
            reward = float(rewards.get(agent_name, 0.0))
            done = bool(terms.get(agent_name, False) or truncs.get(agent_name, False) or not env.agents)

            obs_buf.append(current_obs.copy())
            raw_action_buf.append(raw_np)
            logprob_buf.append(float(log_prob[0].detach().cpu()))
            value_buf.append(float(value[0].detach().cpu()))
            reward_buf.append(reward)
            done_buf.append(1.0 if done else 0.0)

            episode_return += reward
            episode_length += 1
            total_steps += 1
            last_info = infos.get(agent_name, {}) if infos else last_info

            if done:
                completed_returns.append(episode_return)
                completed_lengths.append(episode_length)
                obs, _ = env.reset()
                current_obs = obs[agent_name].astype(np.float32)
                episode_return = 0.0
                episode_length = 0
            else:
                current_obs = step_obs[agent_name].astype(np.float32)

        with torch.no_grad():
            last_obs_t = torch.as_tensor(current_obs[None, :], dtype=torch.float32, device=device)
            _, last_value_t = model(last_obs_t)
            last_value = float(last_value_t[0].detach().cpu())

        advantages, returns = _gae(
            rewards=np.asarray(reward_buf, dtype=np.float32),
            values=np.asarray(value_buf, dtype=np.float32),
            dones=np.asarray(done_buf, dtype=np.float32),
            last_value=last_value,
            gamma=gamma,
            lam=gae_lambda,
        )

        batch_obs = torch.as_tensor(np.asarray(obs_buf), dtype=torch.float32, device=device)
        batch_actions = torch.as_tensor(np.asarray(raw_action_buf), dtype=torch.float32, device=device)
        old_logprobs = torch.as_tensor(logprob_buf, dtype=torch.float32, device=device)
        batch_adv = torch.as_tensor(advantages, dtype=torch.float32, device=device)
        batch_returns = torch.as_tensor(returns, dtype=torch.float32, device=device)
        batch_adv = (batch_adv - batch_adv.mean()) / (batch_adv.std(unbiased=False) + 1e-8)

        final_loss = _ppo_update(
            model=model,
            optimizer=optimizer,
            rng=rng,
            obs=batch_obs,
            raw_actions=batch_actions,
            old_logprobs=old_logprobs,
            advantages=batch_adv,
            returns=batch_returns,
            train_epochs=train_epochs,
            minibatch_size=minibatch_size,
            clip_ratio=clip_ratio,
            entropy_coef=entropy_coef,
            value_coef=value_coef,
            max_grad_norm=max_grad_norm,
        )

        recent_returns = completed_returns[-10:]
        recent_lengths = completed_lengths[-10:]
        reward_components = last_info.get("reward_components", {}) if last_info else {}
        validation = None
        if validation_steps > 0:
            validation = _validate_single_agent_model(
                model,
                agent_name=agent_name,
                action_builder=action_builder,
                seed=validation_seed_value,
                steps=validation_steps,
                device=device,
                update=update,
            )
            if best_validation is None or validation["score"] > best_validation["score"]:
                best_validation = dict(validation)
                best_validation["selected"] = True
                best_state = _clone_model_state(model)

        stats.append({
            "update": update,
            "agent_name": agent_name,
            "steps": total_steps,
            "mean_step_reward": float(np.mean(reward_buf)),
            "mean_episode_return": float(np.mean(recent_returns)) if recent_returns else float(episode_return),
            "mean_episode_length": float(np.mean(recent_lengths)) if recent_lengths else float(episode_length),
            "episodes": len(completed_returns),
            "waypoints_hit": int(last_info.get("waypoints_hit", 0)) if last_info else 0,
            "captures": int(last_info.get("captures", 0)) if last_info else 0,
            "confidence": float(last_info.get("confidence", 0.0)) if last_info else 0.0,
            "decoder_accuracy": float(last_info.get("decoder_accuracy", 0.0)) if last_info else 0.0,
            "decoder_error": float(last_info.get("decoder_error", 0.0)) if last_info else 0.0,
            "spoof_susceptibility": float(last_info.get("spoof_susceptibility", 0.0)) if last_info else 0.0,
            "pursuer_auth_penalty": float(last_info.get("pursuer_auth_penalty", 0.0)) if last_info else 0.0,
            "evader_information_reward": float(last_info.get("evader_information_reward", 0.0)) if last_info else 0.0,
            "counterfactual_deception": float(last_info.get("counterfactual_deception", 0.0)) if last_info else 0.0,
            "counterfactual_evader_bonus": float(last_info.get("counterfactual_evader_bonus", 0.0)) if last_info else 0.0,
            "counterfactual_pursuer_penalty": float(last_info.get("counterfactual_pursuer_penalty", 0.0)) if last_info else 0.0,
            "opponent_count": len(opponent_controllers),
            "loss": float(final_loss),
            "waypoint_progress": float(reward_components.get("waypoint_progress", 0.0)),
            "validation": validation,
        })

    if best_state is not None:
        model.load_state_dict(best_state)
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "kind": checkpoint_kind,
        "agent_name": agent_name,
        "obs_dim": OBS_SIZE,
        "action_dim": 3,
        "hidden": hidden,
        "seed": seed,
        "config": asdict(config),
        "init": init_info,
        "validation": best_validation,
        "stats": stats,
        "state_dict": model.state_dict(),
        "final_loss": float(final_loss),
        "final_mean_episode_return": stats[-1]["mean_episode_return"] if stats else 0.0,
    }
    torch.save(checkpoint, out)
    return PPOTrainingResult(
        checkpoint=str(out),
        updates=updates,
        steps=total_steps,
        episodes=len(completed_returns),
        final_mean_episode_return=float(checkpoint["final_mean_episode_return"]),
        final_loss=float(final_loss),
        stats=stats,
    )


def _train_multi_agent_ppo(
    out: str | Path,
    *,
    agent_names: list[str],
    checkpoint_kind: str,
    action_builder,
    seed: int,
    updates: int,
    steps_per_update: int,
    max_cycles: int,
    control_repeat: int,
    gamma: float,
    gae_lambda: float,
    clip_ratio: float,
    lr: float,
    train_epochs: int,
    minibatch_size: int,
    entropy_coef: float,
    value_coef: float,
    max_grad_norm: float,
    hidden: int,
    decoder_accuracy_alpha: float,
    spoof_susceptibility_gamma: float,
    evader_spoof_reward: float,
    prediction_horizon_steps: int,
    counterfactual_interval: int,
    counterfactual_horizon_steps: int,
    counterfactual_evader_weight: float,
    counterfactual_pursuer_weight: float,
    opponent_evader_checkpoint: str | Path | None = None,
    device: str,
) -> PPOTrainingResult:
    torch, _ = require_torch()
    config = PPOConfig(
        agent_name="pursuer_team",
        agent_names=agent_names,
        checkpoint_kind=checkpoint_kind,
        seed=seed,
        updates=updates,
        steps_per_update=steps_per_update,
        max_cycles=max_cycles,
        control_repeat=control_repeat,
        gamma=gamma,
        gae_lambda=gae_lambda,
        clip_ratio=clip_ratio,
        lr=lr,
        train_epochs=train_epochs,
        minibatch_size=minibatch_size,
        entropy_coef=entropy_coef,
        value_coef=value_coef,
        max_grad_norm=max_grad_norm,
        hidden=hidden,
        decoder_accuracy_alpha=decoder_accuracy_alpha,
        spoof_susceptibility_gamma=spoof_susceptibility_gamma,
        evader_spoof_reward=evader_spoof_reward,
        prediction_horizon_steps=prediction_horizon_steps,
        counterfactual_interval=counterfactual_interval,
        counterfactual_horizon_steps=counterfactual_horizon_steps,
        counterfactual_evader_weight=counterfactual_evader_weight,
        counterfactual_pursuer_weight=counterfactual_pursuer_weight,
        opponent_evader_checkpoint=str(opponent_evader_checkpoint) if opponent_evader_checkpoint else None,
        device=device,
    )

    torch.manual_seed(seed)
    np.random.seed(seed)
    env = CryptHeistParallelEnv(
        seed=seed,
        max_cycles=max_cycles,
        control_repeat=control_repeat,
        reward_weights=RewardWeights(
            decoder_accuracy_alpha=decoder_accuracy_alpha,
            spoof_susceptibility_gamma=spoof_susceptibility_gamma,
            evader_spoof_reward=evader_spoof_reward,
        ),
        prediction_horizon_steps=prediction_horizon_steps,
        counterfactual_interval=counterfactual_interval,
        counterfactual_horizon_steps=counterfactual_horizon_steps,
        counterfactual_evader_weight=counterfactual_evader_weight,
        counterfactual_pursuer_weight=counterfactual_pursuer_weight,
    )
    obs, _ = env.reset(seed=seed)
    current_obs = {name: obs[name].astype(np.float32) for name in agent_names}

    model = PPOActorCritic(obs_dim=OBS_SIZE, hidden=hidden, action_dim=3).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    rng = np.random.default_rng(seed)
    opponent_controllers = _load_opponent_controllers(evader_checkpoint=opponent_evader_checkpoint)

    stats: list[dict[str, Any]] = []
    total_samples = 0
    env_steps = 0
    completed_returns: list[float] = []
    completed_lengths: list[int] = []
    episode_return = 0.0
    episode_length = 0
    final_loss = 0.0
    last_info: dict[str, Any] = {}

    for update in range(1, updates + 1):
        obs_buf: list[np.ndarray] = []
        raw_action_buf: list[np.ndarray] = []
        logprob_buf: list[float] = []
        value_buf: list[float] = []
        reward_buf: list[float] = []
        done_buf: list[float] = []

        for _ in range(steps_per_update):
            obs_batch = np.stack([current_obs[name] for name in agent_names]).astype(np.float32)
            obs_t = torch.as_tensor(obs_batch, dtype=torch.float32, device=device)
            with torch.no_grad():
                raw, action, log_prob, value = model.act(obs_t, deterministic=False)

            raw_np = raw.detach().cpu().numpy().astype(np.float32)
            action_np = action.detach().cpu().numpy().astype(np.float32)
            learner_actions = {name: action_builder(action_np[i]) for i, name in enumerate(agent_names)}
            actions = _merge_action_dicts(_opponent_actions(opponent_controllers, env.sim), learner_actions)
            step_obs, rewards, terms, truncs, infos = env.step(actions)
            done = bool(not env.agents or any(terms.get(name, False) or truncs.get(name, False) for name in agent_names))
            mean_reward = float(np.mean([rewards.get(name, 0.0) for name in agent_names]))

            for i, name in enumerate(agent_names):
                obs_buf.append(current_obs[name].copy())
                raw_action_buf.append(raw_np[i])
                logprob_buf.append(float(log_prob[i].detach().cpu()))
                value_buf.append(float(value[i].detach().cpu()))
                reward_buf.append(float(rewards.get(name, 0.0)))
                done_buf.append(1.0 if done else 0.0)
                total_samples += 1

            episode_return += mean_reward
            episode_length += 1
            env_steps += 1
            last_info = infos.get(agent_names[0], {}) if infos else last_info

            if done:
                completed_returns.append(episode_return)
                completed_lengths.append(episode_length)
                obs, _ = env.reset()
                current_obs = {name: obs[name].astype(np.float32) for name in agent_names}
                episode_return = 0.0
                episode_length = 0
            else:
                current_obs = {name: step_obs[name].astype(np.float32) for name in agent_names}

        with torch.no_grad():
            last_batch = np.stack([current_obs[name] for name in agent_names]).astype(np.float32)
            last_obs_t = torch.as_tensor(last_batch, dtype=torch.float32, device=device)
            _, last_value_t = model(last_obs_t)
            last_values = last_value_t.detach().cpu().numpy().astype(np.float32)

        advantages, returns = _gae_grouped(
            rewards=np.asarray(reward_buf, dtype=np.float32),
            values=np.asarray(value_buf, dtype=np.float32),
            dones=np.asarray(done_buf, dtype=np.float32),
            last_values=last_values,
            group_size=len(agent_names),
            gamma=gamma,
            lam=gae_lambda,
        )

        batch_obs = torch.as_tensor(np.asarray(obs_buf), dtype=torch.float32, device=device)
        batch_actions = torch.as_tensor(np.asarray(raw_action_buf), dtype=torch.float32, device=device)
        old_logprobs = torch.as_tensor(logprob_buf, dtype=torch.float32, device=device)
        batch_adv = torch.as_tensor(advantages, dtype=torch.float32, device=device)
        batch_returns = torch.as_tensor(returns, dtype=torch.float32, device=device)
        batch_adv = (batch_adv - batch_adv.mean()) / (batch_adv.std(unbiased=False) + 1e-8)

        final_loss = _ppo_update(
            model=model,
            optimizer=optimizer,
            rng=rng,
            obs=batch_obs,
            raw_actions=batch_actions,
            old_logprobs=old_logprobs,
            advantages=batch_adv,
            returns=batch_returns,
            train_epochs=train_epochs,
            minibatch_size=minibatch_size,
            clip_ratio=clip_ratio,
            entropy_coef=entropy_coef,
            value_coef=value_coef,
            max_grad_norm=max_grad_norm,
        )

        recent_returns = completed_returns[-10:]
        recent_lengths = completed_lengths[-10:]
        reward_components = last_info.get("reward_components", {}) if last_info else {}
        stats.append({
            "update": update,
            "agent_name": "pursuer_team",
            "agent_names": agent_names,
            "steps": total_samples,
            "env_steps": env_steps,
            "mean_step_reward": float(np.mean(reward_buf)),
            "mean_episode_return": float(np.mean(recent_returns)) if recent_returns else float(episode_return),
            "mean_episode_length": float(np.mean(recent_lengths)) if recent_lengths else float(episode_length),
            "episodes": len(completed_returns),
            "waypoints_hit": int(last_info.get("waypoints_hit", 0)) if last_info else 0,
            "captures": int(last_info.get("captures", 0)) if last_info else 0,
            "confidence": float(last_info.get("confidence", 0.0)) if last_info else 0.0,
            "decoder_accuracy": float(last_info.get("decoder_accuracy", 0.0)) if last_info else 0.0,
            "decoder_error": float(last_info.get("decoder_error", 0.0)) if last_info else 0.0,
            "spoof_susceptibility": float(last_info.get("spoof_susceptibility", 0.0)) if last_info else 0.0,
            "pursuer_auth_penalty": float(last_info.get("pursuer_auth_penalty", 0.0)) if last_info else 0.0,
            "evader_information_reward": float(last_info.get("evader_information_reward", 0.0)) if last_info else 0.0,
            "counterfactual_deception": float(last_info.get("counterfactual_deception", 0.0)) if last_info else 0.0,
            "counterfactual_evader_bonus": float(last_info.get("counterfactual_evader_bonus", 0.0)) if last_info else 0.0,
            "counterfactual_pursuer_penalty": float(last_info.get("counterfactual_pursuer_penalty", 0.0)) if last_info else 0.0,
            "opponent_count": len(opponent_controllers),
            "loss": float(final_loss),
            "pursuit_progress": float(reward_components.get("pursuit_progress", 0.0)),
        })

    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "kind": checkpoint_kind,
        "agent_name": "pursuer_team",
        "agent_names": agent_names,
        "obs_dim": OBS_SIZE,
        "action_dim": 3,
        "hidden": hidden,
        "seed": seed,
        "config": asdict(config),
        "stats": stats,
        "state_dict": model.state_dict(),
        "final_loss": float(final_loss),
        "final_mean_episode_return": stats[-1]["mean_episode_return"] if stats else 0.0,
    }
    torch.save(checkpoint, out)
    return PPOTrainingResult(
        checkpoint=str(out),
        updates=updates,
        steps=total_samples,
        episodes=len(completed_returns),
        final_mean_episode_return=float(checkpoint["final_mean_episode_return"]),
        final_loss=float(final_loss),
        stats=stats,
    )


def _load_initial_policy(model, checkpoint: str | Path | None, *, device: str) -> dict[str, Any] | None:
    """Warm-start a PPO actor from a compatible control checkpoint."""
    if checkpoint is None:
        return None
    torch, _ = require_torch()
    path = Path(checkpoint)
    if not path.exists():
        raise FileNotFoundError(f"init checkpoint not found: {path}")
    ckpt = torch.load(path, map_location=device, weights_only=False)
    kind = str(ckpt.get("kind", ""))
    state_dict = ckpt.get("state_dict")
    if not isinstance(state_dict, dict):
        raise ValueError(f"checkpoint {path} has no state_dict")
    target = model.state_dict()
    updates: dict[str, Any] = {}
    if kind == "evader_imitation":
        mapping = {
            "net.0.weight": "encoder.0.weight",
            "net.0.bias": "encoder.0.bias",
            "net.2.weight": "encoder.2.weight",
            "net.2.bias": "encoder.2.bias",
            "net.4.weight": "actor_mean.0.weight",
            "net.4.bias": "actor_mean.0.bias",
        }
        for src, dst in mapping.items():
            if src in state_dict and dst in target and tuple(state_dict[src].shape) == tuple(target[dst].shape):
                updates[dst] = state_dict[src]
    elif kind in {"evader_ppo", "pursuer_ppo", "pursuer_team_ppo"}:
        for key, value in state_dict.items():
            if key in target and tuple(value.shape) == tuple(target[key].shape):
                updates[key] = value
    else:
        raise ValueError(f"unsupported init checkpoint kind: {kind!r}")
    if not updates:
        raise ValueError(f"init checkpoint {path} has no compatible tensors for PPO warm start")
    target.update(updates)
    model.load_state_dict(target)
    return {
        "path": str(path),
        "kind": kind,
        "loaded_tensors": sorted(updates),
    }


def _clone_model_state(model) -> dict[str, Any]:
    return {
        key: value.detach().cpu().clone()
        for key, value in model.state_dict().items()
    }


def _validate_single_agent_model(
    model,
    *,
    agent_name: str,
    action_builder,
    seed: int,
    steps: int,
    device: str,
    update: int,
) -> dict[str, Any]:
    torch, _ = require_torch()
    sim = HeistSim(seed=seed, reset_on_capture=True)
    start_dist = _waypoint_distance(sim)
    for _ in range(max(1, int(steps))):
        obs = build_observation(sim, agent_name)
        obs_t = torch.as_tensor(obs[None, :], dtype=torch.float32, device=device)
        with torch.no_grad():
            action = model.deterministic_action(obs_t)[0].detach().cpu().numpy().astype(np.float32)
        sim.step(actions={agent_name: action_builder(action)})
    final_dist = _waypoint_distance(sim)
    score = (
        float(sim.waypoints_hit) * 100.0
        - float(sim.captures) * 100.0
        + max(0.0, start_dist - final_dist) * 0.02
        + float(sim.evader.vehicle.speed) * 0.01
    )
    return {
        "update": int(update),
        "seed": int(seed),
        "steps": int(steps),
        "score": float(score),
        "waypoints_hit": int(sim.waypoints_hit),
        "captures": int(sim.captures),
        "final_waypoint_distance": float(final_dist),
        "final_evader_speed": float(sim.evader.vehicle.speed),
        "deception_score": float(sim.deception_score),
    }


def _waypoint_distance(sim) -> float:
    wx, wy = sim.evader_planner.waypoint
    veh = sim.evader.vehicle
    return float(np.hypot(veh.x - wx, veh.y - wy))


def load_ppo_policy(path: str | Path):
    torch, _ = require_torch()
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model = PPOActorCritic(
        obs_dim=int(ckpt["obs_dim"]),
        hidden=int(ckpt["hidden"]),
        action_dim=int(ckpt["action_dim"]),
    )
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, ckpt


def _evader_action(control: np.ndarray) -> dict[str, Any]:
    return {
        "control": np.clip(control.astype(np.float32), -1.0, 1.0),
        "jam": 0,
        "spoof_tokens": np.zeros(5, dtype=np.int64),
        "target_mask": np.zeros(5, dtype=np.int64),
    }


def _pursuer_action(control: np.ndarray) -> dict[str, Any]:
    return {
        "control": np.clip(control.astype(np.float32), -1.0, 1.0),
    }


def _pursuer_names() -> list[str]:
    return [f"pursuer_{i}" for i in range(5)]


def _load_opponent_controllers(
    *,
    evader_checkpoint: str | Path | None = None,
    pursuer_checkpoint: str | Path | None = None,
    pursuer_team_checkpoint: str | Path | None = None,
) -> list[Any]:
    controllers = []
    if evader_checkpoint:
        from .policy_runtime import EvaderCheckpointController

        controllers.append(EvaderCheckpointController(evader_checkpoint))
    if pursuer_checkpoint:
        from .policy_runtime import PursuerCheckpointController

        controllers.append(PursuerCheckpointController(pursuer_checkpoint))
    if pursuer_team_checkpoint:
        from .policy_runtime import PursuerTeamCheckpointController

        controllers.append(PursuerTeamCheckpointController(pursuer_team_checkpoint))
    return controllers


def _opponent_actions(controllers: list[Any], sim) -> dict[str, Any]:
    return _merge_action_dicts(*(controller.action(sim) for controller in controllers))


def _merge_action_dicts(*action_dicts: dict[str, Any] | None) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    for actions in action_dicts:
        if not actions:
            continue
        for agent, payload in actions.items():
            existing = merged.setdefault(agent, {})
            if isinstance(existing, dict) and isinstance(payload, dict):
                existing.update(dict(payload))
            else:
                merged[agent] = payload
    return merged


def _gae(
    *,
    rewards: np.ndarray,
    values: np.ndarray,
    dones: np.ndarray,
    last_value: float,
    gamma: float,
    lam: float,
) -> tuple[np.ndarray, np.ndarray]:
    advantages = np.zeros_like(rewards, dtype=np.float32)
    gae = 0.0
    for t in range(len(rewards) - 1, -1, -1):
        next_value = last_value if t == len(rewards) - 1 else values[t + 1]
        next_nonterminal = 1.0 - dones[t]
        delta = rewards[t] + gamma * next_value * next_nonterminal - values[t]
        gae = delta + gamma * lam * next_nonterminal * gae
        advantages[t] = gae
    returns = advantages + values
    return advantages, returns.astype(np.float32)


def _gae_grouped(
    *,
    rewards: np.ndarray,
    values: np.ndarray,
    dones: np.ndarray,
    last_values: np.ndarray,
    group_size: int,
    gamma: float,
    lam: float,
) -> tuple[np.ndarray, np.ndarray]:
    advantages = np.zeros_like(rewards, dtype=np.float32)
    returns = np.zeros_like(rewards, dtype=np.float32)
    for group_idx in range(group_size):
        idx = np.arange(group_idx, len(rewards), group_size)
        adv, ret = _gae(
            rewards=rewards[idx],
            values=values[idx],
            dones=dones[idx],
            last_value=float(last_values[group_idx]),
            gamma=gamma,
            lam=lam,
        )
        advantages[idx] = adv
        returns[idx] = ret
    return advantages, returns


def _ppo_update(
    *,
    model: PPOActorCritic,
    optimizer,
    rng: np.random.Generator,
    obs,
    raw_actions,
    old_logprobs,
    advantages,
    returns,
    train_epochs: int,
    minibatch_size: int,
    clip_ratio: float,
    entropy_coef: float,
    value_coef: float,
    max_grad_norm: float,
) -> float:
    torch, _ = require_torch()
    n = int(obs.shape[0])
    final_loss = 0.0
    for _ in range(train_epochs):
        order = rng.permutation(n)
        for start in range(0, n, minibatch_size):
            idx = torch.as_tensor(order[start:start + minibatch_size], dtype=torch.long, device=obs.device)
            log_prob, entropy, value = model.evaluate_actions(obs[idx], raw_actions[idx])
            ratio = torch.exp(log_prob - old_logprobs[idx])
            unclipped = ratio * advantages[idx]
            clipped = torch.clamp(ratio, 1.0 - clip_ratio, 1.0 + clip_ratio) * advantages[idx]
            policy_loss = -torch.min(unclipped, clipped).mean()
            value_loss = 0.5 * (returns[idx] - value).pow(2).mean()
            entropy_bonus = entropy.mean()
            loss = policy_loss + value_coef * value_loss - entropy_coef * entropy_bonus

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            final_loss = float(loss.detach().cpu())
    return final_loss
