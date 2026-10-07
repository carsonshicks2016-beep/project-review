"""PettingZoo-style parallel environment for MARL training."""

from __future__ import annotations

from functools import lru_cache
from typing import Any

import numpy as np

try:
    from pettingzoo import ParallelEnv
except Exception:  # pettingzoo is optional for local smoke tests
    class ParallelEnv:  # type: ignore
        pass

try:
    from gymnasium import spaces
except Exception:  # pragma: no cover
    spaces = None

from .comms import VOCABULARY
from .observations import AGENT_NAMES, OBS_SIZE, build_observation
from .rewards import RewardSnapshot, RewardSystem, RewardWeights
from .sim import HeistSim


class CryptHeistParallelEnv(ParallelEnv):
    """Multi-agent training wrapper.

    Actions are gymnasium Dict spaces when gymnasium is installed:
    - evader: control[steer,longitudinal,handbrake], jam, spoof_tokens, target_mask
    - pursuer: control[steer,longitudinal,handbrake], tokens
    """

    metadata = {"name": "crypt_heist_v0", "render_modes": []}
    possible_agents = AGENT_NAMES[:]

    def __init__(
        self,
        seed: int | None = 11,
        max_cycles: int = 2700,
        control_repeat: int = 4,
        reward_weights: RewardWeights | None = None,
        prediction_horizon_steps: int = 8,
        counterfactual_interval: int = 0,
        counterfactual_horizon_steps: int = 24,
        counterfactual_evader_weight: float = 0.0,
        counterfactual_pursuer_weight: float = 0.0,
    ):
        self.seed_value = seed
        self.max_cycles = int(max_cycles)
        self.control_repeat = int(control_repeat)
        self.counterfactual_interval = max(0, int(counterfactual_interval))
        self.counterfactual_horizon_steps = max(1, int(counterfactual_horizon_steps))
        self.counterfactual_evader_weight = float(counterfactual_evader_weight)
        self.counterfactual_pursuer_weight = float(counterfactual_pursuer_weight)
        self.reward_system = RewardSystem(
            weights=reward_weights,
            prediction_horizon_steps=prediction_horizon_steps,
        )
        self.sim = HeistSim(seed=seed, reset_on_capture=False)
        self.agents = self.possible_agents[:]
        self.cycles = 0

    def reset(self, seed: int | None = None, options: dict | None = None):
        if seed is not None:
            self.seed_value = seed
            self.sim = HeistSim(seed=seed, reset_on_capture=False)
        else:
            self.sim.reset()
        self.reward_system.reset()
        self.agents = self.possible_agents[:]
        self.cycles = 0
        obs = {agent: build_observation(self.sim, agent) for agent in self.agents}
        infos = {agent: self._info(agent, {}) for agent in self.agents}
        return obs, infos

    def step(self, actions: dict[str, Any]):
        if not self.agents:
            return {}, {}, {}, {}, {}
        before = RewardSnapshot.from_sim(self.sim)
        for _ in range(self.control_repeat):
            self.sim.step(actions=actions)
            if self.sim.episode_done:
                break
        self.cycles += 1
        after = RewardSnapshot.from_sim(self.sim)
        done = bool(self.sim.episode_done)
        truncated = self.cycles >= self.max_cycles and not done
        rewards, components = self.reward_system.compute(before, after, done)
        self._apply_counterfactual_reward_probe(rewards, components)
        obs = {agent: build_observation(self.sim, agent) for agent in self.agents}
        terminations = {agent: done for agent in self.agents}
        truncations = {agent: truncated for agent in self.agents}
        infos = {agent: self._info(agent, components) for agent in self.agents}
        if done or truncated:
            self.agents = []
        return obs, rewards, terminations, truncations, infos

    @lru_cache(maxsize=None)
    def observation_space(self, agent: str):
        if spaces is None:
            return {"shape": (OBS_SIZE,), "dtype": "float32"}
        return spaces.Box(low=-5.0, high=5.0, shape=(OBS_SIZE,), dtype=np.float32)

    @lru_cache(maxsize=None)
    def action_space(self, agent: str):
        vocab_n = len(VOCABULARY)
        if spaces is None:
            return {"agent": agent}
        if agent == "evader_0":
            return spaces.Dict({
                "control": spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32),
                "jam": spaces.Discrete(2),
                "spoof_tokens": spaces.MultiDiscrete([vocab_n] * 5),
                "target_mask": spaces.MultiBinary(5),
            })
        return spaces.Dict({
            "control": spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32),
            "tokens": spaces.MultiDiscrete([vocab_n] * 6),
        })

    def _info(self, agent: str, reward_components: dict) -> dict:
        return {
            "agent": agent,
            "time": self.sim.time,
            "capture_timer": self.sim.capture_timer,
            "confidence": self.sim.channel.confidence,
            "jamming_budget": self.sim.channel.jamming_budget,
            "waypoints_hit": self.sim.waypoints_hit,
            "captures": self.sim.captures,
            "decoder_accuracy": reward_components.get("decoder_accuracy", 0.0),
            "decoder_error": reward_components.get("decoder_error", 0.0),
            "spoof_susceptibility": reward_components.get("spoof_susceptibility", 0.0),
            "pursuer_auth_penalty": reward_components.get("pursuer_auth_penalty", 0.0),
            "evader_information_reward": reward_components.get("evader_information_reward", 0.0),
            "counterfactual_deception": reward_components.get("counterfactual_deception", 0.0),
            "counterfactual_evader_bonus": reward_components.get("counterfactual_evader_bonus", 0.0),
            "counterfactual_pursuer_penalty": reward_components.get("counterfactual_pursuer_penalty", 0.0),
            "reward_components": reward_components,
        }

    def _apply_counterfactual_reward_probe(self, rewards: dict[str, float], components: dict) -> None:
        components.setdefault("counterfactual_deception", 0.0)
        components.setdefault("counterfactual_evader_bonus", 0.0)
        components.setdefault("counterfactual_pursuer_penalty", 0.0)
        components.setdefault("counterfactual_probe_active", 0)
        if self.counterfactual_interval <= 0:
            return
        if self.cycles % self.counterfactual_interval != 0:
            return
        if self.counterfactual_evader_weight == 0.0 and self.counterfactual_pursuer_weight == 0.0:
            return

        from .jamming import counterfactual_jam_reward, jam_candidate_payload

        spoof_tokens, target_mask = jam_candidate_payload(self.sim)
        score = counterfactual_jam_reward(
            self.sim,
            spoof_tokens=spoof_tokens,
            target_mask=target_mask,
            horizon_steps=self.counterfactual_horizon_steps,
        )
        evader_bonus = self.counterfactual_evader_weight * score
        pursuer_penalty = self.counterfactual_pursuer_weight * score
        rewards["evader_0"] = float(rewards.get("evader_0", 0.0) + evader_bonus)
        for i in range(5):
            name = f"pursuer_{i}"
            rewards[name] = float(rewards.get(name, 0.0) - pursuer_penalty)
        components["counterfactual_deception"] = float(score)
        components["counterfactual_evader_bonus"] = float(evader_bonus)
        components["counterfactual_pursuer_penalty"] = float(pursuer_penalty)
        components["counterfactual_probe_active"] = 1


def scripted_random_actions(env: CryptHeistParallelEnv) -> dict[str, Any]:
    """Small random-action helper used by the training smoke script."""
    actions = {}
    rng = np.random.default_rng(env.cycles + 17)
    for agent in env.agents:
        if agent == "evader_0":
            actions[agent] = {
                "control": rng.uniform(-0.2, 0.7, size=3).astype(np.float32),
                "jam": int(rng.random() < 0.01),
                "spoof_tokens": rng.integers(0, len(VOCABULARY), size=5),
                "target_mask": rng.integers(0, 2, size=5),
            }
        else:
            actions[agent] = {
                "control": rng.uniform(-0.25, 0.65, size=3).astype(np.float32),
                "tokens": rng.integers(0, len(VOCABULARY), size=6),
            }
    return actions
