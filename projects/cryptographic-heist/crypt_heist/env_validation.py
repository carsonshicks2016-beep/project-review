"""MARL environment acceptance checks."""

from __future__ import annotations

from contextlib import redirect_stdout
import io
import json
from pathlib import Path
from typing import Any

import numpy as np

from .env import CryptHeistParallelEnv, scripted_random_actions
from .observations import AGENT_NAMES, OBS_SIZE


def run_env_validation(
    *,
    seed: int = 11,
    steps: int = 12,
    out: str | Path | None = None,
) -> dict[str, Any]:
    """Validate the PettingZoo-style MARL environment contract."""
    checks = [
        _check_space_contract(seed),
        _check_step_contract(seed, steps),
        _check_seeded_determinism(seed, steps),
        _check_truncation(seed),
        _check_counterfactual_probe(seed),
        _check_pettingzoo_parallel_api(seed),
    ]
    passed = all(check["passed"] for check in checks)
    manifest = {
        "version": 1,
        "seed": int(seed),
        "passed": passed,
        "checks_passed": sum(int(check["passed"]) for check in checks),
        "checks": len(checks),
        "score": sum(int(check["passed"]) for check in checks) / max(1, len(checks)),
        "results": checks,
    }
    if out is not None:
        path = Path(out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def _check_space_contract(seed: int) -> dict[str, Any]:
    env = CryptHeistParallelEnv(seed=seed, max_cycles=8, control_repeat=1)
    obs, infos = env.reset(seed=seed)
    missing = [agent for agent in AGENT_NAMES if agent not in obs or agent not in infos]
    finite = {
        agent: bool(agent in obs and obs[agent].shape == (OBS_SIZE,) and np.isfinite(obs[agent]).all())
        for agent in AGENT_NAMES
    }
    contains = {}
    action_shapes = {}
    for agent in AGENT_NAMES:
        observation_space = env.observation_space(agent)
        action_space = env.action_space(agent)
        contains[agent] = bool(hasattr(observation_space, "contains") and observation_space.contains(obs[agent]))
        sample = action_space.sample() if hasattr(action_space, "sample") else {}
        if isinstance(sample, dict):
            action_shapes[agent] = {
                key: list(np.asarray(value).shape)
                for key, value in sample.items()
            }
        else:
            action_shapes[agent] = str(type(sample).__name__)
    spaces_cached = all(env.observation_space(agent) is env.observation_space(agent) for agent in AGENT_NAMES) and all(
        env.action_space(agent) is env.action_space(agent) for agent in AGENT_NAMES
    )
    passed = bool(not missing and all(finite.values()) and all(contains.values()) and spaces_cached)
    return _check(
        "space_contract",
        passed,
        metrics={
            "agents": list(env.agents),
            "possible_agents": list(env.possible_agents),
            "missing": missing,
            "finite_observations": finite,
            "space_contains_observation": contains,
            "spaces_cached": spaces_cached,
            "action_sample_shapes": action_shapes,
        },
        thresholds={"agents": len(AGENT_NAMES), "obs_size": OBS_SIZE, "requires_cached_spaces": True},
    )


def _check_step_contract(seed: int, steps: int) -> dict[str, Any]:
    env = CryptHeistParallelEnv(seed=seed, max_cycles=steps + 4, control_repeat=1)
    obs, _ = env.reset(seed=seed)
    reward_keys_ok = True
    info_keys_ok = True
    finite_obs = True
    finite_rewards = True
    cycles = 0
    for _ in range(steps):
        active_agents = list(env.agents)
        obs, rewards, terminations, truncations, infos = env.step(scripted_random_actions(env))
        cycles += 1
        reward_keys_ok = reward_keys_ok and set(rewards) == set(active_agents)
        info_keys_ok = info_keys_ok and all("reward_components" in infos.get(agent, {}) for agent in active_agents)
        finite_rewards = finite_rewards and all(np.isfinite(float(value)) for value in rewards.values())
        finite_obs = finite_obs and all(np.isfinite(row).all() for row in obs.values())
        if any(terminations.values()) or any(truncations.values()) or not env.agents:
            break
    passed = bool(cycles >= 1 and reward_keys_ok and info_keys_ok and finite_rewards and finite_obs)
    return _check(
        "parallel_step_contract",
        passed,
        metrics={
            "steps_requested": int(steps),
            "steps_run": int(cycles),
            "reward_keys_ok": reward_keys_ok,
            "info_keys_ok": info_keys_ok,
            "finite_rewards": finite_rewards,
            "finite_observations": finite_obs,
            "remaining_agents": list(env.agents),
        },
        thresholds={"min_steps_run": 1, "requires_reward_info_and_finite_observation": True},
    )


def _check_seeded_determinism(seed: int, steps: int) -> dict[str, Any]:
    left = CryptHeistParallelEnv(seed=seed, max_cycles=steps + 2, control_repeat=1)
    right = CryptHeistParallelEnv(seed=seed, max_cycles=steps + 2, control_repeat=1)
    left_obs, _ = left.reset(seed=seed)
    right_obs, _ = right.reset(seed=seed)
    max_obs_delta = _max_obs_delta(left_obs, right_obs)
    max_reward_delta = 0.0
    for _ in range(steps):
        left_actions = scripted_random_actions(left)
        right_actions = scripted_random_actions(right)
        left_obs, left_rewards, left_terms, left_truncs, _ = left.step(left_actions)
        right_obs, right_rewards, right_terms, right_truncs, _ = right.step(right_actions)
        max_obs_delta = max(max_obs_delta, _max_obs_delta(left_obs, right_obs))
        max_reward_delta = max(max_reward_delta, _max_reward_delta(left_rewards, right_rewards))
        if not left.agents or not right.agents:
            break
        if left_terms != right_terms or left_truncs != right_truncs:
            max_obs_delta = float("inf")
            break
    passed = bool(max_obs_delta <= 1e-9 and max_reward_delta <= 1e-9)
    return _check(
        "seeded_env_determinism",
        passed,
        metrics={
            "steps": int(steps),
            "max_observation_delta": float(max_obs_delta),
            "max_reward_delta": float(max_reward_delta),
        },
        thresholds={"max_observation_delta": 1e-9, "max_reward_delta": 1e-9},
    )


def _check_truncation(seed: int) -> dict[str, Any]:
    env = CryptHeistParallelEnv(seed=seed, max_cycles=2, control_repeat=1)
    env.reset(seed=seed)
    final_truncs = {}
    for _ in range(2):
        _, _, _, final_truncs, _ = env.step(scripted_random_actions(env))
    passed = bool(not env.agents and final_truncs and all(final_truncs.values()))
    return _check(
        "max_cycle_truncation",
        passed,
        metrics={
            "agents_remaining": list(env.agents),
            "truncations": final_truncs,
            "cycles": env.cycles,
            "max_cycles": env.max_cycles,
        },
        thresholds={"requires_all_agents_truncated_at_max_cycles": True},
    )


def _check_counterfactual_probe(seed: int) -> dict[str, Any]:
    env = CryptHeistParallelEnv(
        seed=seed,
        max_cycles=4,
        control_repeat=1,
        counterfactual_interval=1,
        counterfactual_horizon_steps=3,
        counterfactual_evader_weight=0.25,
        counterfactual_pursuer_weight=0.25,
    )
    env.reset(seed=seed)
    _, rewards, _, _, infos = env.step(scripted_random_actions(env))
    components = infos["evader_0"]["reward_components"]
    passed = bool(
        components.get("counterfactual_probe_active") == 1
        and components.get("counterfactual_deception", -1.0) >= 0.0
        and components.get("counterfactual_evader_bonus", -1.0) >= 0.0
        and components.get("counterfactual_pursuer_penalty", -1.0) >= 0.0
        and np.isfinite(float(rewards["evader_0"]))
    )
    return _check(
        "counterfactual_reward_probe",
        passed,
        metrics={
            "counterfactual_probe_active": int(components.get("counterfactual_probe_active", 0)),
            "counterfactual_deception": float(components.get("counterfactual_deception", 0.0)),
            "counterfactual_evader_bonus": float(components.get("counterfactual_evader_bonus", 0.0)),
            "counterfactual_pursuer_penalty": float(components.get("counterfactual_pursuer_penalty", 0.0)),
            "evader_reward": float(rewards["evader_0"]),
        },
        thresholds={"requires_active_probe": True, "min_deception": 0.0},
    )


def _check_pettingzoo_parallel_api(seed: int) -> dict[str, Any]:
    try:
        import pettingzoo
        from pettingzoo.test import parallel_api_test

        env = CryptHeistParallelEnv(seed=seed, max_cycles=5, control_repeat=1)
        stream = io.StringIO()
        with redirect_stdout(stream):
            parallel_api_test(env, num_cycles=5)
        passed = True
        metrics = {
            "pettingzoo_version": getattr(pettingzoo, "__version__", "unknown"),
            "log_tail": stream.getvalue()[-600:],
        }
    except Exception as exc:  # pragma: no cover - reported as manifest evidence
        passed = False
        metrics = {"error": f"{type(exc).__name__}: {exc}"}
    return _check(
        "pettingzoo_parallel_api",
        passed,
        metrics=metrics,
        thresholds={"requires_pettingzoo_parallel_api_test": True},
    )


def _max_obs_delta(left: dict[str, np.ndarray], right: dict[str, np.ndarray]) -> float:
    keys = set(left) | set(right)
    if set(left) != set(right):
        return float("inf")
    if not keys:
        return 0.0
    return float(max(np.max(np.abs(left[key] - right[key])) for key in keys))


def _max_reward_delta(left: dict[str, float], right: dict[str, float]) -> float:
    keys = set(left) | set(right)
    if set(left) != set(right):
        return float("inf")
    if not keys:
        return 0.0
    return float(max(abs(float(left[key]) - float(right[key])) for key in keys))


def _check(name: str, passed: bool, *, metrics: dict[str, Any], thresholds: dict[str, Any]) -> dict[str, Any]:
    return {
        "name": name,
        "passed": bool(passed),
        "metrics": metrics,
        "thresholds": thresholds,
    }
