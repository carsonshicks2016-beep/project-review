"""Fixed-size rollout buffer and GAE with truncation bootstrapping.

Terminated episodes (crash, finish, …) have next-state value 0.
Truncated episodes (time limit) bootstrap ``V(final_observation)``.
Treating those the same poisons the value function on long stages.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class RolloutBatch:
    obs: np.ndarray
    actions: np.ndarray
    log_probs: np.ndarray
    rewards: np.ndarray
    values: np.ndarray
    terminated: np.ndarray
    truncated: np.ndarray
    advantages: np.ndarray
    returns: np.ndarray


class RolloutBuffer:
    """``(n_steps, n_envs, ...)`` storage filled during collection."""

    def __init__(
        self,
        n_steps: int,
        n_envs: int,
        obs_dim: int,
        act_dim: int,
    ) -> None:
        if n_steps < 1 or n_envs < 1:
            raise ValueError("n_steps and n_envs must be >= 1")
        self.n_steps = int(n_steps)
        self.n_envs = int(n_envs)
        self.obs_dim = int(obs_dim)
        self.act_dim = int(act_dim)
        self.reset()

    def reset(self) -> None:
        T, N = self.n_steps, self.n_envs
        self.obs = np.zeros((T, N, self.obs_dim), dtype=np.float32)
        self.actions = np.zeros((T, N, self.act_dim), dtype=np.float32)
        self.log_probs = np.zeros((T, N), dtype=np.float32)
        self.rewards = np.zeros((T, N), dtype=np.float32)
        self.values = np.zeros((T, N), dtype=np.float32)
        self.terminated = np.zeros((T, N), dtype=np.bool_)
        self.truncated = np.zeros((T, N), dtype=np.bool_)
        self.bootstrap_values = np.zeros((T, N), dtype=np.float32)
        self._t = 0

    def add(
        self,
        obs: np.ndarray,
        actions: np.ndarray,
        log_probs: np.ndarray,
        rewards: np.ndarray,
        values: np.ndarray,
        terminated: np.ndarray,
        truncated: np.ndarray,
        bootstrap_values: np.ndarray | None = None,
    ) -> None:
        if self._t >= self.n_steps:
            raise RuntimeError("RolloutBuffer is full")
        t = self._t
        self.obs[t] = obs
        self.actions[t] = actions
        self.log_probs[t] = log_probs
        self.rewards[t] = rewards
        self.values[t] = values
        self.terminated[t] = terminated
        self.truncated[t] = truncated
        if bootstrap_values is not None:
            self.bootstrap_values[t] = bootstrap_values
        else:
            self.bootstrap_values[t] = 0.0
        self._t += 1

    @property
    def full(self) -> bool:
        return self._t >= self.n_steps

    def compute_gae(
        self,
        last_values: np.ndarray,
        *,
        gamma: float = 0.995,
        gae_lambda: float = 0.95,
    ) -> RolloutBatch:
        """GAE-λ with distinct terminated vs truncated next-state handling.

        For each transition ``t``:
        - if terminated: next value contribution is 0
        - elif truncated: next value is ``bootstrap_values[t]`` (= V(final_obs))
        - else: next value is ``values[t+1]`` (or ``last_values`` at T-1)
        """
        if not self.full:
            raise RuntimeError(
                f"buffer has {self._t}/{self.n_steps} steps; cannot compute GAE"
            )
        T, N = self.n_steps, self.n_envs
        last_values = np.asarray(last_values, dtype=np.float32).reshape(N)
        adv = np.zeros((T, N), dtype=np.float32)
        last_gae = np.zeros(N, dtype=np.float32)

        for t in reversed(range(T)):
            term = self.terminated[t].astype(np.float32)
            trunc = self.truncated[t].astype(np.float32)
            episode_end = np.clip(term + trunc, 0.0, 1.0)

            if t == T - 1:
                next_values = last_values.copy()
            else:
                next_values = self.values[t + 1]

            # Truncation replaces the post-reset value with V(final_obs).
            # Termination zeros the bootstrap entirely.
            boot = self.bootstrap_values[t]
            next_values = np.where(trunc > 0.5, boot, next_values)
            next_values = np.where(term > 0.5, 0.0, next_values)

            # Nonterminal for TD: episode ended → no carry of value / GAE.
            next_nonterminal = 1.0 - episode_end
            # But for truncated steps the bootstrap is already in next_values,
            # so the TD target still uses gamma * next_values (with episode_end
            # NOT masking that value). Standard form:
            #   δ = r + γ V_next * (1 - terminated) - V
            # where V_next is 0 on terminate, V(final) on truncate, V(s') else.
            # Equivalent: use (1 - terminated) only for the bootstrap mask, and
            # still cut GAE recursion on any episode end.
            not_terminated = 1.0 - term
            delta = (
                self.rewards[t]
                + gamma * next_values * not_terminated
                - self.values[t]
            )
            last_gae = delta + gamma * gae_lambda * next_nonterminal * last_gae
            adv[t] = last_gae

        returns = adv + self.values
        return RolloutBatch(
            obs=self.obs.copy(),
            actions=self.actions.copy(),
            log_probs=self.log_probs.copy(),
            rewards=self.rewards.copy(),
            values=self.values.copy(),
            terminated=self.terminated.copy(),
            truncated=self.truncated.copy(),
            advantages=adv,
            returns=returns,
        )


def compute_gae_from_arrays(
    rewards: np.ndarray,
    values: np.ndarray,
    terminated: np.ndarray,
    truncated: np.ndarray,
    last_values: np.ndarray,
    bootstrap_values: np.ndarray,
    *,
    gamma: float = 0.995,
    gae_lambda: float = 0.95,
) -> tuple[np.ndarray, np.ndarray]:
    """Stateless GAE used by unit tests and callers with hand-built episodes."""
    rewards = np.asarray(rewards, dtype=np.float32)
    values = np.asarray(values, dtype=np.float32)
    terminated = np.asarray(terminated, dtype=bool)
    truncated = np.asarray(truncated, dtype=bool)
    bootstrap_values = np.asarray(bootstrap_values, dtype=np.float32)
    last_values = np.asarray(last_values, dtype=np.float32)

    T, N = rewards.shape
    buf = RolloutBuffer(T, N, obs_dim=1, act_dim=1)
    for t in range(T):
        buf.add(
            obs=np.zeros((N, 1), dtype=np.float32),
            actions=np.zeros((N, 1), dtype=np.float32),
            log_probs=np.zeros(N, dtype=np.float32),
            rewards=rewards[t],
            values=values[t],
            terminated=terminated[t],
            truncated=truncated[t],
            bootstrap_values=bootstrap_values[t],
        )
    batch = buf.compute_gae(last_values, gamma=gamma, gae_lambda=gae_lambda)
    return batch.advantages, batch.returns
