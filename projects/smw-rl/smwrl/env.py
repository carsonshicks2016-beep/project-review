"""Vectorised environment construction for training and evaluation."""

from __future__ import annotations

from typing import Callable

import gymnasium as gym
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import (
    DummyVecEnv,
    SubprocVecEnv,
    VecFrameStack,
    VecTransposeImage,
)

from smwrl.retro_env import make_raw_env
from smwrl.wrappers import CheckpointPool, EpisodeConfig, ObsConfig, RewardConfig, SmwEnv


def make_env(
    state: str,
    reward_cfg: RewardConfig | None = None,
    episode_cfg: EpisodeConfig | None = None,
    checkpoints: CheckpointPool | None = None,
    render_mode: str | None = None,
    monitor: bool = True,
    obs_cfg: ObsConfig | None = None,
) -> gym.Env:
    env = make_raw_env(state=state, render_mode=render_mode)
    env = SmwEnv(env, reward_cfg, episode_cfg, checkpoints, obs_cfg)
    if monitor:
        env = Monitor(env, info_keywords=("max_x", "progress"))
    return env


def _thunk(state, reward_cfg, episode_cfg, checkpoints, seed, obs_cfg=None) -> Callable[[], gym.Env]:
    def _init() -> gym.Env:
        env = make_env(state, reward_cfg, episode_cfg, checkpoints, obs_cfg=obs_cfg)
        env.reset(seed=seed)
        return env

    return _init


def make_vec_env(
    state: str,
    n_envs: int = 12,
    reward_cfg: RewardConfig | None = None,
    episode_cfg: EpisodeConfig | None = None,
    checkpoints: CheckpointPool | None = None,
    frame_stack: int = 4,
    subproc: bool = True,
    seed: int = 0,
    obs_cfg: ObsConfig | None = None,
):
    """Parallel SMW envs, frame-stacked and channel-first for SB3's CnnPolicy.

    Subprocesses are what make this fast: the emulator is single-threaded C, so
    throughput scales with processes while the policy runs batched on the GPU.
    """
    if n_envs > 1 and not subproc:
        raise ValueError(
            "stable-retro allows only one emulator instance per process, so "
            f"n_envs={n_envs} requires subproc=True (DummyVecEnv would build "
            "them all in this process and the second would fail to construct)."
        )
    thunks = [
        _thunk(state, reward_cfg, episode_cfg, checkpoints, seed + i, obs_cfg)
        for i in range(n_envs)
    ]
    venv = SubprocVecEnv(thunks) if n_envs > 1 else DummyVecEnv(thunks)
    venv = VecFrameStack(venv, n_stack=frame_stack)
    return VecTransposeImage(venv)
