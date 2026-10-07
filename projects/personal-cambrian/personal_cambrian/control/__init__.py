"""Learned control (ROADMAP Stage 3): policies that drive creatures.

3.1 ships a per-body MLP PPO baseline (CleanRL-style, in PyTorch) — deliberately
self-contained so its policy can later be swapped for the Stage-3.3 morphology-
aware modular/GNN network without changing the training loop.
"""
# NOTE: the low-level PPO `train` lives in control.ppo (importing the
# control.train *module* would otherwise shadow a `train` name here). The package
# exposes the high-level config-driven `run_training`.
from .ppo import (
    ActorCritic, PPOConfig, evaluate, set_seed,
    save_checkpoint, load_checkpoint, agent_from_checkpoint,
)
from .train import run_training, make_creature_env, new_run_dir
from .modular import ModularPolicy, policy_for_env, make_modular_setup
from .warmstart import warm_start, warm_started_make_agent
from .ablation import (
    find_knee, recommend_budget, budget_curve, warmstart_curve, scratch_make_agent,
)
# Stage 9.2 JAX PPO (GPU training): import-guarded (jax_ppo_available() gates use),
# so the package still works without the Stage-9 GPU extra.
from .jax_ppo import jax_ppo_available, JaxPPOConfig, MjxVecEnv

__all__ = [
    "ActorCritic", "PPOConfig", "evaluate", "set_seed",
    "save_checkpoint", "load_checkpoint", "agent_from_checkpoint",
    "run_training", "make_creature_env", "new_run_dir",
    "ModularPolicy", "policy_for_env", "make_modular_setup",
    "warm_start", "warm_started_make_agent",
    "find_knee", "recommend_budget", "budget_curve", "warmstart_curve",
    "scratch_make_agent",
    "jax_ppo_available", "JaxPPOConfig", "MjxVecEnv",
]
