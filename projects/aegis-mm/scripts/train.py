"""
Training script for AegisMM:
Trains PPO Agent to learn optimal quote offsets against Avellaneda-Stoikov baseline,
maximizing spread capture and maker rebates while minimizing adverse selection.
"""

import os
import sys
import time
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import torch
from rich.console import Console
from rich.progress import Progress, BarColumn, TextColumn, TimeRemainingColumn

from src.config import MarketConfig, AvellanedaConfig, RiskConfig, RLConfig
from src.env.market_making_env import MarketMakingEnv
from src.models.agent import PPOAgent, PPOBuffer

console = Console()


def train(
    total_timesteps: int = 15_000,
    rollout_steps: int = 1_000,
    checkpoint_dir: str = "checkpoints",
):
    os.makedirs(checkpoint_dir, exist_ok=True)
    console.print("[bold bright_cyan]Starting AegisMM PPO Training Engine...[/bold bright_cyan]")

    # Configs
    market_cfg = MarketConfig(symbol="BTC/USD", maker_fee=-0.0001, initial_cash=10_000.0)
    as_cfg = AvellanedaConfig(gamma=0.05, kappa=1.5)
    risk_cfg = RiskConfig(max_inventory=0.50)
    rl_cfg = RLConfig(
        state_dim=16,
        hidden_dim=128,
        learning_rate=3e-4,
        total_timesteps=total_timesteps,
    )

    env = MarketMakingEnv(
        market_cfg=market_cfg,
        as_cfg=as_cfg,
        risk_cfg=risk_cfg,
        episode_length=rollout_steps,
        seed=1337,
    )

    agent = PPOAgent(rl_cfg)
    console.print(f"Device: [bold yellow]{agent.device}[/bold yellow] | Policy Params: {sum(p.numel() for p in agent.policy.parameters()):,}")

    state, _ = env.reset(seed=1337)
    global_step = 0
    best_pnl = -float("inf")
    episode_count = 0

    num_iterations = total_timesteps // rollout_steps

    for iteration in range(1, num_iterations + 1):
        start_time = time.time()
        states_buf = []
        actions_buf = []
        logprobs_buf = []
        rewards_buf = []
        dones_buf = []
        values_buf = []

        episode_reward = 0.0
        episode_rebates = 0.0
        episode_fills = 0
        last_finished_pnl = 0.0

        for step in range(rollout_steps):
            global_step += 1
            action, log_prob, val = agent.select_action(state)

            next_state, reward, terminated, truncated, info = env.step(action)

            states_buf.append(state)
            actions_buf.append(action)
            logprobs_buf.append(log_prob)
            rewards_buf.append(reward)
            dones_buf.append(terminated or truncated)
            values_buf.append(val)

            episode_reward += reward
            episode_rebates += sum(abs(f.fee_paid) for f in env.matching_engine.fill_history[-info.get("step_fills", 0):] if f.fee_paid < 0)
            episode_fills += info.get("step_fills", 0)
            last_finished_pnl = info["total_equity"] - market_cfg.initial_cash

            state = next_state

            if terminated or truncated:
                state, _ = env.reset(seed=1337 + global_step)

        # Compute next value for GAE bootstrap
        _, _, next_val = agent.select_action(state)
        advantages, returns = agent.compute_gae(
            rewards=rewards_buf,
            values=values_buf,
            dones=dones_buf,
            next_value=next_val,
        )

        buffer = PPOBuffer(
            states=torch.as_tensor(np.array(states_buf), dtype=torch.float32),
            actions=torch.as_tensor(np.array(actions_buf), dtype=torch.float32),
            log_probs=torch.as_tensor(np.array(logprobs_buf), dtype=torch.float32),
            rewards=torch.as_tensor(np.array(rewards_buf), dtype=torch.float32),
            dones=torch.as_tensor(np.array(dones_buf), dtype=torch.float32),
            values=torch.as_tensor(np.array(values_buf), dtype=torch.float32),
            advantages=torch.as_tensor(advantages, dtype=torch.float32),
            returns=torch.as_tensor(returns, dtype=torch.float32),
        )

        # Train PPO policy
        metrics = agent.train_step(buffer)
        duration = time.time() - start_time
        fps = int(rollout_steps / duration)

        net_pnl = last_finished_pnl

        console.print(
            f"[bold cyan]Iter {iteration:02d}/{num_iterations:02d}[/bold cyan] "
            f"| Step: {global_step:,} "
            f"| PnL: [{'green' if net_pnl >= 0 else 'red'}]{net_pnl:+.2f}[/] "
            f"| Rebates: [yellow]+${episode_rebates:.2f}[/] "
            f"| Fills: {episode_fills} "
            f"| Loss: {metrics['policy_loss']:.4f} "
            f"| Ent: {metrics['entropy']:.3f} "
            f"| Speed: {fps} steps/s"
        )

        # Save checkpoint
        checkpoint_path = os.path.join(checkpoint_dir, "latest_policy.pt")
        agent.save(checkpoint_path)

        if net_pnl > best_pnl:
            best_pnl = net_pnl
            best_path = os.path.join(checkpoint_dir, "best_policy.pt")
            agent.save(best_path)
            console.print(f"  ⭐ [bold green]New Best Model Saved! PnL: ${best_pnl:+.2f}[/bold green]")

    console.print(f"[bold bright_green]Training Complete! Best Checkpoint: {checkpoint_dir}/best_policy.pt[/bold bright_green]")


if __name__ == "__main__":
    train()
