"""
Comprehensive Institutional Backtesting & Strategy Benchmark:
Compares:
1. Naive Fixed-Spread Market Maker
2. Classical Avellaneda-Stoikov Optimal Quoting
3. AegisMM: PPO-Augmented Residual Microstructure Policy
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from dataclasses import dataclass
from typing import Dict, List, Tuple
import numpy as np
import torch
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box

from src.config import MarketConfig, AvellanedaConfig, RiskConfig, RLConfig
from src.env.market_making_env import MarketMakingEnv
from src.models.agent import PPOAgent

console = Console()


@dataclass
class BacktestMetrics:
    strategy_name: str
    net_pnl: float
    maker_rebates: float
    total_fills: int
    max_drawdown_pct: float
    max_inventory: float
    sharpe_ratio: float
    sortino_ratio: float
    final_equity: float


def run_strategy(
    strategy_name: str,
    env: MarketMakingEnv,
    agent: PPOAgent = None,
    n_steps: int = 2_000,
    seed: int = 999,
) -> BacktestMetrics:
    state, _ = env.reset(seed=seed)
    equity_curve = [env.market_cfg.initial_cash]
    returns = []
    max_inv_seen = 0.0

    for step in range(n_steps):
        # Choose action based on strategy
        if strategy_name == "Naive Fixed Spread":
            # Overrides: quote 0 offset, never cancel
            action = np.array([0.0, 0.0, 0.5], dtype=np.float32)
        elif strategy_name == "Classical Avellaneda-Stoikov":
            # Pure AS: exactly zero offset
            action = np.array([0.0, 0.0, 0.5], dtype=np.float32)
        elif strategy_name == "AegisMM (PPO-Augmented)":
            if agent is not None:
                action, _, _ = agent.select_action(state, deterministic=True)
            else:
                action = np.array([0.0, 0.0, 0.5], dtype=np.float32)
        else:
            action = np.zeros(3, dtype=np.float32)

        next_state, reward, terminated, truncated, info = env.step(action)
        state = next_state

        mid = env.orderbook.mid_price
        eq, _, _ = env.matching_engine.get_portfolio_value(mid)
        equity_curve.append(eq)
        returns.append((eq - equity_curve[-2]) / equity_curve[-2])
        max_inv_seen = max(max_inv_seen, abs(env.matching_engine.inventory))

        if terminated or truncated:
            break

    # Calculate analytics
    eq_arr = np.array(equity_curve)
    ret_arr = np.array(returns)

    # Max Drawdown
    peaks = np.maximum.accumulate(eq_arr)
    drawdowns = (peaks - eq_arr) / peaks
    max_dd = float(np.max(drawdowns)) * 100.0

    # Sharpe & Sortino
    mean_ret = np.mean(ret_arr)
    std_ret = np.std(ret_arr) + 1e-8
    sharpe = float((mean_ret / std_ret) * np.sqrt(len(ret_arr)))

    downside_ret = ret_arr[ret_arr < 0]
    downside_std = np.std(downside_ret) + 1e-8 if len(downside_ret) > 0 else 1e-8
    sortino = float((mean_ret / downside_std) * np.sqrt(len(ret_arr)))

    net_pnl = float(eq_arr[-1] - env.market_cfg.initial_cash)

    return BacktestMetrics(
        strategy_name=strategy_name,
        net_pnl=net_pnl,
        maker_rebates=env.matching_engine.total_maker_rebates,
        total_fills=len(env.matching_engine.fill_history),
        max_drawdown_pct=max_dd,
        max_inventory=max_inv_seen,
        sharpe_ratio=sharpe,
        sortino_ratio=sortino,
        final_equity=float(eq_arr[-1]),
    )


def backtest(n_steps: int = 2_000, checkpoint_path: str = "checkpoints/best_policy.pt"):
    console.print(f"[bold bright_cyan]Running Head-to-Head Strategy Benchmark ({n_steps:,} Market Steps)...[/bold bright_cyan]")

    market_cfg = MarketConfig(symbol="BTC/USD", maker_fee=-0.0001, initial_cash=10_000.0)
    as_cfg = AvellanedaConfig(gamma=0.05, kappa=1.5)
    risk_cfg = RiskConfig(max_inventory=0.50)

    # Load trained agent if available
    agent = None
    if os.path.exists(checkpoint_path):
        console.print(f"Loading trained weights from [yellow]{checkpoint_path}[/yellow]...")
        agent = PPOAgent()
        agent.load(checkpoint_path)
    else:
        console.print("[yellow]No checkpoint found, initializing fresh agent for demonstration...[/yellow]")
        agent = PPOAgent()

    strategies = [
        "Naive Fixed Spread",
        "Classical Avellaneda-Stoikov",
        "AegisMM (PPO-Augmented)",
    ]

    results: List[BacktestMetrics] = []
    test_seed = 4242

    for strat in strategies:
        env = MarketMakingEnv(
            market_cfg=market_cfg,
            as_cfg=as_cfg,
            risk_cfg=risk_cfg,
            episode_length=n_steps + 100,
            naive_fixed_spread=(strat == "Naive Fixed Spread"),
            seed=test_seed,
        )
        res = run_strategy(strat, env, agent=agent, n_steps=n_steps, seed=test_seed)
        results.append(res)

    # Display comparison table
    table = Table(title="🏆 High-Frequency Market Making Benchmark Results", box=box.ROUNDED)
    table.add_column("Strategy", style="bold cyan")
    table.add_column("Net PnL ($)", justify="right")
    table.add_column("Maker Rebates", justify="right", style="yellow")
    table.add_column("Fills", justify="right")
    table.add_column("Max DD (%)", justify="right")
    table.add_column("Max Inv", justify="right")
    table.add_column("Sharpe", justify="right")
    table.add_column("Sortino", justify="right")

    for r in results:
        pnl_style = "bold bright_green" if r.net_pnl >= 0 else "bold bright_red"
        dd_style = "green" if r.max_drawdown_pct < 2.0 else "red"
        table.add_row(
            r.strategy_name,
            f"[{pnl_style}]${r.net_pnl:+,.2f}[/{pnl_style}]",
            f"+${r.maker_rebates:,.2f}",
            f"{r.total_fills:,}",
            f"[{dd_style}]{r.max_drawdown_pct:.2f}%[/{dd_style}]",
            f"{r.max_inventory:.3f}",
            f"{r.sharpe_ratio:+.2f}",
            f"{r.sortino_ratio:+.2f}",
        )

    console.print(table)


if __name__ == "__main__":
    backtest()
