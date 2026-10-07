"""
Live Paper Trading Runner with Real-Time Rich Dashboard:
Executes the AegisMM autonomous market making engine in paper trading mode,
visualizing real-time book updates, quoting decisions, maker rebates, and risk metrics.
"""

import argparse
import asyncio
import os
import signal
import sys
import time
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
from rich.live import Live

from src.config import EngineConfig, ExecutionMode, MarketConfig, AvellanedaConfig, RiskConfig
from src.env.market_making_env import MarketMakingEnv
from src.models.agent import PPOAgent
from src.dashboard.terminal_ui import TerminalDashboard
from src.execution.alpaca_exchange import AlpacaExchange


def run_paper(
    mode: str = "simulated",
    checkpoint_path: str = "checkpoints/best_policy.pt",
    symbol: str = "BTC/USD",
    fps: int = 10,
):
    dashboard = TerminalDashboard(symbol=symbol)
    config = EngineConfig()
    config.market.symbol = symbol

    agent = PPOAgent()
    if os.path.exists(checkpoint_path):
        agent.load(checkpoint_path)

    # Initialize environment
    env = MarketMakingEnv(
        market_cfg=config.market,
        as_cfg=config.avellaneda,
        risk_cfg=config.risk,
        episode_length=10_000,
        seed=int(time.time()),
    )

    state, _ = env.reset()

    # Clean shutdown handler
    running = True

    def signal_handler(sig, frame):
        nonlocal running
        running = False
        print("\n[AegisMM] Clean shutdown initiated. Flattening active orders...")
        env.matching_engine.cancel_all_orders()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)

    print(f"[AegisMM] Starting {mode.upper()} paper engine for {symbol}...")
    time.sleep(1.0)

    with Live(dashboard.console, refresh_per_second=fps, screen=True) as live:
        step = 0
        while running and step < 500:
            step += 1
            # Select action from agent
            action, _, _ = agent.select_action(state, deterministic=True)
            next_state, reward, terminated, truncated, info = env.step(action)
            state = next_state

            # Render dashboard layout
            layout = dashboard.build_layout(
                orderbook=env.orderbook,
                matching_engine=env.matching_engine,
                risk_manager=env.risk_manager,
                as_quotes=env.last_as_quotes,
                micro=env.last_micro,
                recent_fills=env.matching_engine.fill_history,
                step=step,
            )
            live.update(layout)
            time.sleep(1.0 / fps)

            if terminated or truncated:
                state, _ = env.reset(seed=int(time.time()))

    print("[AegisMM] Paper trading session completed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AegisMM Paper Trading Engine")
    parser.add_argument("--mode", type=str, default="simulated", choices=["simulated", "alpaca_paper"])
    parser.add_argument("--symbol", type=str, default="BTC/USD")
    args = parser.parse_args()

    run_paper(mode=args.mode, symbol=args.symbol)
