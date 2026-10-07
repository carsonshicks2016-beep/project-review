"""
train_ppo.py — Proximal Policy Optimization (PPO) Trainer

Trains a gradient-based Deep Reinforcement Learning agent using Stable-Baselines3.
It creates a vectorized environment across multiple crypto assets simultaneously
so the agent learns generalized trading mechanics.

Usage:
    python3 train_ppo.py --tickers BTC-USD ETH-USD SOL-USD --timesteps 500000
"""

import sys, os, argparse
import numpy as np
from pathlib import Path
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from data.fetch_prices import fetch_all
from data.sentiment_loader import get_sentiment_for_ticker, load_fnspid_daily_sentiment
from features.state_builder import build_states
from env.trading_env import TradingEnv


class PrintMetricsCallback(BaseCallback):
    """Custom callback for logging average portfolio returns."""
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_returns = []

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            if "total_pnl" in info and "portfolio_value" in info:
                # We only want to log when the episode is done, but DummyVecEnv resets automatically.
                # The 'terminal_observation' is present in 'infos' when an episode ends.
                if "terminal_observation" in info:
                    ret_pct = info["total_pnl"] / 10000.0 * 100
                    self.episode_returns.append(ret_pct)
                    print(f"   🏁 Episode Complete | Return: {ret_pct:+.1f}% | Trades: {info['total_trades']}")
                    
                    # Log dense data to TensorBoard!
                    self.logger.record("trading/return_pct", ret_pct)
                    self.logger.record("trading/win_rate", info.get("win_rate", 0))
                    self.logger.record("trading/total_trades", info.get("total_trades", 0))
        return True


def load_fnspid_sentiment():
    return load_fnspid_daily_sentiment()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tickers", nargs="+", default=["BTC-USD", "ETH-USD", "SOL-USD", "SPY", "NVDA", "AAPL"])
    parser.add_argument("--start", default="2020-01-01")
    parser.add_argument("--timesteps", type=int, default=500000)
    parser.add_argument("--use-synthetic-sentiment", action="store_true",
                        help="Opt into price-derived synthetic sentiment for bootstrapping only")
    args = parser.parse_args()

    print("""
    ╔══════════════════════════════════════════════════════════════╗
    ║                 🧠 PPO DEEP RL TRAINER 🧠                    ║
    ║                                                              ║
    ║   Training a PyTorch neural network via Stable-Baselines3    ║
    ╚══════════════════════════════════════════════════════════════╝
    """)

    print(f"📊 Fetching data for {', '.join(args.tickers)}...")
    prices = fetch_all(args.tickers, start=args.start, force_refresh=True)

    # Pre-build states for all tickers to avoid building them repeatedly
    ticker_states = {}
    for tk in args.tickers:
        tk_sent, sent_source = get_sentiment_for_ticker(
            prices, ticker=tk, allow_synthetic=args.use_synthetic_sentiment
        )
        print(f"   📰 {tk} sentiment source: {sent_source}")
            
        state_mat, dates, close_prices = build_states(prices, sentiment_df=tk_sent, ticker=tk)
        
        # 80% train / 20% test split
        split_idx = int(len(close_prices) * 0.8)
        ticker_states[tk] = {
            "train": (state_mat[:split_idx], close_prices[:split_idx], dates[:split_idx])
        }
        print(f"   ✅ {tk}: {split_idx} training candles")

    # Create a vector of environments
    def make_env(ticker):
        def _init():
            s, p, d = ticker_states[ticker]["train"]
            return TradingEnv(s, p, d, initial_balance=10000.0)
        return _init

    env_fns = [make_env(tk) for tk in args.tickers]
    vec_env = DummyVecEnv(env_fns)

    # Initialize PPO Model
    # MlpPolicy creates a standard feedforward neural network.
    model = PPO(
        "MlpPolicy", 
        vec_env, 
        verbose=1,
        learning_rate=0.0003,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,  # Discount factor
        ent_coef=0.01,  # Entropy coefficient for exploration
        tensorboard_log=str(PROJECT_ROOT / "tensorboard_logs")
    )

    print(f"\n🚀 Training PPO Agent for {args.timesteps} timesteps...")
    
    callback = PrintMetricsCallback()
    model.learn(total_timesteps=args.timesteps, callback=callback)

    # Save model
    os.makedirs(os.path.join(PROJECT_ROOT, "models"), exist_ok=True)
    save_path = str(PROJECT_ROOT / "models" / "ppo_trader.zip")
    model.save(save_path)
    
    print(f"\n✅ Training Complete! Model saved to {save_path}")
    print(f"   You can view training metrics using: tensorboard --logdir tensorboard_logs")

if __name__ == "__main__":
    main()
