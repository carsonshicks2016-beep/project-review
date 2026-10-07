"""
train_ensemble_factory.py — The Council of 9 Factory

Automates the training of multiple uniquely seeded PPO models.
Each bot is given a different random seed and hyperparameter topology
so they develop distinct "personalities" and trading strategies.
"""

import sys, os, argparse
import numpy as np
from pathlib import Path
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from data.fetch_prices import fetch_all
from data.sentiment_scorer import get_daily_sentiment
from features.state_builder import build_states
from env.trading_env import TradingEnv
from train_ppo import load_fnspid_sentiment, PrintMetricsCallback

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=100000, help="Timesteps per bot")
    parser.add_argument("--count", type=int, default=9, help="Number of bots to generate")
    parser.add_argument("--cash-penalty", type=float, default=0.01,
        help="Per‑step cash penalty for staying flat (default 0.01)")
    parser.add_argument("--benchmark-weight", type=float, default=0.1,
        help="Weight of benchmark‑return component in reward")
    args = parser.parse_args()

    print(f"""
    ╔══════════════════════════════════════════════════════════════╗
    ║               🏭 COUNCIL OF 9 FACTORY 🏭                     ║
    ║                                                              ║
    ║   Generating {args.count} distinct PPO brains for the ensemble...     ║
    ╚══════════════════════════════════════════════════════════════╝
    """)

    tickers = ["BTC-USD", "ETH-USD", "SOL-USD", "SPY", "NVDA", "AAPL"]
    print("📊 Fetching cross-asset dataset...")
    prices = fetch_all(tickers=tickers, interval="5m", force_refresh=False)
    daily_sent = load_fnspid_sentiment()

    ticker_states = {}
    for tk in tickers:
        state_mat, dates, close_prices = build_states(prices, sentiment_df=daily_sent, ticker=tk)
        split_idx = int(len(close_prices) * 0.8)
        ticker_states[tk] = {"train": (state_mat[:split_idx], close_prices[:split_idx], dates[:split_idx])}

    def make_env(ticker):
        def _init():
            s, p, d = ticker_states[ticker]["train"]
            return TradingEnv(s, p, d, initial_balance=10000.0, cash_penalty=args.cash_penalty, max_drawdown_penalty=0.0, benchmark_reward_weight=args.benchmark_weight)
        return _init

    env_fns = [make_env(tk) for tk in tickers]
    vec_env = DummyVecEnv(env_fns)

    model_dir = PROJECT_ROOT / "models"
    model_dir.mkdir(exist_ok=True)

    for i in range(1, args.count + 1):
        print(f"\n{'='*50}")
        print(f"🧠 TRAINING BOT {i} OF {args.count}")
        print(f"{'='*50}")
        
        # Vary hyperparameters to force "cognitive diversity"
        np.random.seed(i * 42)
        learning_rate = np.random.choice([0.0003, 0.0001, 0.0005])
        ent_coef = np.random.choice([0.05, 0.1, 0.15])  # higher = more exploration to overcome fees
        net_arch = np.random.choice([
            dict(pi=[64, 64], vf=[64, 64]),
            dict(pi=[128, 128], vf=[128, 128]),
            dict(pi=[256, 128, 64], vf=[256, 128, 64]), # Deep net
        ])
        
        print(f"   ⚙️ Topology: lr={learning_rate}, entropy={ent_coef}, arch={net_arch['pi']}")

        model = PPO(
            "MlpPolicy", 
            vec_env, 
            verbose=0, # Keep output clean for factory
            learning_rate=learning_rate,
            n_steps=2048,
            ent_coef=ent_coef,
            seed=i * 42,
            policy_kwargs=dict(net_arch=net_arch),
            tensorboard_log=str(PROJECT_ROOT / "tensorboard_logs")
        )
        
        # Curriculum training Phase 1: Bootstrapping (no transaction costs, learn market direction)
        for env_fn in vec_env.envs:
            env_fn.transaction_cost = 0.0
            env_fn.max_drawdown_penalty = 0.0
        
        model.learn(total_timesteps=int(args.timesteps * 0.6))
        
        # Curriculum training Phase 2: Refinement (real fees, prune bad/excess trades)
        for idx, env_fn in enumerate(vec_env.envs):
            ticker = tickers[idx]
            # Use realistic fees: 0.005% for stocks, 0.05% for crypto to encourage active HFT
            if ticker in ["SPY", "NVDA", "AAPL"]:
                env_fn.transaction_cost = 0.00005
            else:
                env_fn.transaction_cost = 0.0005
            
        model.learn(total_timesteps=int(args.timesteps * 0.4), reset_num_timesteps=False)
        
        save_path = str(model_dir / f"ppo_bot_{i}.zip")
        model.save(save_path)
        print(f"   ✅ Saved {save_path}")

    print("\n🎉 ENSEMBLE FACTORY COMPLETE! All 9 brains are ready.")

if __name__ == "__main__":
    main()
