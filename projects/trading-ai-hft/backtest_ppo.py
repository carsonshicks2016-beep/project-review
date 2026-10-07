"""
backtest_ppo.py — Evaluate the trained PPO agent out-of-sample.

Usage:
    python3 backtest_ppo.py --ticker SOL-USD
"""

import sys, os, argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

from stable_baselines3 import PPO

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from data.fetch_prices import fetch_all
from data.sentiment_loader import get_sentiment_for_ticker
from features.state_builder import build_states
from env.trading_env import TradingEnv
from train_ppo import load_fnspid_sentiment
from utils.action_mapping import action_space_mode_for_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ticker", default="BTC-USD")
    parser.add_argument("--start", default="2020-01-01")
    parser.add_argument("--balance", type=float, default=10000.0)
    parser.add_argument("--model", default=None, help="Path to PPO model zip")
    parser.add_argument("--use-synthetic-sentiment", action="store_true",
                        help="Opt into price-derived synthetic sentiment for bootstrapping only")
    args = parser.parse_args()

    model_path = Path(args.model) if args.model else PROJECT_ROOT / "models" / "ppo_trader.zip"
    if not model_path.exists() and args.model is None:
        candidates = sorted((PROJECT_ROOT / "models").glob("ppo_bot_*.zip"))
        if candidates:
            model_path = candidates[0]
            print(f"ℹ️  models/ppo_trader.zip not found; using {model_path.name}")

    if not model_path.exists():
        print(f"❌ Could not find PPO model at {model_path}")
        sys.exit(1)

    print(f"📦 Loading PPO model from {model_path}...")
    model = PPO.load(str(model_path))

    print(f"📊 Fetching out-of-sample data for {args.ticker}...")
    prices = fetch_all([args.ticker], start=args.start, force_refresh=False)
    daily_sent, sent_source = get_sentiment_for_ticker(
        prices, ticker=args.ticker, allow_synthetic=args.use_synthetic_sentiment
    )
    print(f"   📰 Sentiment source: {sent_source}")

    state_matrix, dates, close_prices = build_states(prices, sentiment_df=daily_sent, ticker=args.ticker)
    
    # 80/20 split
    split_idx = int(len(close_prices) * 0.8)
    test_states = state_matrix[split_idx:]
    test_prices = close_prices[split_idx:]
    test_dates = dates[split_idx:]
    
    print(f"   🧪 Running backtest on {len(test_prices)} candles...")
    
    is_crypto = any(c in args.ticker for c in ["USD", "BTC", "ETH", "SOL"])
    fee = 0.0005 if is_crypto else 0.00005
    env = TradingEnv(
        test_states,
        test_prices,
        test_dates,
        initial_balance=args.balance,
        transaction_cost=fee,
        max_drawdown_penalty=0.0,
        action_space_mode=action_space_mode_for_model(model),
    )
    obs, _ = env.reset()
    
    for _ in range(len(test_prices) - 1):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = env.step(action)
        if done:
            break
            
    metrics = env.get_final_metrics()
    
    print(f"\n📈 RESULTS FOR PPO on {args.ticker} (OUT OF SAMPLE)")
    print(f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print(f"Return:      {metrics['total_return_pct']:+.2f}%")
    print(f"Buy & Hold:  {metrics['buy_hold_return_pct']:+.2f}%")
    print(f"Alpha:       {metrics['alpha_pct']:+.2f}%")
    print(f"Sharpe:      {metrics['sharpe_ratio']:.3f}")
    print(f"Max DD:      {metrics['max_drawdown_pct']:.2f}%")
    print(f"Win Rate:    {metrics['win_rate']:.1%}")
    print(f"Trades:      {metrics['total_trades']}")

if __name__ == "__main__":
    main()
