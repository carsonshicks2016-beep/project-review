"""
train_crypto_universe.py — Train and promote a validated PPO crypto ensemble.

This is the "make the council better" pipeline:
  1. Discover tradable Alpaca crypto symbols or use a fallback universe.
  2. Fetch OHLCV data and filter for enough/liquid candles.
  3. Train diverse PPO candidates on the early part of the time series.
  4. Validate each candidate on later sequential windows.
  5. Promote only the top models into models/active for live trading.

Example:
    python3 train_crypto_universe.py --interval 5m --candidates 9 --timesteps 100000
    python3 train_crypto_universe.py --interval 1d --start 2020-01-01 --timesteps 500000
"""

import argparse
import json
import os
import shutil
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

try:
    sys.stdout.reconfigure(line_buffering=True)
except Exception:
    pass

from data.fetch_prices import fetch_all
from data.sentiment_loader import get_sentiment_for_ticker
from env.trading_env import TradingEnv
from features.state_builder import build_states
from utils.model_registry import ACTIVE_MODELS_DIR, write_active_manifest


FALLBACK_CRYPTOS = [
    "BTC-USD", "ETH-USD", "SOL-USD", "DOGE-USD", "AVAX-USD", "LINK-USD",
    "LTC-USD", "BCH-USD", "UNI-USD", "AAVE-USD", "MKR-USD", "SUSHI-USD",
    "CRV-USD", "GRT-USD", "BAT-USD", "YFI-USD", "XTZ-USD", "DOT-USD",
    "ADA-USD", "XRP-USD",
]


PROFILES = [
    {
        "name": "balanced",
        "env": {"action_space_mode": "symmetric",
                "max_drawdown_penalty": 0.8, "turnover_penalty": 0.30, "volatility_penalty": 0.15,
                "benchmark_reward_weight": 0.20, "cash_penalty": 0.0003, "min_rebalance_fraction": 0.05},
        "ppo": {"learning_rate": 0.0003, "ent_coef": 0.01, "n_steps": 2048,
                "policy_kwargs": {"net_arch": dict(pi=[128, 128], vf=[128, 128])}},
    },
    {
        "name": "defensive",
        "env": {"action_space_mode": "symmetric",
                "max_drawdown_penalty": 1.5, "turnover_penalty": 0.50, "volatility_penalty": 0.35,
                "benchmark_reward_weight": 0.10, "cash_penalty": 0.0, "min_rebalance_fraction": 0.08},
        "ppo": {"learning_rate": 0.0001, "ent_coef": 0.005, "n_steps": 2048,
                "policy_kwargs": {"net_arch": dict(pi=[64, 64], vf=[64, 64])}},
    },
    {
        "name": "exploratory",
        "env": {"action_space_mode": "symmetric",
                "max_drawdown_penalty": 0.5, "turnover_penalty": 0.15, "volatility_penalty": 0.05,
                "benchmark_reward_weight": 0.15, "cash_penalty": 0.0007, "min_rebalance_fraction": 0.03},
        "ppo": {"learning_rate": 0.0005, "ent_coef": 0.03, "n_steps": 1024,
                "policy_kwargs": {"net_arch": dict(pi=[256, 128, 64], vf=[256, 128, 64])}},
    },
    {
        "name": "low_turnover",
        "env": {"action_space_mode": "symmetric",
                "max_drawdown_penalty": 1.0, "turnover_penalty": 0.80, "volatility_penalty": 0.20,
                "benchmark_reward_weight": 0.20, "cash_penalty": 0.0001, "min_rebalance_fraction": 0.10},
        "ppo": {"learning_rate": 0.0002, "ent_coef": 0.01, "n_steps": 2048,
                "policy_kwargs": {"net_arch": dict(pi=[128, 64], vf=[128, 64])}},
    },
    {
        "name": "momentum",
        "env": {"action_space_mode": "symmetric",
                "max_drawdown_penalty": 0.7, "turnover_penalty": 0.25, "volatility_penalty": 0.10,
                "benchmark_reward_weight": 0.05, "cash_penalty": 0.0010, "min_rebalance_fraction": 0.04},
        "ppo": {"learning_rate": 0.0003, "ent_coef": 0.02, "n_steps": 2048,
                "policy_kwargs": {"net_arch": dict(pi=[256, 256], vf=[256, 256])}},
    },
    {
        "name": "return_hunter",
        "env": {"action_space_mode": "symmetric",
                "max_drawdown_penalty": 0.35, "turnover_penalty": 0.05, "volatility_penalty": 0.03,
                "benchmark_reward_weight": 0.0, "cash_penalty": 0.0015, "min_rebalance_fraction": 0.02},
        "ppo": {"learning_rate": 0.0007, "ent_coef": 0.04, "n_steps": 1024,
                "policy_kwargs": {"net_arch": dict(pi=[256, 128], vf=[256, 128])}},
    },
]


def periods_per_year(interval: str) -> int:
    if interval == "5m":
        return 365 * 24 * 12
    return 365


def alpaca_symbol_to_yfinance(symbol: str) -> str | None:
    symbol = str(symbol).upper().replace("/", "-")
    if not symbol.endswith("-USD"):
        return None
    return symbol


def discover_alpaca_crypto_universe() -> list:
    """Discover active tradable Alpaca crypto USD pairs."""
    load_dotenv()
    api_key = os.getenv("ALPACA_API_KEY")
    secret_key = os.getenv("ALPACA_SECRET_KEY")
    paper = os.getenv("ALPACA_PAPER", "True").lower() == "true"
    if not api_key or not secret_key:
        return []

    try:
        from alpaca.trading.client import TradingClient
        from alpaca.trading.enums import AssetClass, AssetStatus
        from alpaca.trading.requests import GetAssetsRequest

        client = TradingClient(api_key, secret_key, paper=paper)
        req = GetAssetsRequest(status=AssetStatus.ACTIVE, asset_class=AssetClass.CRYPTO)
        assets = client.get_all_assets(req)
    except Exception as exc:
        print(f"⚠️  Alpaca crypto discovery failed: {exc}")
        return []

    tickers = []
    for asset in assets:
        if not getattr(asset, "tradable", False):
            continue
        ticker = alpaca_symbol_to_yfinance(asset.symbol)
        if ticker:
            tickers.append(ticker)

    return sorted(set(tickers))


def choose_universe(args) -> list:
    if args.tickers:
        universe = args.tickers
    elif args.discover_alpaca:
        universe = discover_alpaca_crypto_universe()
        if not universe:
            print("⚠️  Falling back to built-in crypto universe.")
            universe = FALLBACK_CRYPTOS
    else:
        universe = FALLBACK_CRYPTOS

    return sorted(set(universe))[:args.max_assets]


def filter_liquid_tickers(prices: pd.DataFrame, tickers: list, min_rows: int,
                          min_median_dollar_volume: float) -> list:
    kept = []
    print("\n🔎 Liquidity/data filter:")
    for ticker in tickers:
        df = prices[prices["ticker"] == ticker].copy()
        if len(df) < min_rows:
            print(f"   ❌ {ticker:<10} rows={len(df):>6} < {min_rows}")
            continue

        dollar_volume = (df["close"].astype(float) * df["volume"].astype(float)).replace([np.inf, -np.inf], np.nan)
        median_dv = float(dollar_volume.dropna().median()) if not dollar_volume.dropna().empty else 0.0
        if median_dv < min_median_dollar_volume:
            print(f"   ❌ {ticker:<10} median $vol={median_dv:,.0f} < {min_median_dollar_volume:,.0f}")
            continue

        print(f"   ✅ {ticker:<10} rows={len(df):>6} median $vol={median_dv:,.0f}")
        kept.append(ticker)

    return kept


def make_windows(state_matrix, dates, close_prices, train_fraction: float, validation_windows: int):
    n = len(close_prices)
    split_idx = max(50, int(n * train_fraction))
    split_idx = min(split_idx, n - 20)

    train = (state_matrix[:split_idx], close_prices[:split_idx], dates[:split_idx])
    remaining = n - split_idx
    if remaining < 20:
        return train, []

    windows = []
    chunk = max(20, remaining // validation_windows)
    start = split_idx
    for i in range(validation_windows):
        end = n if i == validation_windows - 1 else min(n, start + chunk)
        if end - start >= 20:
            windows.append((state_matrix[start:end], close_prices[start:end], dates[start:end]))
        start = end
        if start >= n:
            break

    return train, windows


def build_ticker_datasets(prices: pd.DataFrame, tickers: list, args):
    datasets = {}
    for ticker in tickers:
        sent, source = get_sentiment_for_ticker(
            prices, ticker=ticker, allow_synthetic=args.use_synthetic_sentiment
        )
        print(f"\n📦 Building states for {ticker} | sentiment={source}")
        state_matrix, dates, close_prices = build_states(prices, sentiment_df=sent, ticker=ticker)
        train, windows = make_windows(
            state_matrix, dates, close_prices,
            train_fraction=args.train_fraction,
            validation_windows=args.validation_windows,
        )
        if not windows:
            print(f"   ⚠️  Skipping {ticker}: no validation windows")
            continue
        datasets[ticker] = {"train": train, "validation": windows}
    return datasets


def make_env_from_tuple(data_tuple, env_kwargs, interval):
    states, prices, dates = data_tuple
    return TradingEnv(
        states, prices, dates,
        initial_balance=10000.0,
        transaction_cost=env_kwargs.get("transaction_cost", 0.0015),
        max_drawdown_penalty=env_kwargs.get("max_drawdown_penalty", 0.8),
        turnover_penalty=env_kwargs.get("turnover_penalty", 0.0),
        volatility_penalty=env_kwargs.get("volatility_penalty", 0.0),
        benchmark_reward_weight=env_kwargs.get("benchmark_reward_weight", 0.0),
        cash_penalty=env_kwargs.get("cash_penalty", 0.0),
        min_rebalance_fraction=env_kwargs.get("min_rebalance_fraction", 0.01),
        periods_per_year=periods_per_year(interval),
        action_space_mode=env_kwargs.get("action_space_mode", "allocation"),
    )


def train_candidate(candidate_id: int, profile: dict, datasets: dict, args, output_dir: Path):
    env_kwargs = dict(profile["env"])
    env_kwargs["transaction_cost"] = args.transaction_cost
    env_fns = []
    for ticker, data in datasets.items():
        train_tuple = data["train"]
        env_fns.append(lambda t=train_tuple, kw=env_kwargs: make_env_from_tuple(t, kw, args.interval))

    vec_env = DummyVecEnv(env_fns)
    seed = args.seed + candidate_id * 101
    ppo_kwargs = dict(profile["ppo"])

    print(f"\n🧠 Candidate {candidate_id}: profile={profile['name']} seed={seed}")
    model = PPO(
        "MlpPolicy",
        vec_env,
        verbose=0,
        seed=seed,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        gamma=args.gamma,
        tensorboard_log=str(PROJECT_ROOT / "tensorboard_logs"),
        **ppo_kwargs,
    )
    model.learn(total_timesteps=args.timesteps)

    model_path = output_dir / f"candidate_{candidate_id:02d}_{profile['name']}.zip"
    model.save(str(model_path))
    return model, model_path, env_kwargs


def run_model_on_window(model, data_tuple, env_kwargs, interval):
    env = make_env_from_tuple(data_tuple, env_kwargs, interval)
    obs, _ = env.reset()
    for _ in range(len(data_tuple[1]) - 1):
        action, _ = model.predict(obs, deterministic=True)
        obs, _, done, _, _ = env.step(action)
        if done:
            break
    return env.get_final_metrics()


def score_candidate(model, datasets: dict, env_kwargs: dict, interval: str) -> dict:
    rows = []
    for ticker, data in datasets.items():
        for window_idx, window in enumerate(data["validation"], start=1):
            metrics = run_model_on_window(model, window, env_kwargs, interval)
            rows.append({
                "ticker": ticker,
                "window": window_idx,
                **metrics,
            })

    avg_alpha = float(np.mean([r["alpha_pct"] for r in rows]))
    avg_return = float(np.mean([r["total_return_pct"] for r in rows]))
    avg_sharpe = float(np.mean([np.clip(r["sharpe_ratio"], -5, 5) for r in rows]))
    avg_dd = float(np.mean([r["max_drawdown_pct"] for r in rows]))
    positive_alpha_ratio = float(np.mean([r["alpha_pct"] > 0 for r in rows]))
    avg_turnover = float(np.mean([r.get("turnover", 0.0) for r in rows]))
    avg_trades = float(np.mean([r.get("total_trades", 0) for r in rows]))
    trade_penalty = 0.0 if avg_trades >= 2.0 else (2.0 - avg_trades) * 12.5
    overtrade_penalty = max(0.0, avg_trades - 250.0) * 0.03

    score = (
        avg_alpha
        + (3.0 * avg_sharpe)
        + (8.0 * positive_alpha_ratio)
        - (0.30 * avg_dd)
        - (0.03 * avg_turnover)
        - trade_penalty
        - overtrade_penalty
    )

    return {
        "score": float(score),
        "avg_alpha_pct": avg_alpha,
        "avg_return_pct": avg_return,
        "avg_sharpe": avg_sharpe,
        "avg_max_drawdown_pct": avg_dd,
        "positive_alpha_ratio": positive_alpha_ratio,
        "avg_turnover": avg_turnover,
        "avg_trades": avg_trades,
        "trade_penalty": trade_penalty,
        "overtrade_penalty": overtrade_penalty,
        "windows": rows,
    }


def promote_models(results: list, args, run_dir: Path, tickers: list):
    ACTIVE_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    for old in ACTIVE_MODELS_DIR.glob("*.zip"):
        old.unlink()

    eligible = [
        result for result in results
        if (
            result["metrics"].get("avg_trades", 0.0) >= args.min_promote_avg_trades
            and result["metrics"].get("avg_trades", 0.0) <= args.max_promote_avg_trades
            and result["score"] >= args.min_promote_score
        )
    ]
    if not eligible:
        print(
            f"\n⚠️  No candidates met average trade bounds "
            f"({args.min_promote_avg_trades} to {args.max_promote_avg_trades}) "
            f"and min score ({args.min_promote_score}). "
            "models/active left empty."
        )
        write_active_manifest({
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "run_dir": str(run_dir),
            "interval": args.interval,
            "tickers": tickers,
            "promoted": [],
            "reason": "no candidate met min_promote_avg_trades",
        })
        return

    promoted = []
    for rank, result in enumerate(eligible[:args.promote_count], start=1):
        src = Path(result["model_path"])
        dst = ACTIVE_MODELS_DIR / f"active_{rank:02d}_{src.name}"
        shutil.copy2(src, dst)
        promoted.append({
            "rank": rank,
            "path": str(dst),
            "source_path": str(src),
            "profile": result["profile"],
            "score": result["score"],
            "metrics": result["metrics"],
        })

    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "run_dir": str(run_dir),
        "interval": args.interval,
        "tickers": tickers,
        "train_fraction": args.train_fraction,
        "validation_windows": args.validation_windows,
        "timesteps_per_candidate": args.timesteps,
        "synthetic_sentiment": args.use_synthetic_sentiment,
        "promoted": promoted,
    }
    write_active_manifest(manifest)
    print(f"\n✅ Promoted {len(promoted)} model(s) to {ACTIVE_MODELS_DIR}")
    print(f"   Manifest: {ACTIVE_MODELS_DIR / 'manifest.json'}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--discover-alpaca", action="store_true", help="Discover active Alpaca crypto USD pairs")
    parser.add_argument("--tickers", nargs="+", default=None, help="Explicit yfinance tickers")
    parser.add_argument("--max-assets", type=int, default=12)
    parser.add_argument("--start", default="2020-01-01")
    parser.add_argument("--interval", default="5m", choices=["1d", "5m"])
    parser.add_argument("--force-refresh", action="store_true")
    parser.add_argument("--min-rows", type=int, default=500)
    parser.add_argument("--min-median-dollar-volume", type=float, default=0.0)
    parser.add_argument("--use-synthetic-sentiment", action="store_true")
    parser.add_argument("--candidates", type=int, default=9)
    parser.add_argument("--promote-count", type=int, default=5)
    parser.add_argument("--no-promote", action="store_true", help="Train/evaluate only; do not update models/active")
    parser.add_argument("--min-promote-avg-trades", type=float, default=2.0,
                        help="Do not promote candidates that average fewer trades per validation window")
    parser.add_argument("--max-promote-avg-trades", type=float, default=250.0,
                        help="Do not promote candidates that churn too frequently")
    parser.add_argument("--min-promote-score", type=float, default=0.0,
                        help="Do not promote candidates with validation scores below this")
    parser.add_argument("--timesteps", type=int, default=100000)
    parser.add_argument("--train-fraction", type=float, default=0.70)
    parser.add_argument("--validation-windows", type=int, default=3)
    parser.add_argument("--transaction-cost", type=float, default=0.0015)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--n-epochs", type=int, default=10)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    run_dir = PROJECT_ROOT / "models" / "universe_runs" / datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)

    universe = choose_universe(args)
    print("\n🌌 Candidate universe:")
    print("   " + ", ".join(universe))

    prices = fetch_all(universe, start=args.start, force_refresh=args.force_refresh, interval=args.interval)
    filtered = filter_liquid_tickers(
        prices, universe,
        min_rows=args.min_rows,
        min_median_dollar_volume=args.min_median_dollar_volume,
    )
    if len(filtered) < 2:
        raise SystemExit("Need at least 2 usable assets after filtering.")

    datasets = build_ticker_datasets(prices, filtered, args)
    if len(datasets) < 2:
        raise SystemExit("Need at least 2 datasets with validation windows.")

    results = []
    for i in range(1, args.candidates + 1):
        profile = PROFILES[(i - 1) % len(PROFILES)]
        model, model_path, env_kwargs = train_candidate(i, profile, datasets, args, run_dir)
        metrics = score_candidate(model, datasets, env_kwargs, args.interval)
        result = {
            "candidate": i,
            "profile": profile["name"],
            "model_path": str(model_path),
            "env_kwargs": env_kwargs,
            "score": metrics["score"],
            "metrics": metrics,
        }
        results.append(result)
        print(
            f"   📈 score={metrics['score']:+.2f} "
            f"alpha={metrics['avg_alpha_pct']:+.2f}% "
            f"ret={metrics['avg_return_pct']:+.2f}% "
            f"sharpe={metrics['avg_sharpe']:+.2f} "
            f"dd={metrics['avg_max_drawdown_pct']:.2f}% "
            f"pos_alpha={metrics['positive_alpha_ratio']:.0%} "
            f"trades={metrics['avg_trades']:.1f}"
        )

    results = sorted(results, key=lambda r: r["score"], reverse=True)

    leaderboard_path = run_dir / "leaderboard.json"
    with open(leaderboard_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, sort_keys=True)

    print("\n🏁 Leaderboard:")
    for rank, result in enumerate(results, start=1):
        m = result["metrics"]
        print(
            f"   {rank:>2}. {Path(result['model_path']).name:<34} "
            f"score={result['score']:+7.2f} "
            f"alpha={m['avg_alpha_pct']:+7.2f}% "
            f"dd={m['avg_max_drawdown_pct']:6.2f}% "
            f"pos={m['positive_alpha_ratio']:.0%} "
            f"trades={m['avg_trades']:.1f}"
        )

    if args.no_promote:
        print("\nℹ️  --no-promote set; models/active was not changed.")
    else:
        promote_models(results, args, run_dir, filtered)
    print(f"\n📄 Full leaderboard saved to {leaderboard_path}")


if __name__ == "__main__":
    main()
