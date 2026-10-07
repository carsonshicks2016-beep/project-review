"""
train.py — Master training script for the Trading AI.

This is the "run button" — it orchestrates the full pipeline:
  1. Fetch price data
  2. Build/score news sentiment
  3. Compute features
  4. Build state vectors
  5. Launch NEAT evolution

Usage:
    python train.py                          # Default: BTC-USD, 100 generations
    python train.py --ticker ETH-USD         # Train on Ethereum
    python train.py --generations 500        # More generations
    python train.py --resume                 # Resume from checkpoint
"""

import argparse
import sys
import os
import time
import numpy as np

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from data.fetch_prices import fetch_all
from data.fetch_news import build_news_dataset
from data.sentiment_scorer import score_news_dataframe, get_daily_sentiment
from data.sentiment_loader import get_sentiment_for_ticker
from features.state_builder import build_states
from agents.neat_trader import NEATTrader


def main():
    parser = argparse.ArgumentParser(description="Train the Trading AI")
    parser.add_argument("--ticker", type=str, default="BTC-USD",
                       help="Ticker to train on (default: BTC-USD)")
    parser.add_argument("--tickers", type=str, nargs="+",
                       default=["BTC-USD", "ETH-USD"],
                       help="All tickers to download data for")
    parser.add_argument("--start", type=str, default="2017-01-01",
                       help="Start date for data (default: 2017-01-01)")
    parser.add_argument("--generations", type=int, default=100,
                       help="Number of NEAT generations (default: 100)")
    parser.add_argument("--balance", type=float, default=10000.0,
                       help="Starting balance (default: 10000)")
    parser.add_argument("--resume", action="store_true",
                       help="Resume from last checkpoint")
    parser.add_argument("--skip-sentiment", action="store_true",
                       help="Skip FinBERT scoring (use synthetic sentiment)")
    parser.add_argument("--use-synthetic-sentiment", action="store_true",
                       help="Opt into price-derived synthetic sentiment for bootstrapping only")
    parser.add_argument("--refresh-data", action="store_true",
                       help="Force re-download of price data")
    args = parser.parse_args()
    
    print("""
    ╔══════════════════════════════════════════════════════════════╗
    ║                   🧠 TRADING AI TRAINER 🧠                  ║
    ║                                                              ║
    ║   NEAT NeuroEvolution + FinBERT Sentiment Analysis           ║
    ║   "Like Mario TAS, but for making money"                     ║
    ╚══════════════════════════════════════════════════════════════╝
    """)
    
    start_time = time.time()
    
    # ── Phase 1: Fetch Price Data ──
    print("━" * 60)
    print("📊 PHASE 1: Fetching Price Data")
    print("━" * 60)
    
    prices = fetch_all(
        tickers=args.tickers,
        start=args.start,
        force_refresh=args.refresh_data,
    )
    print(f"   Total candles: {len(prices)}")
    print(f"   Tickers: {prices['ticker'].unique().tolist()}")
    print()
    
    # ── Phase 2: Build News & Sentiment ──
    print("━" * 60)
    print("📰 PHASE 2: News & Sentiment")
    print("━" * 60)
    
    news = None

    if args.use_synthetic_sentiment:
        news = build_news_dataset(prices, use_synthetic=True)

    if news is not None and not args.skip_sentiment:
        # Score with FinBERT (if available)
        try:
            news = score_news_dataframe(news, force_rescore=False)
        except Exception as e:
            print(f"  ⚠️  FinBERT scoring failed: {e}")
            print(f"  📝 Using synthetic sentiment scores instead")
    
    if news is not None:
        daily_sentiment = get_daily_sentiment(news, ticker=args.ticker)
        sentiment_source = "synthetic"
    else:
        daily_sentiment, sentiment_source = get_sentiment_for_ticker(
            prices, ticker=args.ticker, allow_synthetic=False
        )
    print(f"   Sentiment source: {sentiment_source}")
    print(f"   Daily sentiment records: {0 if daily_sentiment is None else len(daily_sentiment)}")
    print()
    
    # ── Phase 3: Build State Vectors ──
    print("━" * 60)
    print("🔧 PHASE 3: Building State Vectors")
    print("━" * 60)
    
    state_matrix, dates, close_prices = build_states(
        prices,
        sentiment_df=daily_sentiment,
        ticker=args.ticker,
    )
    print()
    
    # ── Phase 4: Train-Test Split ──
    print("━" * 60)
    print("✂️  PHASE 4: Train/Test Split")
    print("━" * 60)
    
    # Use 80% for training, 20% for testing
    split_idx = int(len(close_prices) * 0.8)
    
    train_states = state_matrix[:split_idx]
    train_prices = close_prices[:split_idx]
    train_dates = dates[:split_idx]
    
    test_states = state_matrix[split_idx:]
    test_prices = close_prices[split_idx:]
    test_dates = dates[split_idx:]
    
    print(f"   Training: {len(train_prices)} candles ({train_dates[0]} → {train_dates[-1]})")
    print(f"   Testing:  {len(test_prices)} candles ({test_dates[0]} → {test_dates[-1]})")
    print()
    
    # ── Phase 5: NEAT Evolution ──
    print("━" * 60)
    print("🧬 PHASE 5: NEAT Evolution")
    print("━" * 60)
    
    trader = NEATTrader(
        state_matrix=train_states,
        prices=train_prices,
        dates=train_dates,
        initial_balance=args.balance,
    )
    
    # Find checkpoint to resume from
    resume_path = None
    if args.resume:
        checkpoint_dir = os.path.join(PROJECT_ROOT, "checkpoints")
        checkpoints = [f for f in os.listdir(checkpoint_dir) if f.startswith("neat-checkpoint-")]
        if checkpoints:
            latest = sorted(checkpoints)[-1]
            resume_path = os.path.join(checkpoint_dir, latest)
    
    winner = trader.evolve(
        num_generations=args.generations,
        resume_from=resume_path,
    )
    
    # ── Phase 6: Results ──
    print()
    print("━" * 60)
    print("📈 RESULTS")
    print("━" * 60)
    
    # Run best genome on training data
    print("\n🏋️ Training Performance:")
    train_results = trader.run_best()
    m = train_results["metrics"]
    print(f"   Total Return:    {m['total_return_pct']:+.2f}%")
    print(f"   Sharpe Ratio:    {m['sharpe_ratio']:.3f}")
    print(f"   Max Drawdown:    {m['max_drawdown_pct']:.2f}%")
    print(f"   Total Trades:    {m['total_trades']}")
    print(f"   Win Rate:        {m['win_rate']:.1%}")
    print(f"   Buy & Hold:      {m['buy_hold_return_pct']:+.2f}%")
    print(f"   Alpha:           {m['alpha_pct']:+.2f}%")
    
    # Run best genome on test data (out-of-sample)
    print("\n🧪 Test Performance (Out-of-Sample):")
    test_trader = NEATTrader(
        state_matrix=test_states,
        prices=test_prices,
        dates=test_dates,
        initial_balance=args.balance,
    )
    test_trader.best_genome = winner
    test_results = test_trader.run_best()
    m = test_results["metrics"]
    print(f"   Total Return:    {m['total_return_pct']:+.2f}%")
    print(f"   Sharpe Ratio:    {m['sharpe_ratio']:.3f}")
    print(f"   Max Drawdown:    {m['max_drawdown_pct']:.2f}%")
    print(f"   Total Trades:    {m['total_trades']}")
    print(f"   Win Rate:        {m['win_rate']:.1%}")
    print(f"   Buy & Hold:      {m['buy_hold_return_pct']:+.2f}%")
    print(f"   Alpha:           {m['alpha_pct']:+.2f}%")
    
    elapsed = time.time() - start_time
    print(f"\n⏱️  Total training time: {elapsed/60:.1f} minutes")
    print(f"🏁 Done! Best genome saved to checkpoints/best_genome.pkl")


if __name__ == "__main__":
    main()
