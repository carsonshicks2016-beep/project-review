"""
walk_forward.py — Walk-Forward Validation for the NEAT trading AI.

Instead of a simple train/test split, this trains fresh genomes on rolling
windows and tests on the next period. This reveals whether the strategy
is robust across different market regimes or only works in specific eras.

Windows:
  1. Train 2020-2022 → Test 2023
  2. Train 2020-2023 → Test 2024
  3. Train 2020-2024 → Test 2025
  4. Train 2020-2025 → Test 2026 (YTD)

Usage:
    python3 walk_forward.py
    python3 walk_forward.py --ticker BTC-USD --generations 200
    python3 walk_forward.py --tickers BTC-USD ETH-USD --generations 300
"""

import sys, os, pickle, argparse, time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

import neat
from data.fetch_prices import fetch_all
from data.fetch_news import build_news_dataset
from data.sentiment_scorer import get_daily_sentiment
from data.sentiment_loader import get_sentiment_for_ticker, load_fnspid_daily_sentiment
from features.state_builder import build_states
from env.trading_env import TradingEnv

COLORS = {
    "bg": "#0a0e17", "card": "#111827", "grid": "#1e293b",
    "text": "#e2e8f0", "dim": "#64748b", "green": "#10b981",
    "red": "#ef4444", "blue": "#3b82f6", "purple": "#8b5cf6",
    "amber": "#f59e0b", "cyan": "#06b6d4", "white": "#ffffff",
}

WINDOWS = [
    {"train_end": "2022-12-31", "test_start": "2023-01-01", "test_end": "2023-12-31", "label": "Test 2023"},
    {"train_end": "2023-12-31", "test_start": "2024-01-01", "test_end": "2024-12-31", "label": "Test 2024"},
    {"train_end": "2024-12-31", "test_start": "2025-01-01", "test_end": "2025-12-31", "label": "Test 2025"},
    {"train_end": "2025-12-31", "test_start": "2026-01-01", "test_end": "2026-12-31", "label": "Test 2026 YTD"},
]


def load_fnspid_sentiment():
    return load_fnspid_daily_sentiment()


def split_by_date(state_matrix, dates, prices, start_date, end_date):
    """Extract a slice of data between start_date and end_date."""
    import pandas as pd
    dates_dt = pd.to_datetime([str(d)[:10] for d in dates])
    mask = (dates_dt >= start_date) & (dates_dt <= end_date)
    idx = np.where(mask)[0]
    if len(idx) == 0:
        return None, None, None
    return state_matrix[idx], prices[idx], [dates[i] for i in idx]


def train_and_test(train_states, train_prices, train_dates,
                   test_states, test_prices, test_dates,
                   generations=100, balance=10000.0):
    """Train a fresh NEAT population and evaluate on test data."""
    config_path = str(PROJECT_ROOT / "agents" / "config-trader")
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation, config_path)

    pop = neat.Population(config)
    # Minimal reporting
    stats = neat.StatisticsReporter()
    pop.add_reporter(stats)

    best_genome = None
    best_fitness = float("-inf")

    def eval_genomes(genomes, cfg):
        nonlocal best_genome, best_fitness
        for gid, genome in genomes:
            net = neat.nn.FeedForwardNetwork.create(genome, cfg)
            env = TradingEnv(train_states, train_prices, train_dates, initial_balance=balance)
            obs, _ = env.reset()
            for _ in range(len(train_prices) - 1):
                output = net.activate(obs.tolist())
                action = int(np.argmax(output))
                obs, reward, done, _, _ = env.step(action)
                if done:
                    break
            m = env.get_final_metrics()
            fitness = m["total_return_pct"]
            fitness += min(20, max(0, m["sharpe_ratio"] * 5))
            fitness -= m["max_drawdown_pct"] * 0.3
            if m["total_trades"] < 3:
                fitness -= 50
            genome.fitness = fitness
            if fitness > best_fitness:
                best_fitness = fitness
                best_genome = genome

    pop.run(eval_genomes, generations)

    # Evaluate best genome on TEST data
    net = neat.nn.FeedForwardNetwork.create(best_genome, config)
    test_env = TradingEnv(test_states, test_prices, test_dates, initial_balance=balance)
    obs, _ = test_env.reset()
    for _ in range(len(test_prices) - 1):
        output = net.activate(obs.tolist())
        action = int(np.argmax(output))
        obs, reward, done, _, _ = test_env.step(action)
        if done:
            break

    test_metrics = test_env.get_final_metrics()

    # Also evaluate on training data for comparison
    train_env = TradingEnv(train_states, train_prices, train_dates, initial_balance=balance)
    obs, _ = train_env.reset()
    for _ in range(len(train_prices) - 1):
        output = net.activate(obs.tolist())
        action = int(np.argmax(output))
        obs, reward, done, _, _ = train_env.step(action)
        if done:
            break

    train_metrics = train_env.get_final_metrics()

    return {
        "train_metrics": train_metrics,
        "test_metrics": test_metrics,
        "test_portfolio": test_env.portfolio_history,
        "best_fitness": best_fitness,
        "genome": best_genome,
    }


def plot_walk_forward(results, ticker, output_path):
    """Create walk-forward validation summary chart."""
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    fig.patch.set_facecolor(COLORS["bg"])

    for ax in axes.flat:
        ax.set_facecolor(COLORS["card"])
        ax.tick_params(colors=COLORS["dim"], labelsize=9)
        for spine in ax.spines.values():
            spine.set_color(COLORS["grid"])
        ax.grid(True, alpha=0.1, color=COLORS["dim"])

    # ── Panel 1: Test Returns by Window ──
    ax1 = axes[0, 0]
    labels = [r["label"] for r in results]
    test_returns = [r["test_metrics"]["total_return_pct"] for r in results]
    bnh_returns = [r["test_metrics"]["buy_hold_return_pct"] for r in results]
    alphas = [r["test_metrics"]["alpha_pct"] for r in results]

    x = np.arange(len(labels))
    w = 0.35
    bars1 = ax1.bar(x - w/2, test_returns, w, label="AI Return", color=COLORS["green"], alpha=0.85)
    bars2 = ax1.bar(x + w/2, bnh_returns, w, label="Buy & Hold", color=COLORS["blue"], alpha=0.6)
    ax1.axhline(y=0, color=COLORS["red"], linestyle=":", alpha=0.5)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=10, color=COLORS["text"])
    ax1.set_ylabel("Return %", color=COLORS["dim"])
    ax1.set_title("RETURN BY TEST PERIOD", fontsize=13, fontweight="bold", color=COLORS["white"])
    ax1.legend(facecolor=COLORS["card"], edgecolor=COLORS["grid"], labelcolor=COLORS["text"])

    # ── Panel 2: Alpha by Window ──
    ax2 = axes[0, 1]
    bar_colors = [COLORS["green"] if a > 0 else COLORS["red"] for a in alphas]
    ax2.bar(x, alphas, color=bar_colors, alpha=0.85, width=0.5)
    ax2.axhline(y=0, color=COLORS["dim"], linestyle="-", alpha=0.5)
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, fontsize=10, color=COLORS["text"])
    ax2.set_ylabel("Alpha %", color=COLORS["dim"])
    ax2.set_title("ALPHA (AI - Buy&Hold) BY PERIOD", fontsize=13, fontweight="bold", color=COLORS["white"])

    # ── Panel 3: Sharpe & Win Rate ──
    ax3 = axes[1, 0]
    sharpes = [r["test_metrics"]["sharpe_ratio"] for r in results]
    winrates = [r["test_metrics"]["win_rate"] * 100 for r in results]

    ax3_twin = ax3.twinx()
    line1 = ax3.plot(x, sharpes, "o-", color=COLORS["cyan"], linewidth=2, markersize=8, label="Sharpe")
    line2 = ax3_twin.plot(x, winrates, "s--", color=COLORS["amber"], linewidth=2, markersize=8, label="Win Rate %")
    ax3.set_xticks(x)
    ax3.set_xticklabels(labels, fontsize=10, color=COLORS["text"])
    ax3.set_ylabel("Sharpe Ratio", color=COLORS["cyan"])
    ax3_twin.set_ylabel("Win Rate %", color=COLORS["amber"])
    ax3_twin.tick_params(colors=COLORS["dim"])
    ax3.set_title("SHARPE & WIN RATE", fontsize=13, fontweight="bold", color=COLORS["white"])
    lines = line1 + line2
    labels_legend = [l.get_label() for l in lines]
    ax3.legend(lines, labels_legend, facecolor=COLORS["card"], edgecolor=COLORS["grid"], labelcolor=COLORS["text"])

    # ── Panel 4: Train vs Test Return (Overfitting Check) ──
    ax4 = axes[1, 1]
    train_returns = [r["train_metrics"]["total_return_pct"] for r in results]
    labels_short = [r["label"].replace("Test ", "") for r in results]

    ax4.bar(x - w/2, train_returns, w, label="Train Return", color=COLORS["purple"], alpha=0.7)
    ax4.bar(x + w/2, test_returns, w, label="Test Return", color=COLORS["green"], alpha=0.85)
    ax4.set_xticks(x)
    ax4.set_xticklabels(labels, fontsize=10, color=COLORS["text"])
    ax4.set_ylabel("Return %", color=COLORS["dim"])
    ax4.set_title("TRAIN vs TEST (Overfitting Check)", fontsize=13, fontweight="bold", color=COLORS["white"])
    ax4.legend(facecolor=COLORS["card"], edgecolor=COLORS["grid"], labelcolor=COLORS["text"])

    # Verdict
    positive_alpha = sum(1 for a in alphas if a > 0)
    total = len(alphas)
    if positive_alpha == total:
        verdict = f"✅ ROBUST — Positive alpha in ALL {total} periods"
        v_color = COLORS["green"]
    elif positive_alpha > total / 2:
        verdict = f"⚠️ MIXED — Positive alpha in {positive_alpha}/{total} periods"
        v_color = COLORS["amber"]
    else:
        verdict = f"❌ FRAGILE — Positive alpha in only {positive_alpha}/{total} periods"
        v_color = COLORS["red"]

    fig.suptitle(f"WALK-FORWARD VALIDATION — {ticker}",
                fontsize=18, fontweight="bold", color=COLORS["white"], y=0.98)
    fig.text(0.5, 0.01, verdict, ha="center", fontsize=14, fontweight="bold", color=v_color)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor=COLORS["bg"])
    plt.close()
    print(f"📊 Chart saved to {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ticker", default="BTC-USD")
    parser.add_argument("--tickers", nargs="+", default=None,
                        help="Multi-ticker training per window")
    parser.add_argument("--generations", type=int, default=200,
                        help="Generations per window (default: 200)")
    parser.add_argument("--balance", type=float, default=10000.0)
    parser.add_argument("--start", default="2020-01-01")
    parser.add_argument("--use-synthetic-sentiment", action="store_true",
                        help="Opt into price-derived synthetic sentiment for bootstrapping only")
    args = parser.parse_args()

    print("""
    ╔══════════════════════════════════════════════════════════════╗
    ║          📅 WALK-FORWARD VALIDATION 📅                       ║
    ║                                                              ║
    ║   Training fresh genomes on rolling windows to test          ║
    ║   strategy robustness across market regimes                  ║
    ╚══════════════════════════════════════════════════════════════╝
    """)

    tickers = args.tickers or [args.ticker]
    primary = tickers[0]

    # Fetch all data
    print(f"📊 Fetching data for {', '.join(tickers)}...")
    prices = fetch_all(tickers, start=args.start, force_refresh=False)

    daily_sent, sentiment_source = get_sentiment_for_ticker(
        prices, ticker=primary, allow_synthetic=args.use_synthetic_sentiment
    )
    print(f"📰 Sentiment source: {sentiment_source}")

    state_matrix, dates, close_prices = build_states(prices, sentiment_df=daily_sent, ticker=primary)

    print(f"📦 Full dataset: {len(dates)} candles ({str(dates[0])[:10]} → {str(dates[-1])[:10]})")

    # Run walk-forward
    results = []
    total_start = time.time()

    for i, window in enumerate(WINDOWS):
        print(f"\n{'━' * 60}")
        print(f"📅 WINDOW {i+1}/{len(WINDOWS)}: Train → {window['train_end']}, {window['label']}")
        print(f"{'━' * 60}")

        # Split data
        train_s, train_p, train_d = split_by_date(
            state_matrix, dates, close_prices, args.start, window["train_end"])
        test_s, test_p, test_d = split_by_date(
            state_matrix, dates, close_prices, window["test_start"], window["test_end"])

        if train_s is None or len(train_s) < 100:
            print(f"   ⚠️ Not enough training data, skipping")
            continue
        if test_s is None or len(test_s) < 20:
            print(f"   ⚠️ Not enough test data, skipping")
            continue

        print(f"   Train: {len(train_p)} candles ({str(train_d[0])[:10]} → {str(train_d[-1])[:10]})")
        print(f"   Test:  {len(test_p)} candles ({str(test_d[0])[:10]} → {str(test_d[-1])[:10]})")
        print(f"   🧬 Evolving {args.generations} generations...")

        window_start = time.time()
        result = train_and_test(
            train_s, train_p, train_d,
            test_s, test_p, test_d,
            generations=args.generations,
            balance=args.balance,
        )
        elapsed = time.time() - window_start

        result["label"] = window["label"]
        results.append(result)

        tm = result["train_metrics"]
        m = result["test_metrics"]
        alpha_icon = "🟢" if m["alpha_pct"] > 0 else "🔴"

        print(f"\n   📈 Train: {tm['total_return_pct']:+,.1f}% | Sharpe: {tm['sharpe_ratio']:.3f}")
        print(f"   🧪 Test:  {m['total_return_pct']:+,.1f}% | B&H: {m['buy_hold_return_pct']:+,.1f}% | {alpha_icon} Alpha: {m['alpha_pct']:+,.1f}%")
        print(f"      Sharpe: {m['sharpe_ratio']:.3f} | Trades: {m['total_trades']} | Win: {m['win_rate']:.1%} | DD: {m['max_drawdown_pct']:.1f}%")
        print(f"   ⏱️  {elapsed:.0f}s")

    # Summary
    if results:
        print(f"\n{'═' * 60}")
        print(f"📋 WALK-FORWARD SUMMARY — {primary}")
        print(f"{'═' * 60}")
        print(f"   {'Period':<15} {'AI Return':>10} {'B&H':>10} {'Alpha':>10} {'Sharpe':>8} {'Win%':>8}")
        print(f"   {'─'*15} {'─'*10} {'─'*10} {'─'*10} {'─'*8} {'─'*8}")

        for r in results:
            m = r["test_metrics"]
            print(f"   {r['label']:<15} {m['total_return_pct']:>+9.1f}% {m['buy_hold_return_pct']:>+9.1f}% "
                  f"{m['alpha_pct']:>+9.1f}% {m['sharpe_ratio']:>7.3f} {m['win_rate']:>7.1%}")

        alphas = [r["test_metrics"]["alpha_pct"] for r in results]
        avg_alpha = np.mean(alphas)
        positive = sum(1 for a in alphas if a > 0)

        print(f"\n   Average Alpha: {avg_alpha:+.1f}%")
        print(f"   Positive Alpha: {positive}/{len(alphas)} periods")
        print(f"   Total Time: {(time.time() - total_start)/60:.1f} minutes")

        # Plot
        output_path = str(PROJECT_ROOT / "backtest_results" / f"walk_forward_{primary.replace('-', '_')}.png")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plot_walk_forward(results, primary, output_path)


if __name__ == "__main__":
    main()
