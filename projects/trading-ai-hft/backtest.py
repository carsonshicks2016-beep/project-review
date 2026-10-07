"""
backtest.py — Out-of-Sample Backtesting for the evolved NEAT trading AI.

Loads the best genome from training and evaluates it on completely unseen
data (different ticker, different timeframe) to check for overfitting.

Usage:
    python3 backtest.py                                # Default: ETH-USD
    python3 backtest.py --ticker SOL-USD               # Test on Solana
    python3 backtest.py --ticker AAPL --start 2015-01-01  # Test on Apple stock
    python3 backtest.py --tickers ETH-USD SOL-USD AAPL # Multi-asset sweep
"""

import sys, os, pickle, argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from data.fetch_prices import fetch_all
from data.fetch_news import build_news_dataset
from data.sentiment_scorer import get_daily_sentiment
from data.sentiment_loader import get_sentiment_for_ticker
from features.state_builder import build_states
from agents.neat_trader import NEATTrader, load_best_genome
import neat


# ── Styling ──────────────────────────────────────────────────
COLORS = {
    "bg":       "#0f1923",
    "card":     "#1a2332",
    "grid":     "#1e2d3d",
    "text":     "#c8d6e5",
    "dim":      "#5c6e81",
    "green":    "#00e676",
    "red":      "#ff1744",
    "blue":     "#2979ff",
    "amber":    "#ffc107",
    "cyan":     "#00e5ff",
    "purple":   "#b388ff",
    "white":    "#ffffff",
}


def load_fnspid_sentiment():
    """Try to load real FNSPID sentiment data."""
    fnspid_path = PROJECT_ROOT / "data" / "cache" / "fnspid_news.db"
    if not fnspid_path.exists():
        return None

    import sqlite3, pandas as pd
    conn = sqlite3.connect(fnspid_path)
    scored = conn.execute("SELECT COUNT(*) FROM fnspid_news WHERE sentiment_score IS NOT NULL").fetchone()[0]
    if scored < 1000:
        conn.close()
        return None

    print(f"   📰 Using FNSPID real sentiment ({scored:,} scored articles)")
    df = pd.read_sql("SELECT date, sentiment_score FROM fnspid_news WHERE sentiment_score IS NOT NULL", conn)
    df["date"] = pd.to_datetime(df["date"].str[:10], errors="coerce")
    df = df.dropna(subset=["date"])
    daily = df.groupby("date").agg(sentiment_mean=("sentiment_score", "mean")).reset_index()
    conn.close()
    return daily


def run_backtest(ticker, genome, start="2020-01-01", balance=10000.0, use_synthetic_sentiment=False):
    """Run the genome on a single ticker and return results."""
    # Fetch price data
    prices = fetch_all([ticker], start=start, force_refresh=False)

    # Load sentiment
    daily_sent, sentiment_source = get_sentiment_for_ticker(
        prices, ticker=ticker, allow_synthetic=use_synthetic_sentiment
    )
    print(f"   📰 Sentiment source for {ticker}: {sentiment_source}")

    # Build state matrix
    state_matrix, dates, close_prices = build_states(prices, sentiment_df=daily_sent, ticker=ticker)

    # Create trader and evaluate
    trader = NEATTrader(
        state_matrix=state_matrix,
        prices=close_prices,
        dates=dates,
        initial_balance=balance,
    )
    trader.best_genome = genome

    results = trader.run_best()
    return results, dates, close_prices


def plot_backtest(results, dates, close_prices, ticker, output_path):
    """Generate a premium backtest report chart."""
    metrics = results["metrics"]
    portfolio = np.array(results["portfolio_history"])
    trades = results["trade_log"]

    fig, axes = plt.subplots(3, 1, figsize=(16, 14), gridspec_kw={"height_ratios": [3, 1.5, 1]})
    fig.patch.set_facecolor(COLORS["bg"])

    for ax in axes:
        ax.set_facecolor(COLORS["card"])
        ax.tick_params(colors=COLORS["dim"], labelsize=9)
        for spine in ax.spines.values():
            spine.set_color(COLORS["grid"])
        ax.grid(True, alpha=0.15, color=COLORS["dim"])

    # ── Panel 1: Equity Curve vs Buy & Hold ──
    ax1 = axes[0]
    x = np.arange(len(portfolio))

    # Normalize both to percentage returns
    ai_returns = (portfolio / portfolio[0] - 1) * 100
    bnh_returns = (close_prices / close_prices[0] - 1) * 100

    ax1.fill_between(x, ai_returns, alpha=0.15, color=COLORS["green"])
    ax1.plot(x, ai_returns, color=COLORS["green"], linewidth=2, label=f"NEAT AI: {metrics['total_return_pct']:+,.1f}%")
    ax1.plot(x, bnh_returns, color=COLORS["blue"], linewidth=1.5, alpha=0.7, linestyle="--",
             label=f"Buy & Hold: {metrics['buy_hold_return_pct']:+,.1f}%")

    # Plot trades
    buy_steps = [t["step"] for t in trades if t["action"] == "BUY"]
    sell_steps = [t["step"] for t in trades if t["action"] == "SELL"]

    if buy_steps:
        buy_vals = [ai_returns[min(s, len(ai_returns)-1)] for s in buy_steps]
        ax1.scatter(buy_steps, buy_vals, marker="^", c=COLORS["green"], s=40, zorder=5, alpha=0.7, label=f"BUY ({len(buy_steps)})")
    if sell_steps:
        sell_vals = [ai_returns[min(s, len(ai_returns)-1)] for s in sell_steps]
        ax1.scatter(sell_steps, sell_vals, marker="v", c=COLORS["red"], s=40, zorder=5, alpha=0.7, label=f"SELL ({len(sell_steps)})")

    ax1.axhline(y=0, color=COLORS["red"], linestyle=":", alpha=0.4, linewidth=1)
    ax1.set_title(f"OUT-OF-SAMPLE BACKTEST — {ticker}", fontsize=16, fontweight="bold",
                  color=COLORS["white"], pad=15)
    ax1.set_ylabel("Return %", color=COLORS["dim"], fontsize=11)
    ax1.legend(loc="upper left", fontsize=10, facecolor=COLORS["card"], edgecolor=COLORS["grid"],
               labelcolor=COLORS["text"])

    # Add date labels on x-axis
    if len(dates) > 0:
        num_labels = min(8, len(dates))
        tick_positions = np.linspace(0, len(dates)-1, num_labels, dtype=int)
        ax1.set_xticks(tick_positions)
        ax1.set_xticklabels([str(dates[i])[:10] for i in tick_positions], rotation=30, ha="right")

    # ── Panel 2: Drawdown ──
    ax2 = axes[1]
    peak = np.maximum.accumulate(portfolio)
    drawdown = (peak - portfolio) / (peak + 1e-10) * 100

    ax2.fill_between(x, drawdown, alpha=0.4, color=COLORS["red"])
    ax2.plot(x, drawdown, color=COLORS["red"], linewidth=1)
    ax2.set_ylabel("Drawdown %", color=COLORS["dim"], fontsize=11)
    ax2.set_title("DRAWDOWN", fontsize=11, fontweight="bold", color=COLORS["dim"], loc="left")
    ax2.invert_yaxis()

    # ── Panel 3: Stats Table ──
    ax3 = axes[2]
    ax3.axis("off")

    alpha = metrics["alpha_pct"]
    alpha_color = COLORS["green"] if alpha > 0 else COLORS["red"]
    ret_color = COLORS["green"] if metrics["total_return_pct"] > 0 else COLORS["red"]

    stats_text = [
        ["Total Return", f"{metrics['total_return_pct']:+,.2f}%"],
        ["Buy & Hold", f"{metrics['buy_hold_return_pct']:+,.2f}%"],
        ["Alpha (vs B&H)", f"{alpha:+,.2f}%"],
        ["Sharpe Ratio", f"{metrics['sharpe_ratio']:.3f}"],
        ["Max Drawdown", f"{metrics['max_drawdown_pct']:.2f}%"],
        ["Total Trades", f"{metrics['total_trades']}"],
        ["Win Rate", f"{metrics['win_rate']:.1%}"],
        ["Final Value", f"${metrics['final_value']:,.2f}"],
    ]

    table = ax3.table(
        cellText=stats_text,
        colLabels=["Metric", "Value"],
        cellLoc="center",
        loc="center",
        colWidths=[0.35, 0.35],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 1.6)

    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor(COLORS["grid"])
        if row == 0:
            cell.set_facecolor(COLORS["blue"])
            cell.set_text_props(color=COLORS["white"], fontweight="bold")
        else:
            cell.set_facecolor(COLORS["card"])
            cell.set_text_props(color=COLORS["text"])
            # Color code alpha and return
            if col == 1 and row == 1:
                cell.set_text_props(color=ret_color, fontweight="bold")
            elif col == 1 and row == 3:
                cell.set_text_props(color=alpha_color, fontweight="bold")

    # ── Verdict Banner ──
    if alpha > 0 and metrics["sharpe_ratio"] > 0.5:
        verdict = "✅ PASS — AI generalizes to unseen data"
        v_color = COLORS["green"]
    elif alpha > 0:
        verdict = "⚠️ MARGINAL — Positive alpha but low Sharpe"
        v_color = COLORS["amber"]
    else:
        verdict = "❌ FAIL — AI overfit to training data"
        v_color = COLORS["red"]

    fig.text(0.5, 0.01, verdict, ha="center", fontsize=14, fontweight="bold", color=v_color)

    plt.tight_layout(rect=[0, 0.03, 1, 1])
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor=COLORS["bg"])
    plt.close()
    print(f"   📊 Chart saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Out-of-Sample Backtest")
    parser.add_argument("--ticker", type=str, default="ETH-USD",
                        help="Ticker to backtest on (default: ETH-USD)")
    parser.add_argument("--tickers", type=str, nargs="+", default=None,
                        help="Multiple tickers for a sweep test")
    parser.add_argument("--start", type=str, default="2020-01-01",
                        help="Start date (default: 2020-01-01)")
    parser.add_argument("--balance", type=float, default=10000.0,
                        help="Starting balance (default: 10000)")
    parser.add_argument("--genome", type=str, default=None,
                        help="Path to genome .pkl file")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Directory for output charts")
    parser.add_argument("--use-synthetic-sentiment", action="store_true",
                        help="Opt into price-derived synthetic sentiment for bootstrapping only")
    args = parser.parse_args()

    print("""
    ╔══════════════════════════════════════════════════════════════╗
    ║            🧪 OUT-OF-SAMPLE BACKTESTER 🧪                    ║
    ║                                                              ║
    ║   Testing if the AI learned real trading logic               ║
    ║   or just memorized the training chart                       ║
    ╚══════════════════════════════════════════════════════════════╝
    """)

    # Load genome
    if args.genome:
        with open(args.genome, "rb") as f:
            genome = pickle.load(f)
        print(f"📂 Loaded genome from {args.genome}")
    else:
        genome = load_best_genome()

    # Output directory
    output_dir = args.output_dir or str(PROJECT_ROOT / "backtest_results")
    os.makedirs(output_dir, exist_ok=True)

    # Determine tickers to test
    tickers = args.tickers or [args.ticker]

    all_results = {}

    for ticker in tickers:
        print(f"\n{'━' * 60}")
        print(f"🧪 BACKTESTING: {ticker}")
        print(f"{'━' * 60}")

        try:
            results, dates, close_prices = run_backtest(
                ticker, genome, start=args.start, balance=args.balance,
                use_synthetic_sentiment=args.use_synthetic_sentiment,
            )
            m = results["metrics"]

            all_results[ticker] = m

            # Print results
            print(f"\n   📈 {ticker} Results:")
            ret_icon = "🟢" if m["total_return_pct"] > 0 else "🔴"
            alpha_icon = "🟢" if m["alpha_pct"] > 0 else "🔴"
            print(f"   {ret_icon} Total Return:   {m['total_return_pct']:+,.2f}%")
            print(f"      Sharpe Ratio:   {m['sharpe_ratio']:.3f}")
            print(f"      Max Drawdown:   {m['max_drawdown_pct']:.2f}%")
            print(f"      Total Trades:   {m['total_trades']}")
            print(f"      Win Rate:       {m['win_rate']:.1%}")
            print(f"      Buy & Hold:     {m['buy_hold_return_pct']:+,.2f}%")
            print(f"   {alpha_icon} Alpha:          {m['alpha_pct']:+,.2f}%")
            print(f"      Final Value:    ${m['final_value']:,.2f}")

            # Generate chart
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            chart_path = os.path.join(output_dir, f"backtest_{ticker.replace('-', '_')}_{timestamp}.png")
            plot_backtest(results, dates, close_prices, ticker, chart_path)

        except Exception as e:
            print(f"   ❌ Failed: {e}")
            import traceback
            traceback.print_exc()

    # ── Summary ──
    if len(all_results) > 1:
        print(f"\n{'═' * 60}")
        print(f"📋 MULTI-ASSET SUMMARY")
        print(f"{'═' * 60}")
        print(f"   {'Ticker':<12} {'Return':>10} {'Alpha':>10} {'Sharpe':>8} {'Trades':>8} {'Win%':>8}")
        print(f"   {'─'*12} {'─'*10} {'─'*10} {'─'*8} {'─'*8} {'─'*8}")

        for ticker, m in all_results.items():
            print(f"   {ticker:<12} {m['total_return_pct']:>+9.1f}% {m['alpha_pct']:>+9.1f}% "
                  f"{m['sharpe_ratio']:>7.3f} {m['total_trades']:>8} {m['win_rate']:>7.1%}")

        # Overall verdict
        alphas = [m["alpha_pct"] for m in all_results.values()]
        avg_alpha = np.mean(alphas)
        positive = sum(1 for a in alphas if a > 0)

        print(f"\n   Average Alpha: {avg_alpha:+.2f}%")
        print(f"   Positive Alpha: {positive}/{len(alphas)} assets")

        if positive == len(alphas):
            print(f"\n   ✅ VERDICT: AI generalizes across all tested assets!")
        elif positive > len(alphas) / 2:
            print(f"\n   ⚠️ VERDICT: Mixed results — AI partially generalizes")
        else:
            print(f"\n   ❌ VERDICT: AI likely overfit to training data")

    print(f"\n🏁 Backtest complete! Charts saved to {output_dir}/")


if __name__ == "__main__":
    main()
