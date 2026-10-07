"""
live_trader.py — Alpaca Live Trading Integration

This script connects your evolved NEAT neural networks to the real world
via the Alpaca API. It runs continuously, evaluating the live market
and executing trades across multiple independent AI agents.

Usage:
    python3 live_trader.py --dry-run
    python3 live_trader.py
"""

import sys, os, time, json, pickle, argparse
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from dotenv import load_dotenv

import neat
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from data.fetch_prices import fetch_all
from data.sentiment_scorer import get_daily_sentiment
from features.state_builder import build_states
from features.portfolio_state import build_portfolio_features, raw_unrealized_pnl


class AILiveAgent:
    """Represents a single instance of a trading genome."""
    def __init__(self, name, ticker, genome_path, initial_cash=5.0):
        self.name = name
        self.ticker = ticker  # e.g. 'BTC-USD'
        self.alpaca_symbol = ticker.replace("-", "/")  # Alpaca format: 'BTC/USD'
        self.genome_path = genome_path
        self.cash = initial_cash
        self.position_qty = 0.0
        self.entry_price = 0.0
        
        # Load genome and create network
        with open(genome_path, "rb") as f:
            self.genome = pickle.load(f)
        
        config_path = str(PROJECT_ROOT / "agents" / "config-trader")
        self.config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                                  neat.DefaultSpeciesSet, neat.DefaultStagnation, config_path)
        self.net = neat.nn.FeedForwardNetwork.create(self.genome, self.config)
        
        print(f"🤖 Initialized agent {self.name} on {self.ticker} with ${self.cash:.2f}")

    def evaluate(self, current_state, current_price, trading_client, dry_run=True):
        """Pass the current market state to the neural network and execute if needed."""
        # The state vector from build_states ends with [..., sentiment]
        # But during training, the TradingEnv appends [position_flag, unrealized_pnl]
        # We must reconstruct that exact vector format here.
        
        raw_pnl = raw_unrealized_pnl(self.position_qty, self.entry_price, current_price)
        obs = np.append(
            current_state,
            build_portfolio_features(self.position_qty, self.entry_price, current_price),
        )
        
        # Neural Network Forward Pass
        output = self.net.activate(obs.tolist())
        action = int(np.argmax(output))
        
        # Actions: 0 = HOLD, 1 = BUY, 2 = SELL
        signal = "HOLD"
        executed = False
        
        if action == 1 and self.position_qty == 0 and self.cash > 0:
            signal = "BUY"
            # Calculate quantity to buy with all available cash
            qty = round(self.cash / current_price, 6)
            
            print(f"\n🟢 [{self.name}] BUY SIGNAL on {self.ticker}")
            print(f"   Action: Buy {qty} {self.alpaca_symbol} at ~${current_price:,.2f}")
            
            if not dry_run:
                try:
                    order_req = MarketOrderRequest(
                        symbol=self.alpaca_symbol,
                        qty=qty,
                        side=OrderSide.BUY,
                        time_in_force=TimeInForce.GTC
                    )
                    trading_client.submit_order(order_data=order_req)
                    
                    self.position_qty = qty
                    self.entry_price = current_price
                    self.cash = 0.0
                    executed = True
                    print(f"   ✅ Order submitted to Alpaca!")
                except Exception as e:
                    print(f"   ❌ Order failed: {e}")
            else:
                self.position_qty = qty
                self.entry_price = current_price
                self.cash = 0.0
                executed = True
                print(f"   ✅ [DRY RUN] Virtual order executed.")
                
        elif action == 2 and self.position_qty > 0:
            signal = "SELL"
            qty = self.position_qty
            
            print(f"\n🔴 [{self.name}] SELL SIGNAL on {self.ticker}")
            print(f"   Action: Sell {qty} {self.alpaca_symbol} at ~${current_price:,.2f}")
            print(f"   P&L: {raw_pnl*100:+.2f}%")
            
            if not dry_run:
                try:
                    order_req = MarketOrderRequest(
                        symbol=self.alpaca_symbol,
                        qty=qty,
                        side=OrderSide.SELL,
                        time_in_force=TimeInForce.GTC
                    )
                    trading_client.submit_order(order_data=order_req)
                    
                    self.cash = qty * current_price
                    self.position_qty = 0.0
                    self.entry_price = 0.0
                    executed = True
                    print(f"   ✅ Order submitted to Alpaca!")
                except Exception as e:
                    print(f"   ❌ Order failed: {e}")
            else:
                self.cash = qty * current_price
                self.position_qty = 0.0
                self.entry_price = 0.0
                executed = True
                print(f"   ✅ [DRY RUN] Virtual order executed.")
                
        return signal, executed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="Run without sending real Alpaca orders")
    parser.add_argument("--allow-live", action="store_true", help="Allow real-money Alpaca trading when ALPACA_PAPER=False")
    parser.add_argument("--interval", type=int, default=3600, help="Seconds between evaluations (default: 1 hour)")
    args = parser.parse_args()

    load_dotenv()
    
    print("""
    ╔══════════════════════════════════════════════════════════════╗
    ║                 ⚡ LIVE AI TRADER ⚡                          ║
    ║                                                              ║
    ║   Connecting evolved NEAT genomes to Alpaca Live Markets     ║
    ╚══════════════════════════════════════════════════════════════╝
    """)

    if args.dry_run:
        print("   🧪 MODE: DRY RUN (No real orders will be sent to Alpaca)")
    else:
        print("   💸 MODE: LIVE TRADING (Executing real Alpaca paper trades)")
        
        api_key = os.getenv("ALPACA_API_KEY")
        secret_key = os.getenv("ALPACA_SECRET_KEY")
        paper = os.getenv("ALPACA_PAPER", "True").lower() == "true"

        if not paper and not args.allow_live:
            print("\n❌ Refusing real-money trading because ALPACA_PAPER is not true.")
            print("   Re-run with --allow-live only after paper validation and risk review.")
            sys.exit(1)
        
        if not api_key or not secret_key:
            print("\n❌ Error: Missing Alpaca credentials in .env file.")
            print("Please copy .env.example to .env and add your keys.")
            sys.exit(1)
            
        # Initialize Alpaca Client
        trading_client = TradingClient(api_key, secret_key, paper=paper)
        try:
            account = trading_client.get_account()
            mode = "paper" if paper else "LIVE"
            print(f"   ✅ Connected to Alpaca {mode} account! Portfolio value: ${float(account.portfolio_value):,.2f}")
            if not account.trading_blocked:
                print("   ✅ Trading is enabled.")
        except Exception as e:
            print(f"\n❌ Error connecting to Alpaca: {e}")
            sys.exit(1)

    print("\nLoading Agents...")
    
    # In the future, we will load 5 different genomes here.
    # For now, we load the single best genome on BTC, ETH, and SOL.
    best_genome_path = str(PROJECT_ROOT / "checkpoints" / "best_genome.pkl")
    if not os.path.exists(best_genome_path):
        print(f"❌ Cannot find genome at {best_genome_path}")
        sys.exit(1)

    agents = [
        AILiveAgent("Agent-BTC", "BTC-USD", best_genome_path, initial_cash=15.0),
        AILiveAgent("Agent-ETH", "ETH-USD", best_genome_path, initial_cash=15.0),
        AILiveAgent("Agent-SOL", "SOL-USD", best_genome_path, initial_cash=15.0),
    ]
    
    tickers = list(set(a.ticker for a in agents))
    
    trading_client = None if args.dry_run else trading_client

    print("\n🚀 Starting main trading loop...")
    
    while True:
        now = datetime.now()
        print(f"\n{'━'*60}")
        print(f"⏱️  Evaluation: {now.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'━'*60}")
        
        # 1. Fetch live market data (past 60 days is enough to calculate 20d moving averages)
        start_date = (now - timedelta(days=60)).strftime("%Y-%m-%d")
        print(f"   Fetching recent prices from {start_date}...")
        try:
            prices = fetch_all(tickers, start=start_date, force_refresh=True)
            
            # Group agents by ticker to save redundant state calculation
            for ticker in tickers:
                # 2. Build current state
                state_matrix, dates, close_prices = build_states(prices, sentiment_df=None, ticker=ticker)
                
                current_state = state_matrix[-1]
                current_price = close_prices[-1]
                current_date = dates[-1]
                
                print(f"   📊 {ticker} @ ${current_price:,.2f} (Latest candle: {str(current_date)[:10]})")
                
                # 3. Evaluate all agents trading this ticker
                for agent in [a for a in agents if a.ticker == ticker]:
                    signal, executed = agent.evaluate(current_state, current_price, trading_client, dry_run=args.dry_run)
                    
                    if signal == "HOLD":
                        portfolio_value = agent.cash + (agent.position_qty * current_price)
                        pos_str = f"LONG ({agent.position_qty})" if agent.position_qty > 0 else "FLAT"
                        print(f"      [{agent.name}] HOLD | Pos: {pos_str} | Value: ${portfolio_value:.2f}")

        except Exception as e:
            print(f"   ❌ Error in main loop: {e}")
            import traceback
            traceback.print_exc()

        print(f"\n💤 Sleeping for {args.interval} seconds...")
        time.sleep(args.interval)

if __name__ == "__main__":
    main()
