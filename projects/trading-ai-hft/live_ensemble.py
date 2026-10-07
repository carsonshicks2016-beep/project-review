"""
live_ensemble.py — "The Council of 9" Live Trader

Loads an ensemble of AI models (can be a mix of NEAT and PPO brains)
and runs them on live market data. The meta-agent only executes a trade
if the majority of the bots agree (e.g., 5 out of 9 bots signal BUY).

Usage:
    python3 live_ensemble.py --dry-run --interval 300
"""

import sys, os, time, pickle, argparse
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from dotenv import load_dotenv

from stable_baselines3 import PPO
import neat

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

try:
    sys.stdout.reconfigure(line_buffering=True)
except Exception:
    pass

from data.fetch_prices import fetch_all
from features.state_builder import build_states
from features.portfolio_state import build_portfolio_features, current_allocation
from utils.action_mapping import ppo_action_to_allocation
from utils.model_registry import discover_model_paths, load_active_manifest
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce


class EnsembleCouncil:
    """Manages a portfolio and queries a committee of AI models."""
    def __init__(self, ticker, model_paths, initial_cash=15.0,
                 min_trade_value=1.0, max_order_value=None):
        self.ticker = ticker
        self.alpaca_symbol = ticker.replace("-", "/")
        self.initial_cash = initial_cash
        self.cash = initial_cash
        self.position_qty = 0.0
        self.entry_price = 0.0
        self.min_trade_value = min_trade_value
        self.max_order_value = max_order_value
        
        self.models = []
        
        # Load all models
        config_path = str(PROJECT_ROOT / "agents" / "config-trader")
        self.neat_config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                                  neat.DefaultSpeciesSet, neat.DefaultStagnation, config_path)
                                  
        for path_str in model_paths:
            path = Path(path_str)
            if not path.exists():
                print(f"⚠️ Warning: Model {path} not found.")
                continue
                
            if path.suffix == ".pkl":
                # NEAT model
                with open(path, "rb") as f:
                    genome = pickle.load(f)
                net = neat.nn.FeedForwardNetwork.create(genome, self.neat_config)
                self.models.append(("NEAT", net, path.name))
                
            elif path.suffix == ".zip":
                # PPO model
                model = PPO.load(str(path))
                self.models.append(("PPO", model, path.name))
                
        print(f"🏛️  Council for {self.ticker} formed with {len(self.models)} members.")

    @staticmethod
    def _normalize_symbol(symbol):
        return str(symbol).replace("/", "").replace("-", "").upper()

    def reconcile_broker_position(self, positions, current_price):
        """Refresh local position state from Alpaca before making decisions."""
        target = self._normalize_symbol(self.alpaca_symbol)
        matched = None
        for position in positions:
            if self._normalize_symbol(position.symbol) == target:
                matched = position
                break

        if matched is None:
            self.position_qty = 0.0
            self.entry_price = 0.0
            self.cash = self.initial_cash
            return

        self.position_qty = float(matched.qty)
        self.entry_price = float(matched.avg_entry_price)
        marked_value = self.position_qty * current_price
        self.cash = max(0.0, self.initial_cash - marked_value)

    def evaluate(self, current_state, current_price, trading_client, dry_run=True):
        obs = np.append(
            current_state,
            build_portfolio_features(self.position_qty, self.entry_price, current_price),
        ).astype(np.float32)
        
        # Collect votes
        allocations = []
        for mtype, model, name in self.models:
            if mtype == "NEAT":
                output = model.activate(obs.tolist())
                action = int(np.argmax(output))
                if action == 1:
                    allocations.append(1.0)
                elif action == 2:
                    allocations.append(0.0)
                else:
                    # HOLD translates to maintaining current allocation
                    allocations.append(current_allocation(self.cash, self.position_qty, current_price))
            elif mtype == "PPO":
                action, _ = model.predict(obs, deterministic=True)
                allocations.append(ppo_action_to_allocation(model, action))
            
        # Average allocation
        raw_target = sum(allocations) / len(allocations) if allocations else 0.0
        
        # Conviction Threshold Logic (Save on Fees)
        if raw_target >= 0.70:
            target_allocation = 1.0  # High conviction BUY
            conviction_str = "HIGH (BUY)"
        elif raw_target <= 0.30:
            target_allocation = 0.0  # High conviction SELL
            conviction_str = "HIGH (SELL)"
        else:
            # Low conviction (30% to 70%). Maintain current position (HOLD).
            target_allocation = current_allocation(self.cash, self.position_qty, current_price)
            conviction_str = "LOW (HOLD)"
            
        print(f"   🗳️  Council Average: {raw_target*100:.1f}% | Conviction: {conviction_str}")
        
        current_portfolio_value = self.cash + (self.position_qty * current_price)
        target_asset_value = current_portfolio_value * target_allocation
        current_asset_value = self.position_qty * current_price
        value_delta = target_asset_value - current_asset_value
        
        signal = "HOLD"
        
        # Only trade if the rebalance is meaningful for both fees and sizing.
        if abs(value_delta) > max(current_portfolio_value * 0.02, self.min_trade_value):
            if value_delta > 0:
                signal = "BUY"
                order_value = min(value_delta, self.max_order_value or value_delta)
                qty = order_value / current_price
                print(f"   🟢 CONSENSUS REACHED: BUY {qty:.6f} {self.alpaca_symbol} at ~${current_price:,.2f}")
                
                if not dry_run:
                    try:
                        req = MarketOrderRequest(symbol=self.alpaca_symbol, qty=qty, side=OrderSide.BUY, time_in_force=TimeInForce.GTC)
                        trading_client.submit_order(order_data=req)
                        self.position_qty += qty
                        self.cash -= order_value
                        self.entry_price = current_price # Simplified entry price
                        print("   ✅ Order submitted!")
                    except Exception as e:
                        print(f"   ❌ Order failed: {e}")
                else:
                    self.position_qty += qty
                    self.cash -= order_value
                    self.entry_price = current_price
                    print("   ✅ [DRY RUN] Virtual execution")
                    
            elif value_delta < 0:
                signal = "SELL"
                sell_value = min(abs(value_delta), self.max_order_value or abs(value_delta))
                qty = sell_value / current_price
                qty = min(qty, self.position_qty) # Ensure we don't sell more than we have
                
                print(f"   🔴 CONSENSUS REACHED: SELL {qty:.6f} {self.alpaca_symbol} at ~${current_price:,.2f}")
                
                if not dry_run:
                    try:
                        req = MarketOrderRequest(symbol=self.alpaca_symbol, qty=qty, side=OrderSide.SELL, time_in_force=TimeInForce.GTC)
                        trading_client.submit_order(order_data=req)
                        self.cash += sell_value
                        self.position_qty -= qty
                        print("   ✅ Order submitted!")
                    except Exception as e:
                        print(f"   ❌ Order failed: {e}")
                else:
                    self.cash += sell_value
                    self.position_qty -= qty
                    print("   ✅ [DRY RUN] Virtual execution")
                
        return signal


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--allow-live", action="store_true", help="Allow real-money Alpaca trading when ALPACA_PAPER=False")
    parser.add_argument("--once", action="store_true", help="Run one evaluation cycle and exit")
    parser.add_argument("--initial-cash", type=float, default=15.0, help="Virtual allocation per council")
    parser.add_argument("--min-trade-value", type=float, default=1.0, help="Skip orders smaller than this notional value")
    parser.add_argument("--max-order-value", type=float, default=15.0, help="Cap each submitted order's notional value")
    parser.add_argument("--interval", type=int, default=300, help="Seconds between evaluations (default: 300)")
    parser.add_argument("--data-interval", default="5m", choices=["1d", "5m"], help="Market data interval for live features")
    parser.add_argument("--use-all-models", action="store_true", help="Ignore models/active and load every model artifact")
    parser.add_argument("--tickers", nargs="+", default=["BTC-USD", "ETH-USD", "SOL-USD", "SPY", "NVDA", "AAPL"])
    args = parser.parse_args()

    load_dotenv()
    trading_client = None
    if not args.dry_run:
        api_key = os.getenv("ALPACA_API_KEY")
        secret_key = os.getenv("ALPACA_SECRET_KEY")
        paper = os.getenv("ALPACA_PAPER", "True").lower() == "true"

        if not api_key or not secret_key:
            print("❌ Missing Alpaca credentials in .env")
            sys.exit(1)

        if not paper and not args.allow_live:
            print("❌ Refusing real-money trading because ALPACA_PAPER is not true.")
            print("   Re-run with --allow-live only after paper validation and risk review.")
            sys.exit(1)

        trading_client = TradingClient(api_key, secret_key, paper=paper)
        account = trading_client.get_account()
        mode = "paper" if paper else "LIVE"
        print(f"✅ Connected to Alpaca {mode} account. Portfolio value: ${float(account.portfolio_value):,.2f}")
        
    print("\n   🏛️ COUNCIL ENSEMBLE INITIALIZATION")
    
    models_to_load = discover_model_paths(prefer_active=not args.use_all_models)
    manifest = load_active_manifest()
    if manifest and not args.use_all_models:
        print(f"Using active model manifest: {manifest.get('created_at', 'unknown date')}")
    print(f"Found {len(models_to_load)} models for the Council.")
    
    councils = []
    for tk in args.tickers:
        councils.append(EnsembleCouncil(
            tk,
            models_to_load,
            initial_cash=args.initial_cash,
            min_trade_value=args.min_trade_value,
            max_order_value=args.max_order_value,
        ))
    
    while True:
        now = datetime.now()
        start_date = (now - timedelta(days=60)).strftime("%Y-%m-%d")
        
        print(f"\n{'━'*60}")
        print(f"⏱️  Evaluation: {now.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'━'*60}")
        
        try:
            prices = fetch_all(args.tickers, start=start_date, force_refresh=True, interval=args.data_interval)
            broker_positions = trading_client.get_all_positions() if trading_client is not None else []
            
            for council in councils:
                state_matrix, _, close_prices = build_states(prices, sentiment_df=None, ticker=council.ticker)
                current_state = state_matrix[-1]
                current_price = close_prices[-1]
                if trading_client is not None:
                    council.reconcile_broker_position(broker_positions, current_price)
                
                print(f"\n   📊 {council.ticker} @ ${current_price:,.2f}")
                council.evaluate(current_state, current_price, trading_client, dry_run=args.dry_run)
                
        except Exception as e:
            print(f"   ❌ Error in main loop: {e}")

        if args.once:
            print("\n✅ One-shot evaluation complete.")
            break
        
        print(f"\n💤 Sleeping for {args.interval} seconds...")
        time.sleep(args.interval)

if __name__ == "__main__":
    main()
