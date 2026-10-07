import sys, os, time, argparse, logging, numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from dotenv import load_dotenv

from stable_baselines3 import PPO
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest, GetOrdersRequest
from alpaca.trading.enums import OrderSide, TimeInForce, QueryOrderStatus

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from data.fetch_prices import fetch_all
from features.state_builder import build_states
from features.portfolio_state import build_portfolio_features
from utils.action_mapping import ppo_action_to_allocation
from utils.model_registry import discover_model_paths

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def get_alpaca_portfolio_state(trading_client, tickers):
    account = trading_client.get_account()
    portfolio_value = float(account.portfolio_value)
    positions = trading_client.get_all_positions()
    positions_dict = {}
    for pos in positions:
        symbol = pos.symbol.replace("/", "-")
        matched_ticker = next((tk for tk in tickers if tk.replace("/", "-") == symbol), None)
        if matched_ticker:
            positions_dict[matched_ticker] = {"qty": float(pos.qty), "entry_price": float(pos.avg_entry_price)}
    return portfolio_value, positions_dict

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--allow-live", action="store_true")
    parser.add_argument("--interval", type=int, default=300)
    args = parser.parse_args()

    load_dotenv()
    api_key, secret_key = os.getenv("ALPACA_API_KEY"), os.getenv("ALPACA_SECRET_KEY")
    paper = os.getenv("ALPACA_PAPER", "True").lower() == "true"
    
    if not args.allow_live and not paper:
        sys.exit("CRITICAL: Refusing to run live-money trading without --allow-live flag.")

    trading_client = TradingClient(api_key, secret_key, paper=paper)
    tickers = ["BTC-USD", "ETH-USD", "SOL-USD", "SPY", "NVDA", "AAPL"]
    models = [(Path(p).name, PPO.load(p)) for p in discover_model_paths()]

    while True:
        try:
            portfolio_value, positions = get_alpaca_portfolio_state(trading_client, tickers)
            prices = fetch_all(tickers, start=(datetime.now() - timedelta(days=5)).strftime("%Y-%m-%d"), interval="5m")

            for ticker in tickers:
                alpaca_symbol = ticker.replace("-", "/")
                
                # FIXED: This uses the correct GetOrdersRequest filter
                req = GetOrdersRequest(status=QueryOrderStatus.OPEN, symbols=[alpaca_symbol])
                if trading_client.get_orders(filter=req):
                    logging.warning(f"Pending orders for {alpaca_symbol}. Skipping.")
                    continue

                state, _, close_prices = build_states(prices, ticker=ticker)
                pos = positions.get(ticker, {"qty": 0.0, "entry_price": 0.0})
                obs = np.append(state[-1], build_portfolio_features(pos["qty"], pos["entry_price"], float(close_prices[-1])))
                
                votes = [ppo_action_to_allocation(m[1], m[1].predict(obs, deterministic=True)[0]) for m in models]
                consensus = float(np.mean(votes))
                
                target_val = portfolio_value * (consensus * 0.15)
                current_val = pos["qty"] * float(close_prices[-1])
                delta = target_val - current_val
                
                if abs(delta) > (portfolio_value * 0.01):
                    side = OrderSide.BUY if delta > 0 else OrderSide.SELL
                    qty = abs(delta) / float(close_prices[-1])
                    trading_client.submit_order(MarketOrderRequest(symbol=alpaca_symbol, qty=qty, side=side, time_in_force=TimeInForce.GTC))
        except Exception as e:
            logging.error(f"Cycle Error: {e}")
        time.sleep(args.interval)

if __name__ == "__main__":
    main()
