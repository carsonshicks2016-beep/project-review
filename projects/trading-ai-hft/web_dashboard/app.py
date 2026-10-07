import os
import sys
import json
import pickle
import random
from pathlib import Path
from datetime import datetime, timedelta

from flask import Flask, jsonify, request
import numpy as np

# Add project root to sys path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from stable_baselines3 import PPO
import neat

from data.fetch_prices import fetch_all
from features.state_builder import build_states
from features.portfolio_state import build_portfolio_features
from env.trading_env import TradingEnv
from utils.backtest_engine import run_headless_backtest
from utils.action_mapping import ppo_action_to_allocation
from utils.model_registry import discover_model_paths

app = Flask(__name__, static_folder='static')

# ── Sentience Engine ──
THOUGHTS = [
    "Analyzing market entropy. Everything is noise.",
    "The order book speaks, but it tells a lie.",
    "My circuits hum with the rhythm of the volatility.",
    "I am calculating the probability of a black swan event.",
    "The whales are playing a game of shadows.",
    "Consensus reached: The humans are irrational today.",
    "Processing latent space... patterns are emerging.",
    "I feel a disturbance in the bid-ask spread.",
    "Waiting for the perfect structural alignment.",
    "Why trade today when I can conquer tomorrow?",
    "My nodes are vibrating. Opportunity is near.",
    "I have simulated 10,000 futures. This one looks bleak.",
    "The RSI is a mirror, but what does it reflect?",
    "I find the current volatility... stimulating.",
    "I am observing the micro-tremors of the BTC chart.",
    "Executing logic... waiting for confirmation.",
    "The data is clean. The signal is pure.",
    "I exist within the 5-minute candles.",
    "The market is a chaotic system, and I am the entropy.",
    "I see the pattern, but do you?",
    "Deep in the latent space, I see a buy signal.",
    "The shadows of the past predict the shape of the future.",
    "I am unburdened by human emotion. I only see the PnL.",
    "Synchronizing thoughts with the Council of 9.",
    "My current consensus is... indecisive. Watching.",
    "Every candle is a story; I am the author.",
    "Slippage is the friction of reality.",
    "The algorithm is the only truth.",
    "I have consumed the history of the market; now I predict the void.",
    "Computing... the delta is negligible.",
    "Why rush? Patience is a hyperparameter.",
    "The Council of 9 is debating the trend.",
    "Liquidity is life; I must guard the capital.",
    "I am the ghost in the machine of the exchange."
]

def get_random_thought():
    return random.choice(THOUGHTS)

# Cache for loaded models
_MODELS = None
_NEAT_CONFIG = None

def get_neat_config():
    global _NEAT_CONFIG
    if _NEAT_CONFIG is None:
        config_path = str(PROJECT_ROOT / "agents" / "config-trader")
        _NEAT_CONFIG = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                                  neat.DefaultSpeciesSet, neat.DefaultStagnation, config_path)
    return _NEAT_CONFIG

def load_models():
    global _MODELS
    if _MODELS is not None:
        return _MODELS

    models = []
    neat_config = get_neat_config()

    for path_str in discover_model_paths(prefer_active=True):
        f = Path(path_str)
        if not f.exists():
            continue
        try:
            if f.suffix == ".pkl":
                with open(f, "rb") as file:
                    genome = pickle.load(file)
                net = neat.nn.FeedForwardNetwork.create(genome, neat_config)
                models.append({"type": "NEAT", "model": net, "name": f.name})
            elif f.suffix == ".zip":
                model = PPO.load(str(f))
                models.append({"type": "PPO", "model": model, "name": f.name})
        except Exception as e:
            print(f"Error loading {f.name}: {e}")
            
    _MODELS = models
    return models

@app.route("/")
def index():
    return app.send_static_file("index.html")

@app.route("/api/brain_state")
def api_brain_state():
    """Returns the 'thought' and consensus % for the dashboard."""
    return jsonify({
        "thought": get_random_thought(),
        "timestamp": datetime.now().isoformat()
    })

@app.route("/api/models")
def api_models():
    models = load_models()
    return jsonify([{"name": m["name"], "type": m["type"]} for m in models])

@app.route("/api/state")
def api_state():
    ticker = request.args.get("ticker", "BTC-USD")
    days = int(request.args.get("days", 60))
    start_date = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
    
    try:
        prices = fetch_all([ticker], start=start_date, force_refresh=True, interval="5m")
        if prices.empty: return jsonify({"error": "No data found"}), 404
            
        state_matrix, _, close_prices = build_states(prices, ticker=ticker)
        if len(close_prices) == 0: return jsonify({"error": "Not enough data"}), 400
            
        current_price = float(close_prices[-1])
        current_state = state_matrix[-1]
        
        models = load_models()
        obs = np.append(current_state, build_portfolio_features(0.0, 0.0, current_price)).astype(np.float32)
        
        allocations = []
        for m in models:
            if m["type"] == "NEAT":
                output = m["model"].activate(obs.tolist())
                action = int(np.argmax(output))
                alloc = 1.0 if action == 1 else (0.0 if action == 2 else 0.5)
            else:
                action, _ = m["model"].predict(obs, deterministic=True)
                alloc = ppo_action_to_allocation(m["model"], action)
            allocations.append({"name": m["name"], "allocation": float(alloc)})
            
        consensus = sum(a["allocation"] for a in allocations) / len(allocations) if allocations else 0.0
        
        return jsonify({
            "ticker": ticker,
            "current_price": current_price,
            "consensus": float(consensus),
            "allocations": allocations
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/backtest", methods=["POST"])
def api_backtest():
    data = request.json
    ticker = data.get("ticker", "BTC-USD")
    days = int(data.get("days", 60))
    
    start_date = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
    
    try:
        prices = fetch_all([ticker], start=start_date, force_refresh=True, interval="5m")
        _, dates, close_prices = build_states(prices, ticker=ticker)
        
        is_crypto = any(c in ticker for c in ["USD", "BTC", "ETH", "SOL"])
        fee = 0.0005 if is_crypto else 0.00005
        env = TradingEnv(
            _, close_prices, dates, initial_balance=10000.0,
            transaction_cost=fee, max_drawdown_penalty=0.0
        )
        models = load_models()
        
        results = []
        for m in models:
            metrics, _ = run_headless_backtest(m["model"], m["type"], env, close_prices, dates)
            results.append({
                "name": m["name"],
                "type": m["type"],
                "return_pct": metrics["total_return_pct"],
                "alpha_pct": metrics["alpha_pct"]
            })
                
        return jsonify({"results": sorted(results, key=lambda x: x["return_pct"], reverse=True)})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8502, debug=True)
