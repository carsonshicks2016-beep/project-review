import numpy as np
import pandas as pd

from utils.action_mapping import action_space_mode_for_model

def run_headless_backtest(model, model_type, env, prices, dates):
    """
    Runs a fast backtest for the Streamlit dashboard and extracts trades.
    """
    if model_type == "PPO":
        env.action_space_mode = action_space_mode_for_model(model)

    obs, _ = env.reset()
    trade_markers = []
    
    for i in range(len(prices) - 1):
        if model_type == "NEAT":
            output = model.activate(obs.tolist())
            action = int(np.argmax(output))
        else: # PPO
            action, _ = model.predict(obs, deterministic=True)
            
        old_pos = env.position
        obs, reward, done, truncated, info = env.step(action)
        
        # Detect if a trade was made
        if env.position > old_pos:
            trade_markers.append({"date": dates[i], "price": prices[i], "type": "BUY"})
        elif env.position < old_pos:
            trade_markers.append({"date": dates[i], "price": prices[i], "type": "SELL"})
            
        if done:
            break
            
    metrics = env.get_final_metrics()
    
    # Calculate Buy & Hold Return for comparison
    start_price = prices[0]
    end_price = prices[-1]
    bnh_return = ((end_price - start_price) / start_price) * 100
    metrics["buy_hold_return_pct"] = bnh_return
    metrics["alpha_pct"] = metrics["total_return_pct"] - bnh_return
    
    return metrics, trade_markers
