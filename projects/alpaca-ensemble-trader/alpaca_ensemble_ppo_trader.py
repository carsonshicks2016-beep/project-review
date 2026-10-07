import os
import time
import argparse
from collections import Counter
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO

from alpaca.data.historical import CryptoHistoricalDataClient
from alpaca.data.requests import CryptoBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce

# Replace the text inside the quotes with your actual Alpaca Paper Trading credentials
ALPACA_API_KEY = os.environ.get("ALPACA_API_KEY", "")
ALPACA_SECRET_KEY = os.environ.get("ALPACA_SECRET_KEY", "")

SYMBOL = "BTC/USD"
TIMEFRAME_MINUTES = 1
NUM_AGENTS = 10
LOOKBACK_WINDOW = 10

def fetch_historical_data(days_back=7):
    """Fetches historical 1-minute bars from Alpaca for training."""
    print(f"Fetching {days_back} days of historical data for {SYMBOL}...")
    client = CryptoHistoricalDataClient()
    
    end_time = datetime.now(timezone.utc)
    start_time = end_time - timedelta(days=days_back)
    
    request_params = CryptoBarsRequest(
        symbol_or_symbols=[SYMBOL],
        timeframe=TimeFrame(amount=1, unit=TimeFrameUnit.Minute),
        start=start_time,
        end=end_time
    )
    
    bars = client.get_crypto_bars(request_params)
    df = bars.df.reset_index()
    
    if df.empty:
        raise ValueError("No data fetched from Alpaca. Check API keys or timeframe.")
        
    df = df.rename(columns={'timestamp': 'date'})
    df.set_index('date', inplace=True)
    return df

def process_features(df):
    """
    Engineers technical indicators natively using pure Pandas/Numpy.
    Includes Multi-Timeframe Macro Awareness.
    """
    print("Calculating technical indicators and normalizing features...")
    df = df.copy()
    
    # 1. Log Returns
    df['log_return'] = np.log(df['close'] / df['close'].shift(1))
    
    # 2. Normalized RSI (14-period)
    delta = df['close'].diff()
    gain = delta.clip(lower=0).ewm(alpha=1/14, adjust=False).mean()
    loss = -delta.clip(upper=0).ewm(alpha=1/14, adjust=False).mean()
    rs = gain / loss
    rsi_14 = 100 - (100 / (1 + rs))
    df['rsi_norm'] = (rsi_14 / 50) - 1 
    
    # 3. MACD Histogram (Z-score standardized)
    ema12 = df['close'].ewm(span=12, adjust=False).mean()
    ema26 = df['close'].ewm(span=26, adjust=False).mean()
    macd_line = ema12 - ema26
    signal_line = macd_line.ewm(span=9, adjust=False).mean()
    df['macd_hist'] = macd_line - signal_line
    df['macd_hist_z'] = (df['macd_hist'] - df['macd_hist'].rolling(100).mean()) / df['macd_hist'].rolling(100).std()
        
    # 4. Normalized Average True Range (NATR)
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['low'] - df['close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = np.max(ranges, axis=1)
    atr = true_range.ewm(alpha=1/14, adjust=False).mean()
    df['natr_14'] = atr / df['close']
    
    # 5. On-Balance Volume Rate of Change
    obv = (np.sign(df['close'].diff()) * df['volume']).fillna(0).cumsum()
    df['obv_roc'] = obv.pct_change()
    
    # 6. Bollinger Bands %B
    rolling_mean = df['close'].rolling(window=20).mean()
    rolling_std = df['close'].rolling(window=20).std()
    upper_band = rolling_mean + (rolling_std * 2)
    lower_band = rolling_mean - (rolling_std * 2)
    df['bb_pct'] = (df['close'] - lower_band) / (upper_band - lower_band)
    
    # 7. MULTI-TIMEFRAME MACRO TREND (1-Hour)
    df_1h = df['close'].resample('60min').last()
    ema9_1h = df_1h.ewm(span=9, adjust=False).mean()
    ema21_1h = df_1h.ewm(span=21, adjust=False).mean()
    macro_trend = np.where(ema9_1h > ema21_1h, 1.0, -1.0)
    macro_series = pd.Series(macro_trend, index=df_1h.index)
    df['macro_trend_1h'] = macro_series.reindex(df.index, method='ffill').bfill()
    
    features = ['log_return', 'rsi_norm', 'macd_hist_z', 'natr_14', 'obv_roc', 'bb_pct', 'macro_trend_1h']
    
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df_clean = df[features].dropna()
    df_clean = df_clean.clip(lower=-5, upper=5)
    
    return df_clean

class CryptoTradingEnv(gym.Env):
    """Custom trading environment with Risk-Adjusted Rewards."""
    def __init__(self, df, lookback_window=10):
        super(CryptoTradingEnv, self).__init__()
        self.df = df
        self.lookback_window = lookback_window
        self.current_step = self.lookback_window
        
        self.action_space = spaces.Discrete(3) # 0 = Hold, 1 = Buy, 2 = Sell
        self.observation_space = spaces.Box(
            low=-10.0, high=10.0, 
            shape=(self.lookback_window, len(self.df.columns)), 
            dtype=np.float32
        )
        self.current_position = 0 
        self.time_in_trade = 0 # NEW: Tracker for how long the AI holds

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = self.lookback_window
        self.current_position = 0
        self.time_in_trade = 0 # Reset tracker on episode restart
        return self._next_observation(), {}

    def _next_observation(self):
        obs = self.df.iloc[self.current_step - self.lookback_window : self.current_step].values
        return np.array(obs, dtype=np.float32)

    def step(self, action):
        self.current_step += 1
        
        if self.current_step < len(self.df) - 1:
            step_return = self.df.iloc[self.current_step]['log_return']
        else:
            step_return = 0.0
            
        reward = 0.0
        trade_fee = 0.001 
        time_penalty_base = 0.0001 
        drawdown_punishment = 2.0 # Punish losses 2x more than gains
        
        # 1. State Transitions & Fees
        if action == 1 and self.current_position == 0: # Entering a Long
            reward -= trade_fee
            self.current_position = 1
            self.time_in_trade = 0 # Reset timer upon entry
            
        elif action == 2 and self.current_position == 1: # Exiting a Long
            reward -= trade_fee
            self.current_position = 0
            self.time_in_trade = 0 
            
        # 2. Advanced Reward Mechanics (Active Position)
        if self.current_position == 1:
            self.time_in_trade += 1
            
            # Sortino-Style Risk Adjustment
            if step_return > 0:
                reward += step_return # Standard reward for profit
            else:
                reward += (step_return * drawdown_punishment) # Severe punishment for drawdown
                
            # Compounding Time-in-Market Penalty
            reward -= (time_penalty_base * self.time_in_trade)
                
        done = self.current_step >= len(self.df) - 2
        truncated = False
        
        return self._next_observation(), reward, done, truncated, {}

def train_ensemble(num_agents=10):
    """Trains an ensemble of PPO agents on the historical dataset."""
    print("--- STARTING ENSEMBLE TRAINING ---")
    raw_data = fetch_historical_data(days_back=3) 
    rl_data = process_features(raw_data)
    
    env = CryptoTradingEnv(rl_data, lookback_window=LOOKBACK_WINDOW)
    
    log_dir = "./tensorboard_logs/"
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs("./models/", exist_ok=True)

    models = []
    for i in range(num_agents):
        print(f"\n--- Training Agent {i+1} of {num_agents} ---")
        
        model = PPO(
            "MlpPolicy", 
            env, 
            verbose=0, 
            seed=i, 
            tensorboard_log=log_dir,
            learning_rate=0.0003,
            n_steps=2048
        )
        
        model.learn(total_timesteps=50_000, tb_log_name=f"Agent_{i}")
        
        model_path = f"./models/ppo_agent_{i}"
        model.save(model_path)
        models.append(model)
        print(f"Agent {i+1} saved to {model_path}.zip")
        
    print("\nTraining Complete! Run 'tensorboard --logdir ./tensorboard_logs/' to view Visualizer 1.")
    return models

def load_models(num_agents=10):
    """Loads previously trained agents from disk."""
    models = []
    for i in range(num_agents):
        try:
            model = PPO.load(f"./models/ppo_agent_{i}")
            models.append(model)
        except Exception as e:
            print(f"Failed to load model {i}: {e}. Have you run training mode yet?")
            exit(1)
    return models

def get_live_state(data_client, lookback=10):
    """Fetches the most recent 1m bars and formats them for the agents."""
    end_time = datetime.now(timezone.utc)
    start_time = end_time - timedelta(minutes=200) 
    
    request = CryptoBarsRequest(
        symbol_or_symbols=[SYMBOL],
        timeframe=TimeFrame(amount=1, unit=TimeFrameUnit.Minute),
        start=start_time,
        end=end_time
    )
    
    bars = data_client.get_crypto_bars(request).df.reset_index()
    if bars.empty:
        return None
        
    bars.set_index('timestamp', inplace=True)
    features_df = process_features(bars)
    
    if len(features_df) < lookback:
        return None
        
    return features_df.iloc[-lookback:].values

def run_live_trading():
    """Connects to Alpaca, gathers votes, and executes majority decisions."""
    print("--- INITIALIZING LIVE AUTONOMOUS TRADING ---")
    trading_client = TradingClient(ALPACA_API_KEY, ALPACA_SECRET_KEY, paper=True)
    data_client = CryptoHistoricalDataClient()
    
    try:
        account = trading_client.get_account()
        print(f"Connected to Alpaca! Paper Balance: ${account.portfolio_value}")
    except Exception as e:
        print(f"Failed to connect to Alpaca: {e}")
        return

    models = load_models(NUM_AGENTS)
    print(f"Successfully loaded {len(models)} agents. Starting 1-minute loop...")
    
    while True:
        loop_start_time = time.time()
        
        try:
            state = get_live_state(data_client, LOOKBACK_WINDOW)
            
            if state is not None:
                votes = []
                for model in models:
                    action, _ = model.predict(state, deterministic=True)
                    votes.append(int(action))
                
                vote_counts = Counter(votes)
                majority_action = vote_counts.most_common(1)[0][0]
                
                timestamp_str = datetime.now(timezone.utc).strftime('%H:%M:%S')
                
                with open("live_votes.log", "w") as f:
                    f.write(f"{','.join(map(str, votes))},{majority_action}\n")
                
                with open("vote_history.csv", "a") as f:
                    f.write(f"{timestamp_str},{','.join(map(str, votes))},{majority_action}\n")
                
                action_map = {0: "HOLD", 1: "BUY", 2: "SELL"}
                print(f"[{datetime.now(timezone.utc).strftime('%H:%M:%S')}] Votes: {votes} -> Majority: {action_map[majority_action]}")
                
                if majority_action == 1:
                    trading_client.submit_order(
                        MarketOrderRequest(symbol=SYMBOL, qty=0.005, side=OrderSide.BUY, time_in_force=TimeInForce.GTC)
                    )
                elif majority_action == 2:
                    try:
                        trading_client.submit_order(
                            MarketOrderRequest(symbol=SYMBOL, qty=0.005, side=OrderSide.SELL, time_in_force=TimeInForce.GTC)
                        )
                    except Exception:
                        pass 
                        
            else:
                print("Not enough data to calculate state. Waiting...")
                
        except Exception as e:
            print(f"Error in trading loop: {e}")
            
        elapsed = time.time() - loop_start_time
        sleep_time = max(0, 60.0 - elapsed)
        time.sleep(sleep_time)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Alpaca Ensemble PPO Trader")
    parser.add_argument('--mode', type=str, choices=['train', 'trade'], required=True)
    
    args = parser.parse_args()
    
    if args.mode == 'train':
        train_ensemble(num_agents=NUM_AGENTS)
    elif args.mode == 'trade':
        run_live_trading()
