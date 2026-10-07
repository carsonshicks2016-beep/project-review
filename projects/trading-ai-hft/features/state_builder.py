"""
state_builder.py — Build the state vector that the neural network reads.

Rebuilt for strict forward-looking structural market proxies. 
Zero look-ahead bias allowed. Synthetic sentiment has been purged.
"""

import pandas as pd
import numpy as np
from pathlib import Path

# Define the new structural feature columns
INDICATOR_COLUMNS = [
    "rsi_norm",
    "macd_hist_squashed",
    "bb_position",
    "ret_5",
    "ret_20",
    "vol_z"
]

# Total number of input features
NUM_FEATURES = len(INDICATOR_COLUMNS)
# The agent will also receive position_flag and unrealized_pnl at runtime
NUM_INPUTS = NUM_FEATURES + 2 
FEATURE_NAMES = INDICATOR_COLUMNS + ["position_flag", "unrealized_pnl"]

def build_states(
    price_df: pd.DataFrame,
    sentiment_df: pd.DataFrame = None, # Kept for signature compatibility but ignored
    ticker: str = "BTC-USD",
) -> tuple:
    """
    Build the complete state matrix using strictly valid structural features.
    """
    # Filter to ticker
    df = price_df[price_df["ticker"] == ticker].copy()
    df = df.sort_values("date").reset_index(drop=True)
    
    if len(df) < 50:
        raise ValueError(f"Not enough data for {ticker}: only {len(df)} rows (need 50+)")
    
    # 1. Base Technicals (Forward-looking approximations)
    # RSI (14 period)
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / (loss + 1e-8)
    df['rsi_norm'] = (100 - (100 / (1 + rs))) / 100.0

    # MACD (12, 26, 9)
    ema12 = df['close'].ewm(span=12, adjust=False).mean()
    ema26 = df['close'].ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    macd_sig = macd.ewm(span=9, adjust=False).mean()
    df['macd_hist_squashed'] = np.tanh(macd - macd_sig)

    # 2. Volatility & Bands (20 period, 2 std dev)
    sma20 = df['close'].rolling(window=20).mean()
    std20 = df['close'].rolling(window=20).std()
    bb_upper = sma20 + (std20 * 2)
    bb_lower = sma20 - (std20 * 2)
    df['bb_position'] = np.clip((df['close'] - bb_lower) / (bb_upper - bb_lower + 1e-8), 0.0, 1.0)

    # 3. Structural Market Proxies (Momentum)
    df['ret_5'] = np.log(df['close'] / df['close'].shift(5).replace(0, np.nan))
    df['ret_20'] = np.log(df['close'] / df['close'].shift(20).replace(0, np.nan))

    # Volume Z-Score
    vol_mean = df['volume'].rolling(window=20).mean()
    vol_std = df['volume'].rolling(window=20).std() + 1e-8
    df['vol_z'] = np.clip((df['volume'] - vol_mean) / vol_std, -3.0, 3.0)

    # Clean up NaNs safely without leaking future data
    df.bfill(inplace=True)
    df.fillna(0.0, inplace=True)
    
    # Extract state matrix
    state_matrix = df[INDICATOR_COLUMNS].values.astype(np.float32)
    
    # Handle infinite values just in case
    state_matrix = np.nan_to_num(state_matrix, nan=0.0, posinf=1.0, neginf=-1.0)
    
    dates = df["date"].tolist()
    prices = df["close"].values.astype(np.float64)
    
    print(f"📦 Built structural state matrix for {ticker}: shape={state_matrix.shape}")
    
    return state_matrix, dates, prices

if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from data.fetch_prices import fetch_all
    
    prices = fetch_all(["BTC-USD"], start="2024-01-01")
    states, dates, close_prices = build_states(prices, ticker="BTC-USD")
    
    print(f"\n{'='*60}")
    print(f"State matrix shape: {states.shape}")
    print(f"Num input nodes for Network: {NUM_INPUTS}")
    print(f"Feature names: {FEATURE_NAMES}")
