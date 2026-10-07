"""
technical_indicators.py — Compute standard technical indicators from OHLCV data.

All indicators are normalized to [0, 1] for neural network input.
"""

import pandas as pd
import numpy as np


def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Relative Strength Index — momentum oscillator (0-100, normalized to 0-1)."""
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    
    avg_gain = gain.rolling(window=period, min_periods=1).mean()
    avg_loss = loss.rolling(window=period, min_periods=1).mean()
    
    rs = avg_gain / (avg_loss + 1e-10)  # Avoid division by zero
    rsi = 100 - (100 / (1 + rs))
    
    return rsi / 100.0  # Normalize to [0, 1]


def compute_macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    """
    MACD — trend-following momentum indicator.
    
    Returns:
        macd_line: Normalized MACD line
        signal_line: Normalized signal line  
        histogram: Normalized histogram (MACD - signal)
    """
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    
    # Normalize using rolling z-score (scale-independent)
    def zscore_normalize(s, window=100):
        mean = s.rolling(window=window, min_periods=1).mean()
        std = s.rolling(window=window, min_periods=1).std().replace(0, 1)
        z = (s - mean) / std
        # Sigmoid to squash to [0, 1]
        return 1 / (1 + np.exp(-z))
    
    return (
        zscore_normalize(macd_line),
        zscore_normalize(signal_line),
        zscore_normalize(histogram),
    )


def compute_bollinger_bands(series: pd.Series, period: int = 20, std_dev: int = 2):
    """
    Bollinger Bands — volatility indicator.
    
    Returns:
        bb_position: Where price is within the bands (0 = lower, 1 = upper)
        bb_width: Band width (normalized)
    """
    sma = series.rolling(window=period, min_periods=1).mean()
    std = series.rolling(window=period, min_periods=1).std().fillna(0)
    
    upper = sma + std_dev * std
    lower = sma - std_dev * std
    
    # Position within bands: 0 = at lower, 1 = at upper
    band_range = upper - lower
    bb_position = (series - lower) / (band_range + 1e-10)
    bb_position = bb_position.clip(0, 1)
    
    # Band width (normalized by price)
    bb_width = band_range / (sma + 1e-10)
    # Normalize width to [0, 1] using sigmoid
    bb_width = 1 / (1 + np.exp(-10 * (bb_width - bb_width.rolling(100, min_periods=1).mean())))
    
    return bb_position, bb_width


def compute_volume_zscore(volume: pd.Series, window: int = 20) -> pd.Series:
    """Volume Z-Score — detects unusual volume spikes."""
    mean = volume.rolling(window=window, min_periods=1).mean()
    std = volume.rolling(window=window, min_periods=1).std().replace(0, 1)
    z = (volume - mean) / std
    
    # Sigmoid to [0, 1]
    return 1 / (1 + np.exp(-z))


def compute_returns(series: pd.Series, periods: list = [1, 5, 20]) -> dict:
    """
    Price returns over multiple horizons, normalized to [0, 1].
    
    Returns:
        Dict of {f"ret_{p}d": normalized_return_series}
    """
    result = {}
    for p in periods:
        ret = series.pct_change(periods=p)
        # Normalize: sigmoid of z-scored return
        mean = ret.rolling(window=100, min_periods=1).mean()
        std = ret.rolling(window=100, min_periods=1).std().replace(0, 1)
        z = (ret - mean) / std
        result[f"ret_{p}d"] = 1 / (1 + np.exp(-z))
    
    return result


def compute_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute all technical indicators for a single ticker's OHLCV data.
    
    Args:
        df: DataFrame with columns [date, open, high, low, close, volume]
        
    Returns:
        DataFrame with original columns + all indicator columns (normalized to [0,1])
    """
    result = df.copy()
    close = result["close"]
    volume = result["volume"]
    
    # RSI
    result["rsi"] = compute_rsi(close)
    
    # MACD
    result["macd"], result["macd_signal"], result["macd_hist"] = compute_macd(close)
    
    # Bollinger Bands
    result["bb_position"], result["bb_width"] = compute_bollinger_bands(close)
    
    # Volume Z-Score
    result["vol_zscore"] = compute_volume_zscore(volume)
    
    # Multi-horizon returns
    returns = compute_returns(close)
    for name, series in returns.items():
        result[name] = series
    
    # Fill any NaN from rolling calculations
    result = result.fillna(0.5)  # 0.5 = neutral for normalized features
    
    return result


# The feature columns that the neural network will use
INDICATOR_COLUMNS = [
    "rsi",
    "macd",
    "macd_signal", 
    "macd_hist",
    "bb_position",
    "bb_width",
    "vol_zscore",
    "ret_1d",
    "ret_5d",
    "ret_20d",
]


if __name__ == "__main__":
    # Quick test with random data
    import yfinance as yf
    
    print("📊 Computing indicators for BTC-USD...")
    data = yf.download("BTC-USD", start="2023-01-01", progress=False)
    
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    
    df = pd.DataFrame({
        "date": data.index,
        "open": data["Open"].values,
        "high": data["High"].values,
        "low": data["Low"].values,
        "close": data["Close"].values,
        "volume": data["Volume"].values,
    })
    
    result = compute_all_indicators(df)
    
    print(f"\n{'='*60}")
    print(f"Computed {len(INDICATOR_COLUMNS)} indicators for {len(result)} rows")
    print(f"\nIndicator ranges (should all be ~[0, 1]):")
    for col in INDICATOR_COLUMNS:
        print(f"  {col:15s}: [{result[col].min():.3f}, {result[col].max():.3f}] mean={result[col].mean():.3f}")
