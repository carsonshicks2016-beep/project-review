"""
Central configuration for AegisMM: Adaptive Microstructure Liquidity Provision Engine.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class ExecutionMode(str, Enum):
    SIMULATED = "simulated"
    ALPACA_PAPER = "alpaca_paper"
    ALPACA_LIVE = "alpaca_live"
    HYPERLIQUID_TESTNET = "hyperliquid_testnet"
    HYPERLIQUID_MAINNET = "hyperliquid_mainnet"


@dataclass
class MarketConfig:
    symbol: str = "BTC/USD"
    tick_size: float = 0.50
    lot_size: float = 0.001
    min_order_size: float = 0.001
    max_order_size: float = 0.100
    # Maker fee in decimal (negative means exchange pays rebate!)
    # e.g., -0.0001 = -1 bps maker rebate; 0.0 = zero fee; 0.0002 = +2 bps fee
    maker_fee: float = -0.0001
    taker_fee: float = 0.0005
    initial_cash: float = 10_000.0


@dataclass
class AvellanedaConfig:
    """Parameters for classical Avellaneda-Stoikov & Guéant-Tapia-Manziadi equations."""
    gamma: float = 0.01          # Inventory risk-aversion coefficient
    sigma: float = 2.50          # Asset dollar volatility per sqrt(second)
    kappa: float = 1.5           # Order arrival intensity parameter (dP = A * exp(-kappa * delta))
    time_horizon_sec: float = 60.0  # Terminal inventory penalty horizon (seconds)
    min_spread_ticks: int = 1    # Minimum allowed spread in ticks


@dataclass
class RLConfig:
    """Deep Reinforcement Learning (PPO) parameters."""
    state_dim: int = 16          # Dimension of normalized microstructure state
    hidden_dim: int = 128
    learning_rate: float = 3e-4
    gamma: float = 0.99          # Discount factor
    gae_lambda: float = 0.95     # Generalized Advantage Estimation lambda
    clip_epsilon: float = 0.2    # PPO surrogate objective clip epsilon
    entropy_coef: float = 0.01   # Exploration entropy regularization
    value_loss_coef: float = 0.5
    batch_size: int = 64
    n_epochs: int = 10
    total_timesteps: int = 50_000
    device: str = "mps"          # 'mps', 'cuda', or 'cpu'


@dataclass
class RiskConfig:
    """Hard institutional risk limits and circuit breakers."""
    max_inventory: float = 0.50            # Maximum inventory allowed in base asset
    max_position_notional: float = 25_000.0 # Maximum position value in USD
    max_drawdown_pct: float = 0.05         # 5% max drawdown triggers emergency liquidation
    stale_quote_timeout_sec: float = 10.0  # Cancel unrefreshed orders older than this
    stop_loss_pct: float = 0.03            # Individual trade/inventory stop-loss
    one_sided_quoting_threshold: float = 0.70 # At 70% max inventory, quote only reducing side


@dataclass
class EngineConfig:
    market: MarketConfig = field(default_factory=MarketConfig)
    avellaneda: AvellanedaConfig = field(default_factory=AvellanedaConfig)
    rl: RLConfig = field(default_factory=RLConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    execution_mode: ExecutionMode = ExecutionMode.SIMULATED
    
    # Alpaca Credentials (defaults or loaded from env)
    alpaca_api_key: Optional[str] = "PKYGTH6PFVW4I6QC52IGPCFUAO"
    alpaca_secret_key: Optional[str] = "H1fZA9QSodMh2KXZywC8H6cjRS9kpMDn4gm1xpYrTXMo"
    alpaca_base_url: str = "https://paper-api.alpaca.markets"
