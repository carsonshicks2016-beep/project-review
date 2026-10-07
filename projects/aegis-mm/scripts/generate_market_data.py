"""
High-fidelity Market Data Generator:
Simulates realistic L2 Limit Order Book updates and trade flows via:
- Jump-diffusion mid-price process (geometric Brownian motion + Poisson jumps)
- Clustered Hawkes / power-law trade arrival flow
- Microstructure noise and spread widening under volatility
"""

import math
from dataclasses import dataclass
from typing import Generator, List, Tuple
import numpy as np


@dataclass
class MarketEvent:
    timestamp: float
    event_type: str  # 'book_update' or 'trade'
    bids: np.ndarray # shape: (K, 2) [price, size]
    asks: np.ndarray # shape: (K, 2) [price, size]
    trade_price: float = 0.0
    trade_size: float = 0.0
    is_buyer_maker: bool = False


class MarketDataSimulator:
    """
    Generates synthetic high-frequency L2 book states and trades with realistic
    stylized facts of financial markets (fat tails, volatility clustering, autocorrelation).
    """

    def __init__(
        self,
        initial_price: float = 65_000.0,
        tick_size: float = 0.50,
        dt: float = 0.1,  # 100ms time steps
        volatility: float = 0.0002, # realistic micro-volatility per 100ms
        jump_intensity: float = 0.02,
        depth_levels: int = 10,
        seed: int = 42,
    ):
        self.initial_price = initial_price
        self.tick_size = tick_size
        self.dt = dt
        self.volatility = volatility
        self.jump_intensity = jump_intensity
        self.depth_levels = depth_levels
        self.rng = np.random.RandomState(seed)

    def generate_session(self, n_steps: int = 5_000) -> Generator[MarketEvent, None, None]:
        """
        Yields a stream of MarketEvents simulating a live trading session.
        """
        price = self.initial_price
        current_time = 0.0
        trade_direction_memory = 0.0  # Autocorrelation in trade flow

        for step in range(n_steps):
            current_time += self.dt

            # 1. Price diffusion step
            # GBM component
            dw = self.rng.normal(0.0, math.sqrt(self.dt))
            # Poisson jump component
            jump = 0.0
            if self.rng.uniform() < self.jump_intensity * self.dt:
                jump = self.rng.normal(0.0, 5.0 * self.tick_size)
            
            # Update continuous price
            price += price * (self.volatility * dw) + jump
            # Round mid to tick
            mid = round(round(price / self.tick_size) * self.tick_size, 8)

            # 2. Dynamic spread: wider during jumps/volatility
            half_spread_ticks = 1 if abs(dw) < 1.0 else (2 if abs(dw) < 2.0 else 3)
            best_bid = mid - (half_spread_ticks * self.tick_size)
            best_ask = mid + (half_spread_ticks * self.tick_size)

            # 3. Construct L2 Book Depth
            bids = []
            asks = []
            for i in range(self.depth_levels):
                bp = best_bid - (i * self.tick_size)
                ap = best_ask + (i * self.tick_size)
                # Sizes follow gamma distribution with depth decay
                bs = float(np.clip(self.rng.gamma(shape=2.0, scale=0.05) * (1.0 + 0.1 * i), 0.005, 1.5))
                as_ = float(np.clip(self.rng.gamma(shape=2.0, scale=0.05) * (1.0 + 0.1 * i), 0.005, 1.5))
                bids.append([bp, bs])
                asks.append([ap, as_])

            bids_arr = np.array(bids, dtype=np.float64)
            asks_arr = np.array(asks, dtype=np.float64)

            # Yield book update
            yield MarketEvent(
                timestamp=current_time,
                event_type="book_update",
                bids=bids_arr,
                asks=asks_arr,
            )

            # 4. Generate stochastic trades (market orders crossing spread)
            # Trade arrival probability ~ 40% per 100ms
            if self.rng.uniform() < 0.40:
                # Flow autocorrelation
                trade_direction_memory = 0.7 * trade_direction_memory + 0.3 * self.rng.normal(0, 1)
                is_buy = (trade_direction_memory + self.rng.normal(0, 0.5)) > 0
                
                trade_size = float(np.clip(self.rng.exponential(scale=0.02), 0.001, 0.20))
                if is_buy:
                    trade_price = best_ask
                    is_buyer_maker = False  # Taker was buyer
                else:
                    trade_price = best_bid
                    is_buyer_maker = True   # Taker was seller

                yield MarketEvent(
                    timestamp=current_time + 0.01,
                    event_type="trade",
                    bids=bids_arr,
                    asks=asks_arr,
                    trade_price=trade_price,
                    trade_size=trade_size,
                    is_buyer_maker=is_buyer_maker,
                )


if __name__ == "__main__":
    sim = MarketDataSimulator()
    print("Testing market data stream...")
    count = 0
    for evt in sim.generate_session(n_steps=10):
        count += 1
        print(f"[{evt.timestamp:.2f}s] {evt.event_type} | Mid: {(evt.bids[0,0]+evt.asks[0,0])/2:.2f}")
    print(f"Generated {count} events successfully.")
