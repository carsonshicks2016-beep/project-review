"""
Mathematical formulations for classical market making:
- Avellaneda & Stoikov (2008): High-frequency trading in a limit order book
- Guéant, Tapia & Manziadi (2012): Dealing with inventory risk
"""

import math
from dataclasses import dataclass
from typing import Tuple


@dataclass
class ASQuotes:
    reservation_price: float
    bid_price: float
    ask_price: float
    bid_spread: float
    ask_spread: float
    total_spread: float


class AvellanedaStoikov:
    """
    Computes optimal bid and ask quotes based on inventory risk aversion,
    asset volatility, and order arrival intensity.
    """

    def __init__(
        self,
        gamma: float = 0.05,
        sigma: float = 0.02,
        kappa: float = 1.5,
        time_horizon: float = 60.0,
        tick_size: float = 0.01,
        min_spread_ticks: int = 1,
    ):
        """
        Args:
            gamma: Inventory risk-aversion coefficient (> 0)
            sigma: Volatility of the asset price (per second or normalized)
            kappa: Liquidity/order arrival decay parameter (intensity = A * exp(-kappa * delta))
            time_horizon: Terminal inventory planning horizon in seconds (T)
            tick_size: Minimum discrete price step of the exchange
            min_spread_ticks: Minimum half-spread in tick increments
        """
        self.gamma = max(1e-6, gamma)
        self.sigma = max(1e-6, sigma)
        self.kappa = max(1e-6, kappa)
        self.time_horizon = max(1.0, time_horizon)
        self.tick_size = tick_size
        self.min_spread_ticks = min_spread_ticks

    def round_to_tick(self, price: float) -> float:
        """Round price to the nearest valid exchange tick."""
        return round(round(price / self.tick_size) * self.tick_size, 8)

    def reservation_price(self, mid_price: float, inventory: float, time_left: float) -> float:
        """
        Calculates the indifference (reservation) price r(s, q, t):
        r(s, q, t) = s - q * gamma * sigma^2 * (T - t)
        
        If long (q > 0), reservation price drops below mid to encourage selling.
        If short (q < 0), reservation price rises above mid to encourage buying.
        """
        tau = max(0.1, min(time_left, self.time_horizon))
        # Variance term
        inventory_penalty = inventory * self.gamma * (self.sigma ** 2) * tau
        return mid_price - inventory_penalty

    def optimal_spread(self, time_left: float) -> float:
        """
        Calculates the optimal total spread s*(t):
        s*(t) = gamma * sigma^2 * (T - t) + (2 / gamma) * ln(1 + gamma / kappa)
        """
        tau = max(0.1, min(time_left, self.time_horizon))
        variance_comp = self.gamma * (self.sigma ** 2) * tau
        arrival_comp = (2.0 / self.gamma) * math.log(1.0 + self.gamma / self.kappa)
        total_spread = variance_comp + arrival_comp
        min_allowed = 2 * self.min_spread_ticks * self.tick_size
        return max(min_allowed, total_spread)

    def compute_quotes(
        self,
        mid_price: float,
        inventory: float,
        time_left: float = 30.0,
    ) -> ASQuotes:
        """
        Calculates closed-form optimal bid and ask quotes for current inventory.
        """
        r = self.reservation_price(mid_price, inventory, time_left)
        s_star = self.optimal_spread(time_left)
        half_spread = s_star / 2.0

        # Raw continuous quotes centered on reservation price
        raw_ask = r + half_spread
        raw_bid = r - half_spread

        # Enforce minimum spread from mid price
        min_dist = self.min_spread_ticks * self.tick_size
        if raw_ask <= mid_price:
            raw_ask = mid_price + min_dist
        if raw_bid >= mid_price:
            raw_bid = mid_price - min_dist

        # Discrete tick-aligned quotes
        bid_price = self.round_to_tick(raw_bid)
        ask_price = self.round_to_tick(raw_ask)

        # Ensure no cross
        if bid_price >= ask_price:
            ask_price = bid_price + self.tick_size

        bid_spread = mid_price - bid_price
        ask_spread = ask_price - mid_price

        return ASQuotes(
            reservation_price=r,
            bid_price=bid_price,
            ask_price=ask_price,
            bid_spread=bid_spread,
            ask_spread=ask_spread,
            total_spread=ask_price - bid_price,
        )

    def gueant_tapia_manziadi_quotes(
        self,
        mid_price: float,
        inventory: float,
    ) -> ASQuotes:
        """
        Infinite-horizon stationary approximation from Gueant, Tapia & Manziadi (2012).
        Independent of time horizon, depends purely on stationary inventory state q.
        """
        base_spread = (1.0 / self.gamma) * math.log(1.0 + self.gamma / self.kappa)
        
        # Asymmetric inventory adjustment
        c = (self.gamma * (self.sigma ** 2)) / (2.0 * self.kappa)
        delta_ask = base_spread + (2.0 * inventory + 1.0) * c
        delta_bid = base_spread - (2.0 * inventory - 1.0) * c

        min_dist = self.min_spread_ticks * self.tick_size
        delta_ask = max(min_dist, delta_ask)
        delta_bid = max(min_dist, delta_bid)

        bid_price = self.round_to_tick(mid_price - delta_bid)
        ask_price = self.round_to_tick(mid_price + delta_ask)

        if bid_price >= ask_price:
            ask_price = bid_price + self.tick_size

        return ASQuotes(
            reservation_price=mid_price - inventory * self.gamma * (self.sigma ** 2),
            bid_price=bid_price,
            ask_price=ask_price,
            bid_spread=mid_price - bid_price,
            ask_spread=ask_price - mid_price,
            total_spread=ask_price - bid_price,
        )
