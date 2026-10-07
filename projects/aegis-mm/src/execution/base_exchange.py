"""
Abstract Asynchronous Exchange Interface for Market Making Execution.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Callable, Dict, List, Optional, Tuple


class OrderType(str, Enum):
    LIMIT = "limit"
    MARKET = "market"


class OrderSide(str, Enum):
    BUY = "buy"
    SELL = "sell"


@dataclass
class ExchangeOrder:
    order_id: str
    client_order_id: str
    symbol: str
    side: OrderSide
    order_type: OrderType
    price: float
    size: float
    filled_size: float = 0.0
    status: str = "new"  # 'new', 'partially_filled', 'filled', 'canceled'


class BaseExchange(ABC):
    """Abstract interface defining the methods required for an exchange connector."""

    @abstractmethod
    async def connect(self):
        """Establish WebSocket streams and REST session."""
        pass

    @abstractmethod
    async def disconnect(self):
        """Disconnect all active sockets and sessions."""
        pass

    @abstractmethod
    async def get_account_balance(self) -> Tuple[float, float]:
        """Returns (cash_balance, total_equity)."""
        pass

    @abstractmethod
    async def get_position(self, symbol: str) -> float:
        """Returns current base asset inventory/position size."""
        pass

    @abstractmethod
    async def place_limit_order(
        self,
        symbol: str,
        side: OrderSide,
        price: float,
        size: float,
        client_order_id: Optional[str] = None,
        post_only: bool = True,
    ) -> ExchangeOrder:
        """Submits a passive limit order."""
        pass

    @abstractmethod
    async def cancel_order(self, order_id: str) -> bool:
        """Cancels an open order by ID."""
        pass

    @abstractmethod
    async def cancel_all_orders(self, symbol: Optional[str] = None):
        """Cancels all active orders."""
        pass
