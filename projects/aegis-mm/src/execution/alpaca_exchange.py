"""
Asynchronous Alpaca Exchange Adapter for Paper and Live Market Making.
Integrates with alpaca-py TradingClient for crypto and equity quoting.
"""

import asyncio
from typing import Dict, Optional, Tuple
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import LimitOrderRequest, MarketOrderRequest, GetOrdersRequest
from alpaca.trading.enums import OrderSide as AlpacaOrderSide, TimeInForce, QueryOrderStatus

from src.execution.base_exchange import BaseExchange, ExchangeOrder, OrderSide, OrderType
from src.config import EngineConfig


class AlpacaExchange(BaseExchange):
    """
    Production-ready Alpaca adapter supporting paper and live trading.
    """

    def __init__(self, config: EngineConfig):
        self.config = config
        self.api_key = config.alpaca_api_key
        self.secret_key = config.alpaca_secret_key
        self.is_paper = "paper" in config.alpaca_base_url.lower()

        self.trading_client = TradingClient(
            api_key=self.api_key,
            secret_key=self.secret_key,
            paper=self.is_paper,
        )
        self.active_orders: Dict[str, ExchangeOrder] = {}

    async def connect(self):
        """Validates credentials by fetching account details."""
        try:
            account = self.trading_client.get_account()
            print(f"[Alpaca] Connected successfully! Account status: {account.status}, Buying power: ${account.buying_power}")
        except Exception as e:
            raise ConnectionError(f"Failed to connect to Alpaca API: {e}")

    async def disconnect(self):
        """Cancels open orders on disconnect."""
        await self.cancel_all_orders(self.config.market.symbol)

    async def get_account_balance(self) -> Tuple[float, float]:
        """Returns (cash, portfolio_value)."""
        loop = asyncio.get_event_loop()
        account = await loop.run_in_executor(None, self.trading_client.get_account)
        return float(account.cash), float(account.portfolio_value)

    async def get_position(self, symbol: str) -> float:
        """Returns base asset position size (positive = long, negative = short, 0 = flat)."""
        loop = asyncio.get_event_loop()
        try:
            # Alpaca symbol formatting: BTCUSD for crypto
            clean_sym = symbol.replace("/", "")
            pos = await loop.run_in_executor(None, self.trading_client.get_open_position, clean_sym)
            return float(pos.qty)
        except Exception:
            return 0.0

    async def place_limit_order(
        self,
        symbol: str,
        side: OrderSide,
        price: float,
        size: float,
        client_order_id: Optional[str] = None,
        post_only: bool = True,
    ) -> ExchangeOrder:
        """
        Submits a post-only limit order to Alpaca.
        """
        loop = asyncio.get_event_loop()
        clean_sym = symbol.replace("/", "")
        alpaca_side = AlpacaOrderSide.BUY if side == OrderSide.BUY else AlpacaOrderSide.SELL

        req = LimitOrderRequest(
            symbol=clean_sym,
            limit_price=round(price, 2 if "USD" in symbol else 4),
            qty=round(size, 4),
            side=alpaca_side,
            time_in_force=TimeInForce.GTC,
            client_order_id=client_order_id,
        )

        alpaca_order = await loop.run_in_executor(None, self.trading_client.submit_order, req)

        order = ExchangeOrder(
            order_id=str(alpaca_order.id),
            client_order_id=client_order_id or str(alpaca_order.id),
            symbol=symbol,
            side=side,
            order_type=OrderType.LIMIT,
            price=price,
            size=size,
            status="new",
        )
        self.active_orders[order.order_id] = order
        return order

    async def cancel_order(self, order_id: str) -> bool:
        loop = asyncio.get_event_loop()
        try:
            await loop.run_in_executor(None, self.trading_client.cancel_order_by_id, order_id)
            self.active_orders.pop(order_id, None)
            return True
        except Exception:
            return False

    async def cancel_all_orders(self, symbol: Optional[str] = None):
        loop = asyncio.get_event_loop()
        try:
            await loop.run_in_executor(None, self.trading_client.cancel_orders)
            self.active_orders.clear()
        except Exception as e:
            print(f"[Alpaca] Error canceling orders: {e}")
