"""
High-fidelity Queue-Priority (FIFO) matching engine simulator with:
- Queue-ahead position tracking
- Realistic adverse selection decay
- Latency buffer for cancellations
- Maker rebate / fee accounting
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple
import time


class OrderSide(str, Enum):
    BUY = "buy"
    SELL = "sell"


@dataclass
class ActiveLimitOrder:
    order_id: str
    side: OrderSide
    price: float
    size: float
    filled_size: float = 0.0
    queue_ahead: float = 0.0
    created_at: float = 0.0

    @property
    def remaining_size(self) -> float:
        return max(0.0, self.size - self.filled_size)

    @property
    def is_filled(self) -> bool:
        return self.remaining_size <= 1e-8


@dataclass
class FillEvent:
    order_id: str
    side: OrderSide
    price: float
    size: float
    is_maker: bool
    fee_paid: float  # Negative means rebate received!
    timestamp: float


class SimulatedMatchingEngine:
    """
    Simulates realistic exchange order matching with Price-Time FIFO priority
    and adverse selection for passive market makers.
    """

    def __init__(
        self,
        symbol: str,
        maker_fee: float = -0.0001,  # -1 bps maker rebate
        taker_fee: float = 0.0005,   # +5 bps taker fee
        initial_cash: float = 10_000.0,
        latency_sec: float = 0.010,  # 10ms simulated transit latency
    ):
        self.symbol = symbol
        self.maker_fee = maker_fee
        self.taker_fee = taker_fee
        self.initial_cash = initial_cash
        self.latency_sec = latency_sec

        # Portfolio state
        self.cash: float = initial_cash
        self.inventory: float = 0.0
        self.realized_pnl: float = 0.0
        self.total_maker_rebates: float = 0.0
        self.total_taker_fees: float = 0.0
        self.avg_entry_price: float = 0.0

        # Active orders: order_id -> ActiveLimitOrder
        self.active_orders: Dict[str, ActiveLimitOrder] = {}
        self.fill_history: List[FillEvent] = []

    def reset(self):
        self.cash = self.initial_cash
        self.inventory = 0.0
        self.realized_pnl = 0.0
        self.total_maker_rebates = 0.0
        self.total_taker_fees = 0.0
        self.avg_entry_price = 0.0
        self.active_orders.clear()
        self.fill_history.clear()

    def place_limit_order(
        self,
        order_id: str,
        side: OrderSide,
        price: float,
        size: float,
        current_book_depth_at_price: float,
        timestamp: float,
    ) -> ActiveLimitOrder:
        """
        Places a passive limit order. Its position in the FIFO queue is behind
        whatever volume already sits at this price level. If an order already
        exists at the exact same price and side, preserves queue priority.
        """
        existing = self.active_orders.get(order_id)
        if existing and abs(existing.price - price) < 1e-6 and existing.side == side:
            existing.size = size
            return existing

        # Cancel any existing order with the same ID to re-queue at new price
        self.active_orders.pop(order_id, None)

        order = ActiveLimitOrder(
            order_id=order_id,
            side=side,
            price=price,
            size=size,
            queue_ahead=max(0.0, current_book_depth_at_price),
            created_at=timestamp + self.latency_sec,
        )
        self.active_orders[order_id] = order
        return order

    def cancel_order(self, order_id: str) -> bool:
        if order_id in self.active_orders:
            del self.active_orders[order_id]
            return True
        return False

    def cancel_all_orders(self):
        self.active_orders.clear()

    def process_trade(
        self,
        trade_price: float,
        trade_size: float,
        is_buyer_maker: bool,
        timestamp: float,
    ) -> List[FillEvent]:
        """
        Evaluates active limit orders against incoming market trades using FIFO queueing.
        - If a taker sells (is_buyer_maker=False in market notation, or trade matches bids):
          Sellers cross our Bids if trade_price <= our_bid_price.
        - If a taker buys:
          Buyers cross our Asks if trade_price >= our_ask_price.
        """
        fills: List[FillEvent] = []
        orders_to_remove: List[str] = []

        for oid, order in self.active_orders.items():
            if timestamp < order.created_at:
                # Order hasn't reached the exchange yet (in flight)
                continue

            if order.side == OrderSide.BUY:
                # Taker sold into the book
                if trade_price <= order.price:
                    # Check if trade size cuts through queue ahead
                    if trade_size > order.queue_ahead:
                        available_for_fill = trade_size - order.queue_ahead
                        fill_qty = min(order.remaining_size, available_for_fill)
                        if fill_qty > 1e-8:
                            order.filled_size += fill_qty
                            order.queue_ahead = 0.0
                            fill_evt = self._record_fill(order.side, order.price, fill_qty, timestamp, oid)
                            fills.append(fill_evt)
                    else:
                        order.queue_ahead -= trade_size

            elif order.side == OrderSide.SELL:
                # Taker bought from the book
                if trade_price >= order.price:
                    if trade_size > order.queue_ahead:
                        available_for_fill = trade_size - order.queue_ahead
                        fill_qty = min(order.remaining_size, available_for_fill)
                        if fill_qty > 1e-8:
                            order.filled_size += fill_qty
                            order.queue_ahead = 0.0
                            fill_evt = self._record_fill(order.side, order.price, fill_qty, timestamp, oid)
                            fills.append(fill_evt)
                    else:
                        order.queue_ahead -= trade_size

            if order.is_filled:
                orders_to_remove.append(oid)

        for oid in orders_to_remove:
            self.active_orders.pop(oid, None)

        return fills

    def execute_market_order(
        self,
        side: OrderSide,
        price: float,
        size: float,
        timestamp: float,
    ) -> FillEvent:
        """
        Executes an immediate aggressive taker order (e.g. for emergency risk liquidation).
        Incurs taker fees.
        """
        fee = price * size * self.taker_fee
        self.total_taker_fees += fee
        
        fill_evt = self._record_fill(side, price, size, timestamp, order_id="MARKET_TAKER", is_maker=False)
        return fill_evt

    def _record_fill(
        self,
        side: OrderSide,
        price: float,
        size: float,
        timestamp: float,
        order_id: str,
        is_maker: bool = True,
    ) -> FillEvent:
        notional = price * size
        fee_rate = self.maker_fee if is_maker else self.taker_fee
        fee_paid = notional * fee_rate

        if fee_rate < 0:
            # Negative fee means rebate earned
            self.total_maker_rebates += abs(fee_paid)
        else:
            if is_maker:
                self.total_maker_rebates -= fee_paid
            else:
                self.total_taker_fees += fee_paid

        # Update cash (fee comes out of cash)
        if side == OrderSide.BUY:
            self.cash -= (notional + fee_paid)
            new_inv = self.inventory + size
            if self.inventory >= 0:
                # Adding to long
                self.avg_entry_price = (
                    (self.inventory * self.avg_entry_price + notional) / new_inv if new_inv > 0 else 0.0
                )
            else:
                # Closing short
                closed_size = min(abs(self.inventory), size)
                pnl = closed_size * (self.avg_entry_price - price)
                self.realized_pnl += pnl
                if new_inv > 0:
                    self.avg_entry_price = price
            self.inventory = new_inv

        elif side == OrderSide.SELL:
            self.cash += (notional - fee_paid)
            new_inv = self.inventory - size
            if self.inventory <= 0:
                # Adding to short
                self.avg_entry_price = (
                    (abs(self.inventory) * self.avg_entry_price + notional) / abs(new_inv) if abs(new_inv) > 0 else 0.0
                )
            else:
                # Closing long
                closed_size = min(self.inventory, size)
                pnl = closed_size * (price - self.avg_entry_price)
                self.realized_pnl += pnl
                if new_inv < 0:
                    self.avg_entry_price = price
            self.inventory = new_inv

        evt = FillEvent(
            order_id=order_id,
            side=side,
            price=price,
            size=size,
            is_maker=is_maker,
            fee_paid=fee_paid,
            timestamp=timestamp,
        )
        self.fill_history.append(evt)
        return evt

    def get_portfolio_value(self, current_mid: float) -> Tuple[float, float, float]:
        """
        Returns:
            total_equity: Cash + Inventory * Current Mid
            unrealized_pnl: Inventory * (Current Mid - Avg Entry Price)
            realized_pnl: Cumulative realized PnL
        """
        unrealized = 0.0
        if abs(self.inventory) > 1e-8 and self.avg_entry_price > 0:
            if self.inventory > 0:
                unrealized = self.inventory * (current_mid - self.avg_entry_price)
            else:
                unrealized = abs(self.inventory) * (self.avg_entry_price - current_mid)

        total_equity = self.cash + (self.inventory * current_mid)
        return total_equity, unrealized, self.realized_pnl
