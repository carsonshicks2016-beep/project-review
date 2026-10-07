"""
High-resolution Institutional Terminal Dashboard for Market Making Telemetry.
Uses the 'rich' library to render real-time L2 order book ladders, active quotes,
inventory gauges, PnL attribution, and risk circuit breaker statuses.
"""

from typing import Dict, List, Optional
import numpy as np
from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich import box

from src.core.orderbook import OrderBook
from src.core.matching_engine import SimulatedMatchingEngine, FillEvent
from src.risk.risk_manager import RiskManager
from src.math.avellaneda_stoikov import ASQuotes
from src.math.microstructure import MicrostructureFeatures


class TerminalDashboard:
    """
    Renders high-frequency market making metrics into an interactive terminal view.
    """

    def __init__(self, symbol: str = "BTC/USD"):
        self.symbol = symbol
        self.console = Console()

    def build_layout(
        self,
        orderbook: OrderBook,
        matching_engine: SimulatedMatchingEngine,
        risk_manager: RiskManager,
        as_quotes: Optional[ASQuotes] = None,
        micro: Optional[MicrostructureFeatures] = None,
        recent_fills: Optional[List[FillEvent]] = None,
        step: int = 0,
    ) -> Layout:
        layout = Layout()
        layout.split_column(
            Layout(name="header", size=3),
            Layout(name="main", ratio=1),
            Layout(name="footer", size=7),
        )

        layout["main"].split_row(
            Layout(name="book_ladder", ratio=3),
            Layout(name="portfolio_telemetry", ratio=3),
        )

        # 1. Header Panel
        mid = orderbook.mid_price
        spread = orderbook.spread
        spread_bps = (spread / mid * 10_000) if mid > 0 else 0.0

        header_text = Text()
        header_text.append(" 🛡️  AEGIS-MM ", style="bold bright_cyan")
        header_text.append(f"| {self.symbol} ", style="bold white")
        header_text.append(f"| Mid: ${mid:,.2f} ", style="bold bright_yellow")
        header_text.append(f"| Spread: ${spread:.2f} ({spread_bps:.1f} bps) ", style="dim")
        header_text.append(f"| Step: {step} ", style="cyan")

        status_style = "bold green" if not risk_manager.circuit_breaker_tripped else "bold red blink"
        status_text = "NORMAL" if not risk_manager.circuit_breaker_tripped else "CIRCUIT BREAKER TRIPPED"
        header_text.append(f"| State: [{status_text}]", style=status_style)

        layout["header"].update(Panel(header_text, style="blue", box=box.ROUNDED))

        # 2. Book Depth Ladder (Left Column)
        book_table = Table(title="Level-2 Order Book Depth", box=box.SIMPLE, expand=True)
        book_table.add_column("Type", justify="center", style="bold")
        book_table.add_column("Price", justify="right")
        book_table.add_column("Size", justify="right")
        book_table.add_column("Depth Bar", justify="left")

        # Top 5 asks in reverse (so lowest ask is closest to center)
        asks = orderbook.asks[:5]
        max_size = 1.0
        if len(asks) > 0 and len(orderbook.bids) > 0:
            max_size = max(np.max(asks[:, 1]), np.max(orderbook.bids[:5, 1]), 0.01)

        for p, s in reversed(asks):
            bar_len = int((s / max_size) * 15)
            bar = "█" * max(1, bar_len)
            is_agent = any(abs(o.price - p) < 1e-4 for o in matching_engine.active_orders.values() if o.side.value == "sell")
            tag = "⚡ ASK" if is_agent else "  ASK"
            style = "bold bright_red" if is_agent else "red"
            book_table.add_row(tag, f"${p:,.2f}", f"{s:.4f}", bar, style=style)

        # Center Spread Line
        book_table.add_row("───", f"── SPREAD: ${spread:.2f} ──", "───", "───────────────", style="dim")

        # Top 5 bids
        bids = orderbook.bids[:5]
        for p, s in bids:
            bar_len = int((s / max_size) * 15)
            bar = "█" * max(1, bar_len)
            is_agent = any(abs(o.price - p) < 1e-4 for o in matching_engine.active_orders.values() if o.side.value == "buy")
            tag = "⚡ BID" if is_agent else "  BID"
            style = "bold bright_green" if is_agent else "green"
            book_table.add_row(tag, f"${p:,.2f}", f"{s:.4f}", bar, style=style)

        layout["book_ladder"].update(Panel(book_table, title="Order Book", border_style="cyan"))

        # 3. Portfolio & Performance Telemetry (Right Column)
        total_equity, unrl_pnl, rl_pnl = matching_engine.get_portfolio_value(mid)
        net_pnl = total_equity - matching_engine.initial_cash

        perf_table = Table(title="Account & Risk Engine", box=box.SIMPLE, expand=True)
        perf_table.add_column("Metric", style="bold")
        perf_table.add_column("Value", justify="right")

        perf_table.add_row("Total Portfolio Equity", f"${total_equity:,.2f}", style="bold white")
        pnl_color = "bright_green" if net_pnl >= 0 else "bright_red"
        perf_table.add_row("Net Profit & Loss", f"${net_pnl:+,.2f}", style=pnl_color)
        perf_table.add_row("Realized Spread PnL", f"${rl_pnl:+,.2f}")
        perf_table.add_row("Unrealized Position PnL", f"${unrl_pnl:+,.2f}")
        perf_table.add_row("Maker Rebates Earned", f"${matching_engine.total_maker_rebates:+,.4f}", style="bold yellow")
        perf_table.add_row("Cash Balance", f"${matching_engine.cash:,.2f}")

        # Inventory Gauge
        max_inv = risk_manager.risk_cfg.max_inventory
        inv = matching_engine.inventory
        inv_pct = (inv / max_inv) * 100.0 if max_inv > 0 else 0.0
        inv_bar = ("■" * int(abs(inv / max_inv) * 10)).ljust(10, "·")
        inv_style = "bright_green" if abs(inv_pct) < 50 else ("yellow" if abs(inv_pct) < 80 else "bright_red")
        perf_table.add_row("Inventory Position", f"{inv:+.4f} ({inv_pct:+.1f}%) [{inv_bar}]", style=inv_style)

        # Microstructure features
        if micro is not None:
            perf_table.add_row("Micro-Price Offset", f"${micro.micro_price - mid:+.2f}")
            perf_table.add_row("Order Flow Imbalance (OFI)", f"{micro.ofi:+.3f}")
            perf_table.add_row("Book Depth Imbalance", f"{micro.depth_imbalance:+.3f}")
            perf_table.add_row("Toxicity Score", f"{micro.toxicity_score:.2f}")

        # AS Quotes
        if as_quotes is not None:
            perf_table.add_row(
                "Avellaneda Optimal Quotes",
                f"B: ${as_quotes.bid_price:,.2f} | A: ${as_quotes.ask_price:,.2f}",
                style="dim cyan",
            )

        layout["portfolio_telemetry"].update(Panel(perf_table, title="Telemetry", border_style="magenta"))

        # 4. Footer Fill Stream
        fill_table = Table(box=box.SIMPLE_HEAD, expand=True)
        fill_table.add_column("Time", justify="center", style="dim")
        fill_table.add_column("Side", justify="center")
        fill_table.add_column("Price", justify="right")
        fill_table.add_column("Size", justify="right")
        fill_table.add_column("Role", justify="center")
        fill_table.add_column("Rebate/Fee", justify="right")

        fills_to_show = (recent_fills or matching_engine.fill_history)[-4:]
        if len(fills_to_show) == 0:
            fill_table.add_row("-", "NO RECENT FILLS", "-", "-", "-", "-")
        else:
            for f in reversed(fills_to_show):
                side_style = "bright_green" if f.side.value == "buy" else "bright_red"
                role_style = "bold yellow" if f.is_maker else "dim"
                role_text = "MAKER (+REBATE)" if f.is_maker and f.fee_paid < 0 else ("MAKER" if f.is_maker else "TAKER")
                fee_text = f"+${abs(f.fee_paid):.4f}" if f.fee_paid < 0 else f"-${f.fee_paid:.4f}"
                fill_table.add_row(
                    f"{f.timestamp:.1f}s",
                    f.side.value.upper(),
                    f"${f.price:,.2f}",
                    f"{f.size:.4f}",
                    role_text,
                    fee_text,
                    style=side_style,
                )

        layout["footer"].update(Panel(fill_table, title="Recent Executions & Maker Rebates", border_style="green"))

        return layout
