"""
Asynchronous Web Dashboard Server for AegisMM:
Serves the web UI and streams real-time order book ladders, active quotes,
and PnL metrics via WebSockets using aiohttp.
"""

import asyncio
import json
import os
from typing import Set
import numpy as np
from aiohttp import web

from src.config import MarketConfig, AvellanedaConfig, RiskConfig
from src.env.market_making_env import MarketMakingEnv
from src.models.agent import PPOAgent


class DashboardServer:
    def __init__(
        self,
        symbol: str = "BTC/USD",
        checkpoint_path: str = "checkpoints/best_policy.pt",
        host: str = "127.0.0.1",
        port: int = 8080,
    ):
        self.symbol = symbol
        self.host = host
        self.port = port
        self.static_dir = os.path.join(os.path.dirname(__file__), "static")

        # Initialize Environment & PPO Agent
        market_cfg = MarketConfig(symbol=symbol, maker_fee=-0.0001, initial_cash=10_000.0)
        as_cfg = AvellanedaConfig(gamma=0.01, sigma=2.50)
        risk_cfg = RiskConfig(max_inventory=0.50)

        self.env = MarketMakingEnv(
            market_cfg=market_cfg,
            as_cfg=as_cfg,
            risk_cfg=risk_cfg,
            episode_length=2_000,
            seed=42,
        )
        self.agent = PPOAgent()
        if os.path.exists(checkpoint_path):
            self.agent.load(checkpoint_path)

        self.state, _ = self.env.reset()
        self.is_running = True
        self.ws_clients: Set[web.WebSocketResponse] = set()

    async def index_handler(self, request: web.Request) -> web.Response:
        html_path = os.path.join(self.static_dir, "index.html")
        with open(html_path, "r", encoding="utf-8") as f:
            return web.Response(text=f.read(), content_type="text/html")

    async def toggle_handler(self, request: web.Request) -> web.Response:
        self.is_running = not self.is_running
        return web.json_response({"running": self.is_running})

    async def flatten_handler(self, request: web.Request) -> web.Response:
        # Flatten all inventory immediately
        inv = self.env.matching_engine.inventory
        if abs(inv) > 1e-6:
            from src.core.matching_engine import OrderSide
            side = OrderSide.SELL if inv > 0 else OrderSide.BUY
            flatten_price = self.env.orderbook.best_bid[0] if side == OrderSide.SELL else self.env.orderbook.best_ask[0]
            self.env.matching_engine.execute_market_order(
                side=side,
                price=flatten_price,
                size=abs(inv),
                timestamp=self.env.current_step * 0.1,
            )
        self.env.matching_engine.cancel_all_orders()
        return web.json_response({"status": "flattened", "inventory": 0.0})

    async def websocket_handler(self, request: web.Request) -> web.WebSocketResponse:
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        self.ws_clients.add(ws)

        try:
            async for msg in ws:
                pass
        finally:
            self.ws_clients.discard(ws)

        return ws

    async def broadcast_state(self):
        """Continuously steps the environment and streams state to connected WebSockets."""
        while True:
            await asyncio.sleep(0.1)  # 10 Hz refresh
            if not self.ws_clients:
                continue

            if self.is_running:
                action, _, _ = self.agent.select_action(self.state, deterministic=True)
                next_state, reward, terminated, truncated, info = self.env.step(action)
                self.state = next_state

                if terminated or truncated:
                    self.state, _ = self.env.reset()

            # Prepare telemetry payload
            mid = self.env.orderbook.mid_price
            spread = self.env.orderbook.spread
            spread_bps = (spread / mid * 10_000) if mid > 0 else 0.0
            eq, unrl, rl = self.env.matching_engine.get_portfolio_value(mid)

            # Active agent quotes
            agent_bid = None
            agent_ask = None
            for o in self.env.matching_engine.active_orders.values():
                if o.side.value == "buy":
                    agent_bid = o.price
                elif o.side.value == "sell":
                    agent_ask = o.price

            # Recent fills
            recent_fills = []
            for f in self.env.matching_engine.fill_history[-10:]:
                recent_fills.append({
                    "time": f"{f.timestamp:.1f}s",
                    "side": f.side.value,
                    "price": f.price,
                    "size": f.size,
                    "is_maker": f.is_maker,
                    "fee": f.fee_paid,
                })

            bids_list = self.env.orderbook.bids[:10].tolist() if len(self.env.orderbook.bids) > 0 else []
            asks_list = self.env.orderbook.asks[:10].tolist() if len(self.env.orderbook.asks) > 0 else []

            payload = {
                "equity": eq,
                "initial_cash": self.env.market_cfg.initial_cash,
                "realized_pnl": rl,
                "unrealized_pnl": unrl,
                "total_maker_rebates": self.env.matching_engine.total_maker_rebates,
                "inventory": self.env.matching_engine.inventory,
                "max_inventory": self.env.risk_cfg.max_inventory,
                "mid_price": mid,
                "spread": spread,
                "spread_bps": spread_bps,
                "ofi": self.env.last_micro.ofi if self.env.last_micro else 0.0,
                "micro_price": self.env.last_micro.micro_price if self.env.last_micro else mid,
                "toxicity": self.env.last_micro.toxicity_score if self.env.last_micro else 0.0,
                "bids": bids_list,
                "asks": asks_list,
                "agent_quotes": {"bid": agent_bid, "ask": agent_ask},
                "recent_fills": recent_fills,
            }

            text_data = json.dumps(payload)
            for ws in list(self.ws_clients):
                if not ws.closed:
                    try:
                        await ws.send_str(text_data)
                    except Exception:
                        pass

    def create_app(self) -> web.Application:
        app = web.Application()
        app.router.add_get("/", self.index_handler)
        app.router.add_get("/ws", self.websocket_handler)
        app.router.add_post("/api/toggle", self.toggle_handler)
        app.router.add_post("/api/flatten", self.flatten_handler)
        return app

    async def start(self):
        app = self.create_app()
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, self.host, self.port)
        await site.start()
        print(f"\n🚀 [AegisMM] Web Dashboard live at: http://{self.host}:{self.port}")
        await self.broadcast_state()
