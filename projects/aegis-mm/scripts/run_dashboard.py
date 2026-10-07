"""
Entry point to launch the AegisMM Web Dashboard.
Usage:
    .venv/bin/python scripts/run_dashboard.py --port 8080
"""

import argparse
import asyncio
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.dashboard.web_server import DashboardServer


def main():
    parser = argparse.ArgumentParser(description="Launch AegisMM Web Dashboard")
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--symbol", type=str, default="BTC/USD")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_policy.pt")
    args = parser.parse_args()

    server = DashboardServer(
        symbol=args.symbol,
        checkpoint_path=args.checkpoint,
        host=args.host,
        port=args.port,
    )

    try:
        asyncio.run(server.start())
    except KeyboardInterrupt:
        print("\n[AegisMM] Dashboard stopped cleanly.")


if __name__ == "__main__":
    main()
