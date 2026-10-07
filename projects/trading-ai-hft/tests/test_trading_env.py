import unittest

import numpy as np

from env.trading_env import TradingEnv


def make_env(prices, transaction_cost=0.001):
    states = np.full((len(prices), 11), 0.5, dtype=np.float32)
    dates = [f"2026-01-{i + 1:02d}" for i in range(len(prices))]
    return TradingEnv(
        states,
        np.array(prices, dtype=np.float64),
        dates=dates,
        initial_balance=10000.0,
        transaction_cost=transaction_cost,
    )


class TradingEnvAccountingTests(unittest.TestCase):
    def test_hold_cash_stays_flat(self):
        env = make_env([100.0, 120.0, 140.0])
        env.reset()

        _, reward, done, _, _ = env.step(0)

        self.assertFalse(done)
        self.assertAlmostEqual(env.portfolio_value, 10000.0, places=6)
        self.assertAlmostEqual(reward, 0.0, places=6)
        self.assertEqual(env.total_trades, 0)

    def test_buy_reward_includes_next_price_move(self):
        env = make_env([100.0, 110.0, 120.0])
        env.reset()

        _, reward, done, _, info = env.step(1)

        self.assertFalse(done)
        self.assertGreater(reward, 0.0)
        self.assertGreater(info["portfolio_value"], 10000.0)
        self.assertEqual(env.total_trades, 1)
        self.assertEqual(env.trade_log[0]["action"], "BUY")

    def test_partial_allocation_preserves_cash_at_final_liquidation(self):
        env = make_env([100.0, 120.0])
        env.reset()

        _, reward, done, _, info = env.step(np.array([0.5], dtype=np.float32))

        self.assertTrue(done)
        self.assertGreater(info["portfolio_value"], 10000.0)
        self.assertGreater(reward, 0.0)
        self.assertAlmostEqual(env.position, 0.0, places=8)
        self.assertEqual(len(env.trade_log), 2)
        self.assertTrue(env.trade_log[-1]["forced"])

    def test_win_rate_uses_closed_trades_not_order_count(self):
        env = make_env([100.0, 120.0])
        env.reset()

        env.step(1)
        metrics = env.get_final_metrics()

        self.assertEqual(metrics["total_trades"], 2)
        self.assertEqual(metrics["closed_trades"], 1)
        self.assertAlmostEqual(metrics["win_rate"], 1.0, places=6)


if __name__ == "__main__":
    unittest.main()
