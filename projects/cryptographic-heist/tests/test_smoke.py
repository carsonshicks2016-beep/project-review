from __future__ import annotations

import math
import unittest

from crypt_heist.config import get_car
from crypt_heist.physics import Controls, Vehicle
from crypt_heist.sim import HeistSim


class SmokeTests(unittest.TestCase):
    def test_evader_physics_steps(self):
        veh = Vehicle(get_car("evader"))
        veh.reset(speed=8.0)
        for _ in range(24):
            veh.step(Controls(steer=0.35, throttle=0.8, clutch=1.0))
        self.assertTrue(math.isfinite(veh.x))
        self.assertTrue(math.isfinite(veh.y))
        self.assertGreater(veh.speed, 1.0)

    def test_heist_sim_runs(self):
        sim = HeistSim(seed=3)
        for _ in range(90):
            sim.step()
        snap = sim.snapshot()
        self.assertEqual(len(snap["agents"]), 6)
        self.assertGreaterEqual(len(snap["events"]), 1)
        self.assertTrue(0.0 <= snap["confidence"] <= 1.0)


if __name__ == "__main__":
    unittest.main()
