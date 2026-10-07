"""
Unit tests for pedestrian crosswalk conflicts and arterial capacity derating.
"""

import unittest
from soonergrid.demand.pedestrian_conflicts import PedestrianConflictManager


class TestPedestrianConflicts(unittest.TestCase):
    def setUp(self):
        self.mgr = PedestrianConflictManager(kickoff_time_s=14400.0, game_duration_s=11520.0)

    def test_pedestrian_surge_timing(self):
        """Foot traffic intensity must peak near T - 45 min and T_end + 10 min."""
        t_pre_peak = 14400.0 - 2700.0  # T - 45 min (11,700s)
        intensity_peak = self.mgr.get_pedestrian_surge_multiplier(t_pre_peak)
        self.assertAlmostEqual(intensity_peak, 1.0, places=2)

        # During game (e.g. halftime T + 1.5h = 19,800s), foot traffic on Lindsey crosswalks is lower
        intensity_ingame = self.mgr.get_pedestrian_surge_multiplier(19800.0)
        self.assertLess(intensity_ingame, 0.25)

    def test_capacity_derating_bounds(self):
        """Capacity factor on Lindsey St must drop during surge but never below 0.18 floor."""
        t_peak = 14400.0 - 2700.0
        factor_peak = self.mgr.get_effective_capacity_factor("W Lindsey St", t_peak)
        # Factor should drop to near 0.18
        self.assertLessEqual(factor_peak, 0.25)
        self.assertGreaterEqual(factor_peak, 0.18)

        # Non-impacted corridor maintains 1.0
        factor_highway = self.mgr.get_effective_capacity_factor("Interstate 35 Mainline", t_peak)
        self.assertEqual(factor_highway, 1.0)


if __name__ == "__main__":
    unittest.main()
