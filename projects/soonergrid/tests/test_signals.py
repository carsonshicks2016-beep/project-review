"""
Unit tests for baseline traffic signals and phase timings.
"""

import unittest
from soonergrid.sim.signal_controller import BaselineSignalController, BASELINE_INTERSECTIONS


class TestSignals(unittest.TestCase):
    def setUp(self):
        self.controller = BaselineSignalController()

    def test_cycle_durations(self):
        """All baseline intersections must have 90-second cycle times."""
        for sig in BASELINE_INTERSECTIONS:
            total_time = sig.arterial_green_s + sig.cross_green_s + sig.clearance_s
            self.assertEqual(total_time, sig.cycle_length_s)
            self.assertEqual(sig.cycle_length_s, 90.0)

    def test_signal_phase_switching(self):
        """Signal on Lindsey St at I-35 must alternate between green (1.0) and red (0.05)."""
        # At t = 10s (offset 0, arterial green 42s) -> green
        green_val = self.controller.get_signal_green_fraction("W Lindsey St", t_sim_s=10.0)
        self.assertEqual(green_val, 1.0)

        # At t = 60s (outside 42s green) -> red
        red_val = self.controller.get_signal_green_fraction("W Lindsey St", t_sim_s=60.0)
        self.assertEqual(red_val, 0.05)

    def test_uncontrolled_road_free_flow(self):
        """Roads without signal controllers must always return 1.0."""
        val = self.controller.get_signal_green_fraction("Random Residential Way", t_sim_s=45.0)
        self.assertEqual(val, 1.0)


if __name__ == "__main__":
    unittest.main()
