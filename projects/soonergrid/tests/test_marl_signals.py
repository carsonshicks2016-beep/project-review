"""
Unit tests for Step 4 MARL Signal Controller.
Validates green wave offset synchronization, adaptive green splits, and O(1) state lookup.
"""

import unittest
from soonergrid.policy.signal_marl_agent import MARLSignalController, MARL_SIGNAL_NODES


class TestMARLSignals(unittest.TestCase):

    def setUp(self):
        self.controller = MARLSignalController()
        # Mock edges along Lindsey St
        self.edge_meta = {
            "edge_spui": {"name": "W Lindsey St & I-35 SPUI"},
            "edge_mcgee": {"name": "W Lindsey St & McGee Dr"},
            "edge_berry": {"name": "W Lindsey St & Berry Rd"},
            "edge_jenkins": {"name": "W Lindsey St & S Jenkins Ave"},
            "edge_residential": {"name": "Elm Ave Residential"},
        }
        self.controller.register_signal_edges(self.edge_meta)

    def test_green_wave_offsets_increase_with_distance(self):
        """In inbound green wave, phase offset differences must match inter-intersection travel time modulo cycle length."""
        self.controller.synchronize_green_waves(is_egress=False)

        spui_offset = self.controller.active_offsets["SIG_I35_LINDSEY_SPUI"]
        mcgee_offset = self.controller.active_offsets["SIG_LINDSEY_MCGEE"]
        berry_offset = self.controller.active_offsets["SIG_LINDSEY_BERRY"]

        self.assertEqual(spui_offset, 0.0)
        # Expected travel time from SPUI to McGee: 650m / 13.41 m/s = 48.5s
        expected_spui_mcgee = (650.0 / 13.41) % 90.0
        self.assertAlmostEqual((mcgee_offset - spui_offset) % 90.0, expected_spui_mcgee, delta=0.2)

        # Expected travel time from McGee to Berry: 800m / 13.41 m/s = 59.7s
        expected_mcgee_berry = (800.0 / 13.41) % 90.0
        self.assertAlmostEqual((berry_offset - mcgee_offset) % 90.0, expected_mcgee_berry, delta=0.2)

    def test_egress_green_wave_reversal(self):
        """In outbound egress green wave, phase offsets reverse direction, flowing west towards I-35."""
        self.controller.synchronize_green_waves(is_egress=True)

        spui_offset = self.controller.active_offsets["SIG_I35_LINDSEY_SPUI"]
        jenkins_offset = self.controller.active_offsets["SIG_LINDSEY_JENKINS"]

        # Jenkins is near origin of outbound flow, so its offset should be small / 0
        self.assertEqual(jenkins_offset, 0.0)
        self.assertGreater(spui_offset, 0.0)

    def test_adaptive_green_split_expansion(self):
        """During peak surge, arterial green must expand up to max_green (e.g. 60-65s of 90s cycle)."""
        self.controller.adapt_green_splits({}, is_peak_surge=True)
        for node in MARL_SIGNAL_NODES:
            green = self.controller.active_greens[node.id]
            self.assertGreaterEqual(green, node.base_arterial_green_s)
            self.assertLessEqual(green, node.max_green_s)

    def test_uncontrolled_edge_returns_full_green(self):
        """Links not controlled by an intersection must return 1.0 (free flow)."""
        frac = self.controller.get_signal_green_fraction("edge_residential", 123.4)
        self.assertEqual(frac, 1.0)


if __name__ == "__main__":
    unittest.main()
