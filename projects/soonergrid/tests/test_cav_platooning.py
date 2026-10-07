"""
Unit tests for Step 5 CAV Platooning, Headway Scaling, and Mixed Autonomy.
"""

import unittest
from soonergrid.physics.cav_platooning import CAVPlatooningModel


class TestCAVPlatooning(unittest.TestCase):

    def setUp(self):
        self.cav = CAVPlatooningModel(
            cav_penetration_rate=0.0,
            h_hdv_s=1.50,
            h_mixed_s=1.10,
            h_cav_s=0.60,
        )

    def test_headway_monotonic_decrease(self):
        """Mean headway must strictly decrease as CAV penetration increases."""
        h_0 = self.cav.compute_mean_headway_s(0.0)
        h_25 = self.cav.compute_mean_headway_s(0.25)
        h_50 = self.cav.compute_mean_headway_s(0.50)
        h_75 = self.cav.compute_mean_headway_s(0.75)
        h_100 = self.cav.compute_mean_headway_s(1.0)

        self.assertEqual(h_0, 1.50)
        self.assertLess(h_25, h_0)
        self.assertLess(h_50, h_25)
        self.assertLess(h_75, h_50)
        self.assertLess(h_100, h_75)
        self.assertEqual(h_100, 0.60)

    def test_capacity_scaling_bounds(self):
        """Capacity multiplier must range from 1.0x at 0% CAV to 2.5x at 100% CAV."""
        cap_0 = self.cav.compute_capacity_multiplier(0.0)
        cap_50 = self.cav.compute_capacity_multiplier(0.50)
        cap_100 = self.cav.compute_capacity_multiplier(1.0)

        self.assertAlmostEqual(cap_0, 1.0, places=2)
        self.assertGreater(cap_50, 1.35)
        self.assertAlmostEqual(cap_100, 2.50, places=2)

    def test_composite_fleet_compliance(self):
        """At 100% CAV penetration, fleet compliance must be 1.0 regardless of human compliance."""
        # 100% CAV with 20% human compliance
        c_full = self.cav.compute_fleet_compliance(human_compliance=0.20, p=1.0)
        self.assertEqual(c_full, 1.0)

        # 0% CAV with 40% human compliance
        c_zero = self.cav.compute_fleet_compliance(human_compliance=0.40, p=0.0)
        self.assertEqual(c_zero, 0.40)

        # 50% CAV with 40% human compliance -> 0.5*0.4 + 0.5*1.0 = 0.70
        c_half = self.cav.compute_fleet_compliance(human_compliance=0.40, p=0.50)
        self.assertAlmostEqual(c_half, 0.70, places=2)


if __name__ == "__main__":
    unittest.main()
