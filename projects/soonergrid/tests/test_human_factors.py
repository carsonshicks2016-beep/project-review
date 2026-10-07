"""
Unit tests for Step 5 Human Factors, Compliance, Rubbernecking, and Jaywalking.
"""

import unittest
from soonergrid.policy.human_factors import HumanFactorsManager


class TestHumanFactors(unittest.TestCase):

    def setUp(self):
        self.mgr = HumanFactorsManager(
            base_compliance_rate=0.45,
            rubbernecking_max_penalty=0.25,
            jaywalking_leakage_rate=0.10,
        )

    def test_compliance_diversion_scaling(self):
        """Effective diversion must scale linearly with driver compliance rate."""
        target_diversion = 0.50

        # At 45% compliance
        eff_45 = self.mgr.compute_effective_diversion(target_diversion, compliance_override=0.45)
        self.assertAlmostEqual(eff_45, 0.50 * 0.45, places=4)

        # At 80% compliance
        eff_80 = self.mgr.compute_effective_diversion(target_diversion, compliance_override=0.80)
        self.assertAlmostEqual(eff_80, 0.50 * 0.80, places=4)

        # At 0% compliance (complete defiance)
        eff_0 = self.mgr.compute_effective_diversion(target_diversion, compliance_override=0.0)
        self.assertEqual(eff_0, 0.0)

    def test_rubbernecking_proximity_decay(self):
        """Rubbernecking penalty must be highest directly at the stadium and decay to zero beyond 1500m."""
        # At stadium core (d = 0m): max 25% penalty -> 0.75 multiplier
        mult_stadium = self.mgr.get_rubbernecking_multiplier(0.0)
        self.assertEqual(mult_stadium, 0.75)

        # At 750m: half penalty -> 0.875 multiplier
        mult_mid = self.mgr.get_rubbernecking_multiplier(750.0)
        self.assertAlmostEqual(mult_mid, 0.875, places=3)

        # Beyond 1500m: 0 penalty -> 1.00 multiplier
        mult_far = self.mgr.get_rubbernecking_multiplier(2000.0)
        self.assertEqual(mult_far, 1.00)

    def test_pedestrian_jaywalking_leakage(self):
        """Jaywalking must reduce green vehicle phase capacity by the leakage fraction."""
        # Nominal vehicle green factor without jaywalking = 1.0
        # With 10% leakage: factor should drop to 0.90
        factor = self.mgr.get_pedestrian_factor_with_jaywalking(1.0, jaywalking_leakage_override=0.10)
        self.assertEqual(factor, 0.90)


if __name__ == "__main__":
    unittest.main()
