"""
Unit tests for Step 4 Perimeter Route Diversion and Pulsed Pedestrian Scramble.
Validates flow conservation, diversion activation, and pedestrian scramble capacity cycles.
"""

import unittest
from soonergrid.policy.perimeter_router import PerimeterRouter


class TestPerimeterRouter(unittest.TestCase):

    def setUp(self):
        self.router = PerimeterRouter()
        self.raw_inflows = {
            "GW_I35_NORTH": 1000.0,
            "GW_CLASSEN_NORTH": 300.0,
            "GW_SOONER_NORTH": 200.0,
            "GW_US77_SOUTH": 400.0,
            "GW_SH9_WEST": 250.0,
            "GW_SH9_EAST": 150.0,
        }

    def test_diversion_mass_conservation(self):
        """Total network inflow must be exactly conserved after perimeter diversion."""
        diverted = self.router.divert_gateway_inflows(
            t_sim_s=7200.0,
            raw_inflows=self.raw_inflows,
            is_surge_phase=True,
            lindsey_density_ratio=0.60,
        )

        total_raw = sum(self.raw_inflows.values())
        total_diverted = sum(diverted.values())

        self.assertAlmostEqual(total_raw, total_diverted, places=4)
        # Verify I-35 decreased and Sooner bypass increased
        self.assertLess(diverted["GW_I35_NORTH"], self.raw_inflows["GW_I35_NORTH"])
        self.assertGreater(diverted["GW_SOONER_NORTH"], self.raw_inflows["GW_SOONER_NORTH"])

    def test_pulsed_pedestrian_scramble_cycle(self):
        """Vehicle phase (first 90s of 135s cycle) must give full factor 1.0; scramble phase must give 0.05."""
        # Active surge:
        mult = 0.90
        # t = 30s is within [0, 90) -> Vehicle phase
        factor_veh = self.router.get_pedestrian_scramble_factor(30.0, mult)
        self.assertEqual(factor_veh, 1.0)

        # t = 100s is within [90, 135) -> Pedestrian scramble phase
        factor_ped = self.router.get_pedestrian_scramble_factor(100.0, mult)
        self.assertEqual(factor_ped, 0.05)

    def test_pedestrian_factor_inactive_outside_surge(self):
        """When pedestrian surge is minimal (< 0.15), factor must always be 1.0."""
        factor = self.router.get_pedestrian_scramble_factor(100.0, 0.05)
        self.assertEqual(factor, 1.0)


if __name__ == "__main__":
    unittest.main()
