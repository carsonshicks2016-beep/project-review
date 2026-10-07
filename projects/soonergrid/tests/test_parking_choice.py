"""
Unit tests for Multinomial Logit parking selection and capacity spillover.
"""

import unittest
from soonergrid.data.game_day_zones import ParkingSink
from soonergrid.demand.parking_choice import ParkingAllocationManager


class TestParkingChoice(unittest.TestCase):
    def setUp(self):
        # 3 Test Sinks: Lloyd Noble (Large, Free, Shuttle), Elm Garage (Medium, $20), Heisman (Small, $35)
        self.sinks = [
            ParkingSink("S1_LNC", "Lloyd Noble Center", 35.18, -97.44, 1000, "satellite_hub", True, "Free shuttle"),
            ParkingSink("S2_ELM", "Elm Garage", 35.20, -97.44, 500, "garage", False, "Campus garage"),
            ParkingSink("S3_HEISMAN", "Heisman Lot", 35.20, -97.44, 200, "stadium_core", False, "Donor lot"),
        ]
        self.manager = ParkingAllocationManager(self.sinks)
        self.dists = {"S1_LNC": 12.0, "S2_ELM": 10.0, "S3_HEISMAN": 9.5}

    def test_probabilities_sum_to_one(self):
        """MNL choice probabilities must strictly sum to 1.0."""
        probs = self.manager.get_choice_probabilities("GW_I35", self.dists)
        total_p = sum(probs.values())
        self.assertAlmostEqual(total_p, 1.0, places=5)

    def test_lot_saturation_spillover(self):
        """When a lot fills completely, its choice probability must drop to zero."""
        # Fill S3_HEISMAN completely (200 stalls)
        for _ in range(200):
            self.manager.allocate_vehicle("S3_HEISMAN")

        self.assertEqual(self.manager.get_occupancy_ratio("S3_HEISMAN"), 1.0)
        probs = self.manager.get_choice_probabilities("GW_I35", self.dists)
        self.assertNotIn("S3_HEISMAN", probs)
        self.assertAlmostEqual(sum(probs.values()), 1.0, places=5)

    def test_overflow_graceful_handling(self):
        """Attempting to allocate to a full lot must gracefully spill over to an available lot."""
        # First fill S3 to capacity (200/200)
        for _ in range(200):
            self.manager.allocate_vehicle("S3_HEISMAN")

        self.assertEqual(self.manager.occupancy["S3_HEISMAN"], 200)

        # Attempting to add 201st vehicle should spill over to S1 or S2
        success = self.manager.allocate_vehicle("S3_HEISMAN")
        self.assertTrue(success)
        # Verify that either S1 or S2 took the extra car
        self.assertTrue(self.manager.occupancy["S1_LNC"] > 0 or self.manager.occupancy["S2_ELM"] > 0)


if __name__ == "__main__":
    unittest.main()
