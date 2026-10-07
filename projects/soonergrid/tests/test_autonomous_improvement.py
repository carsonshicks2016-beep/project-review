"""
Counterfactual Benchmark Unit Test for Step 4 Autonomous Traffic Optimization.
Validates quantitative improvements of the autonomous policy over the Step 3 baseline.
"""

import unittest
import os
import json


class TestAutonomousImprovement(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.baseline_file = "data/norman_baseline_simulation_results.json"
        cls.autonomous_file = "data/norman_autonomous_simulation_results.json"

        if not os.path.exists(cls.baseline_file) or not os.path.exists(cls.autonomous_file):
            raise unittest.SkipTest("Simulation results files not found; run pipelines first.")

        with open(cls.baseline_file, "r", encoding="utf-8") as f:
            cls.baseline = json.load(f)["summary"]

        with open(cls.autonomous_file, "r", encoding="utf-8") as f:
            cls.autonomous = json.load(f)["summary"]

    def test_tstt_reduction(self):
        """Autonomous control must reduce Total System Travel Time by at least 25%."""
        base_tstt = self.baseline["total_system_travel_time_veh_hrs"]
        auto_tstt = self.autonomous["total_system_travel_time_veh_hrs"]
        reduction_pct = (base_tstt - auto_tstt) / base_tstt * 100.0

        self.assertGreater(reduction_pct, 25.0, f"TSTT reduction too low: {reduction_pct:.1f}%")

    def test_queue_delay_reduction(self):
        """Autonomous control must reduce Total Gridlock Queue Delay by at least 35%."""
        base_delay = self.baseline["total_queue_delay_veh_hrs"]
        auto_delay = self.autonomous["total_queue_delay_veh_hrs"]
        reduction_pct = (base_delay - auto_delay) / base_delay * 100.0

        self.assertGreater(reduction_pct, 35.0, f"Queue delay reduction too low: {reduction_pct:.1f}%")

    def test_fuel_conservation(self):
        """Autonomous control must waste substantially less fuel than baseline."""
        base_fuel = self.baseline["excess_fuel_wasted_gallons"]
        auto_fuel = self.autonomous["excess_fuel_wasted_gallons"]

        self.assertLess(auto_fuel, base_fuel)
        self.assertGreater(base_fuel - auto_fuel, 20000.0, "Expected at least 20,000 gallons saved.")


if __name__ == "__main__":
    unittest.main()
