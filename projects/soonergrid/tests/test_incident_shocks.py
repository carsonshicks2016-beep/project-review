"""
Unit tests for Step 5 Incident Stress Testing (Collision, Squall, Overtime).
"""

import unittest
from soonergrid.sim.incident_manager import IncidentStressTester


class TestIncidentShocks(unittest.TestCase):

    def setUp(self):
        self.tester = IncidentStressTester(kickoff_time_s=14400.0, game_duration_s=11520.0)
        self.mock_edge_meta = {
            "edge_lindsey_berry": {"name": "W Lindsey St & Berry Rd", "lanes": 3},
            "edge_sh9": {"name": "State Highway 9", "lanes": 2},
        }

    def test_collision_shock_derating(self):
        """Active collision must derate targeted corridor capacity by 67% (0.33 factor)."""
        self.tester.activate_shock("SHOCK_COLLISION")

        # t = 9500s is inside crash window [9000s, 11100s)
        deratings, speed_factor, evac_vps = self.tester.evaluate_step(9500.0, self.mock_edge_meta)

        self.assertIn("edge_lindsey_berry", deratings)
        self.assertEqual(deratings["edge_lindsey_berry"], 0.33)
        self.assertNotIn("edge_sh9", deratings)

    def test_weather_squall_global_impact(self):
        """Active weather squall must derate all edges and trigger unscheduled evacuation."""
        self.tester.activate_shock("SHOCK_THUNDERSTORM")

        # t = 20000s is inside squall window [19800s, 22500s)
        deratings, speed_factor, evac_vps = self.tester.evaluate_step(20000.0, self.mock_edge_meta)

        self.assertEqual(deratings["edge_lindsey_berry"], 0.60)
        self.assertEqual(deratings["edge_sh9"], 0.60)
        self.assertEqual(speed_factor, 0.65)
        self.assertGreater(evac_vps, 0.0)

    def test_overtime_extension_shift(self):
        """Activating overtime must shift egress timeline by +45 minutes (2700s)."""
        self.assertEqual(self.tester.overtime_shift_s, 0.0)
        self.tester.activate_shock("SHOCK_OVERTIME")
        self.assertEqual(self.tester.overtime_shift_s, 2700.0)


if __name__ == "__main__":
    unittest.main()
