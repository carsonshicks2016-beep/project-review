"""
Unit tests for physical transportation attributes and BPR/LWR performance functions.
"""

import unittest
from soonergrid.physics.link_attributes import (
    EdgeAttributes,
    compute_edge_attributes,
    parse_speed_mph,
    parse_lanes_count,
    check_contraflow_candidate,
)


class TestAttributes(unittest.TestCase):
    def setUp(self):
        self.edge = EdgeAttributes(
            edge_id="test_edge",
            u=10,
            v=20,
            name="W Lindsey St",
            highway_type="primary",
            length_m=1000.0,
            lanes=2,
            free_speed_mps=15.65,
            free_speed_mph=35.0,
            free_flow_time_s=63.9,
            capacity_vph=2200.0,
            jam_storage_veh=267,
            is_oneway=False,
            is_reversible=True
        )

    def test_bpr_monotonicity(self):
        """BPR travel time must strictly increase as volume increases."""
        t_free = self.edge.bpr_travel_time(0.0)
        self.assertAlmostEqual(t_free, self.edge.free_flow_time_s, places=1)

        t_capacity = self.edge.bpr_travel_time(2200.0)
        # At V = C, BPR formula gives t_0 * (1 + 0.15) = 1.15 * t_0
        self.assertAlmostEqual(t_capacity, self.edge.free_flow_time_s * 1.15, places=1)

        t_congested = self.edge.bpr_travel_time(4400.0)
        self.assertGreater(t_congested, t_capacity)

    def test_lwr_density_velocity(self):
        """LWR velocity must decrease linearly from free-flow speed to 0 at jam density."""
        v_free = self.edge.density_to_velocity_lwr(0.0)
        self.assertAlmostEqual(v_free, self.edge.free_speed_mps, places=2)

        v_half = self.edge.density_to_velocity_lwr(self.edge.jam_storage_veh / 2.0)
        self.assertAlmostEqual(v_half, self.edge.free_speed_mps * 0.5, places=2)

        v_jam = self.edge.density_to_velocity_lwr(self.edge.jam_storage_veh)
        self.assertAlmostEqual(v_jam, 0.0, places=2)

    def test_speed_parser(self):
        """Validates mph parsing from strings."""
        self.assertEqual(parse_speed_mph("45 mph", "primary"), 45.0)
        self.assertEqual(parse_speed_mph("30", "secondary"), 30.0)
        # Fallback heuristic
        self.assertEqual(parse_speed_mph(None, "motorway"), 65.0)

    def test_contraflow_candidate_detection(self):
        """Corridors with 'lindsey' or 'jenkins' should be marked reversible."""
        self.assertTrue(check_contraflow_candidate("West Lindsey Street"))
        self.assertTrue(check_contraflow_candidate("South Jenkins Avenue"))
        self.assertFalse(check_contraflow_candidate("Robinson Street"))


if __name__ == "__main__":
    unittest.main()
