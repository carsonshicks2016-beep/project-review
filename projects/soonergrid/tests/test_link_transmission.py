"""
Unit tests for hydrodynamic Link Transmission Model (LTM) physics and shockwave equations.
"""

import unittest
from soonergrid.sim.link_transmission import LinkTransmissionModel, LinkFlowState


class TestLinkTransmission(unittest.TestCase):
    def setUp(self):
        self.ltm = LinkTransmissionModel(backward_wave_speed_mps=5.0)

    def test_sending_capacity(self):
        """Sending capacity must scale linearly with density until saturating at capacity."""
        # Edge: length 1000m, 2 lanes, free speed 20 m/s (45 mph), capacity 2200 vph (~0.61 vps), jam storage 267 veh
        state_low = self.ltm.compute_link_state(
            edge_id="e1",
            vehicles_on_link=5.0,  # 5 veh / 1000m = 0.005 vpm -> v_free * rho = 20 * 0.005 = 0.1 vps
            length_m=1000.0,
            lanes=2,
            v_free_mps=20.0,
            capacity_vph=2200.0,
            jam_storage_veh=267,
        )
        self.assertAlmostEqual(state_low.sending_capacity_vps, 0.1, places=2)
        self.assertEqual(state_low.speed_mps, 20.0)

        # High density (100 veh on link) should saturate at link capacity (0.61 vps)
        state_high = self.ltm.compute_link_state(
            edge_id="e1",
            vehicles_on_link=100.0,
            length_m=1000.0,
            lanes=2,
            v_free_mps=20.0,
            capacity_vph=2200.0,
            jam_storage_veh=267,
        )
        expected_cap = 2200.0 / 3600.0
        self.assertAlmostEqual(state_high.sending_capacity_vps, expected_cap, places=2)
        self.assertLess(state_high.speed_mps, 20.0)

    def test_receiving_capacity_zero_at_jam(self):
        """When link reaches jam density, receiving capacity must drop to zero."""
        state_jam = self.ltm.compute_link_state(
            edge_id="e1",
            vehicles_on_link=267.0,  # 100% jam storage
            length_m=1000.0,
            lanes=2,
            v_free_mps=20.0,
            capacity_vph=2200.0,
            jam_storage_veh=267,
        )
        self.assertEqual(state_jam.receiving_capacity_vps, 0.0)
        self.assertTrue(state_jam.is_spillback)

    def test_flow_transfer_min_rule(self):
        """Transfer between upstream and downstream links is governed by min(S_up, R_down)."""
        dt = 5.0
        # Upstream sending is high (1.0 vps), downstream receiving is throttled (0.2 vps)
        q = self.ltm.transfer_flow(
            upstream_sending_vps=1.0,
            downstream_receiving_vps=0.2,
            dt_s=dt,
            signal_green_ratio=1.0,
        )
        self.assertAlmostEqual(q, 0.2 * dt, places=2)

        # When signal is red (green_ratio = 0.0), flow is 0
        q_red = self.ltm.transfer_flow(
            upstream_sending_vps=1.0,
            downstream_receiving_vps=0.2,
            dt_s=dt,
            signal_green_ratio=0.0,
        )
        self.assertEqual(q_red, 0.0)


if __name__ == "__main__":
    unittest.main()
