"""
Unit tests for Step 4 Dynamic Contraflow Policy.
Validates lane conservation, safety clearance buffer, and directional capacity scaling.
"""

import unittest
from soonergrid.policy.contraflow_manager import DynamicContraflowManager


class TestContraflowPolicy(unittest.TestCase):

    def setUp(self):
        self.manager = DynamicContraflowManager(
            kickoff_time_s=14400.0,
            game_duration_s=11520.0,
            clearance_duration_s=600.0,
        )
        # Mock Lindsey St eastbound (inbound) and westbound (outbound) edges
        self.edge_meta = {
            "edge_lindsey_east": {
                "name": "W Lindsey St",
                "u": 100,
                "v": 200,
                "lanes": 2,
                "capacity_vph": 2200.0,
                "capacity_vps": 2200.0 / 3600.0,
                "is_reversible": True,
            },
            "edge_lindsey_west": {
                "name": "W Lindsey St",
                "u": 200,
                "v": 100,
                "lanes": 2,
                "capacity_vph": 2200.0,
                "capacity_vps": 2200.0 / 3600.0,
                "is_reversible": True,
            },
        }
        self.manager.register_contraflow_twins(self.edge_meta)

    def test_lane_conservation_across_modes(self):
        """Total physical lanes across bidirectional corridor must always sum to 4 lanes."""
        modes = ["MODE_BALANCED", "MODE_INGRESS_TIDAL", "MODE_EGRESS_FLUSH"]
        for mode in modes:
            self.manager.apply_contraflow_modifications(self.edge_meta, mode)
            total_lanes = (
                self.edge_meta["edge_lindsey_east"]["lanes"] +
                self.edge_meta["edge_lindsey_west"]["lanes"]
            )
            self.assertEqual(total_lanes, 4, f"Lanes not conserved in {mode}: got {total_lanes}")

    def test_ingress_capacity_expansion(self):
        """In Ingress Tidal mode, eastbound capacity must increase from 2 lanes to 3 lanes (+50%)."""
        self.manager.apply_contraflow_modifications(self.edge_meta, "MODE_INGRESS_TIDAL")
        self.assertEqual(self.edge_meta["edge_lindsey_east"]["lanes"], 3)
        self.assertEqual(self.edge_meta["edge_lindsey_west"]["lanes"], 1)
        self.assertEqual(self.edge_meta["edge_lindsey_east"]["capacity_vph"], 3300.0)

    def test_egress_flush_capacity_expansion(self):
        """In Egress Flush mode, westbound capacity must expand to 4 lanes (0:4 flush)."""
        self.manager.apply_contraflow_modifications(self.edge_meta, "MODE_EGRESS_FLUSH")
        self.assertEqual(self.edge_meta["edge_lindsey_east"]["lanes"], 0)
        self.assertEqual(self.edge_meta["edge_lindsey_west"]["lanes"], 4)
        self.assertEqual(self.edge_meta["edge_lindsey_east"]["capacity_vph"], 0.0)
        self.assertEqual(self.edge_meta["edge_lindsey_west"]["capacity_vph"], 4400.0)

    def test_safety_clearance_buffer(self):
        """10 minutes prior to game end, mode must transition to MODE_CLEARANCE_BUFFER."""
        # Kickoff = 14400s, Duration = 11520s, End = 25920s.
        # Clearance window: [25320s, 25920s)
        t_before = 25300.0
        t_clearance = 25500.0
        t_after = 26000.0

        mode_before = self.manager.evaluate_mode(t_before)
        mode_clearance = self.manager.evaluate_mode(t_clearance)
        mode_after = self.manager.evaluate_mode(t_after)

        self.assertNotEqual(mode_clearance, "MODE_EGRESS_FLUSH")
        self.assertEqual(mode_clearance, "MODE_CLEARANCE_BUFFER")
        self.assertEqual(mode_after, "MODE_EGRESS_FLUSH")


if __name__ == "__main__":
    unittest.main()
