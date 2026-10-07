"""
Unit tests for dynamic contraflow (reversible lane) inversion logic.
"""

import unittest
import networkx as nx
from soonergrid.physics.link_attributes import EdgeAttributes
from soonergrid.data.network_builder import NetworkBuilder


class TestContraflow(unittest.TestCase):
    def setUp(self):
        self.builder = NetworkBuilder()
        self.G = nx.MultiDiGraph()

        # Inbound Lindsey (Interchange -> Campus)
        self.edge_in = EdgeAttributes(
            edge_id="e_lindsey_east",
            u=100, v=200,
            name="W Lindsey St",
            highway_type="primary",
            length_m=2000.0,
            lanes=2,
            free_speed_mps=15.65,
            free_speed_mph=35.0,
            free_flow_time_s=127.8,
            capacity_vph=2200.0,
            jam_storage_veh=533,
            is_oneway=False,
            is_reversible=True
        )

        # Outbound Lindsey (Campus -> Interchange)
        self.edge_out = EdgeAttributes(
            edge_id="e_lindsey_west",
            u=200, v=100,
            name="W Lindsey St",
            highway_type="primary",
            length_m=2000.0,
            lanes=2,
            free_speed_mps=15.65,
            free_speed_mph=35.0,
            free_flow_time_s=127.8,
            capacity_vph=2200.0,
            jam_storage_veh=533,
            is_oneway=False,
            is_reversible=True
        )

        self.G.add_node(100, x=0.0, y=0.0, lat=35.20, lon=-97.48)
        self.G.add_node(200, x=2000.0, y=0.0, lat=35.20, lon=-97.45)
        self.G.add_edge(100, 200, key="e_lindsey_east", attr=self.edge_in)
        self.G.add_edge(200, 100, key="e_lindsey_west", attr=self.edge_out)

    def test_twin_registration(self):
        """Builder should register each edge as the other's twin."""
        self.builder._register_twin_edges(self.G)
        self.assertEqual(self.edge_in.twin_edge_id, "e_lindsey_west")
        self.assertEqual(self.edge_out.twin_edge_id, "e_lindsey_east")

    def test_contraflow_lane_reallocation(self):
        """Simulate post-game egress flush: out-lanes become 4, in-lanes become 0."""
        self.builder._register_twin_edges(self.G)
        
        # Original state: 2 lanes in each direction
        self.assertEqual(self.edge_in.lanes, 2)
        self.assertEqual(self.edge_out.lanes, 2)
        self.assertEqual(self.edge_in.capacity_vph, 2200.0)
        self.assertEqual(self.edge_out.capacity_vph, 2200.0)

        # Trigger 4:0 post-game contraflow towards I-35 (Westbound / out)
        total_lanes = self.edge_in.lanes + self.edge_out.lanes
        self.edge_out.lanes = total_lanes
        self.edge_out.capacity_vph = total_lanes * 1100.0
        self.edge_in.lanes = 0
        self.edge_in.capacity_vph = 0.0

        self.assertEqual(self.edge_out.lanes, 4)
        self.assertEqual(self.edge_out.capacity_vph, 4400.0)
        self.assertEqual(self.edge_in.lanes, 0)
        self.assertEqual(self.edge_in.capacity_vph, 0.0)


if __name__ == "__main__":
    unittest.main()
