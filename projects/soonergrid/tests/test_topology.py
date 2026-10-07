"""
Unit tests for topological connectivity and strongly connected components.
"""

import unittest
import networkx as nx
from soonergrid.physics.link_attributes import EdgeAttributes


class TestTopology(unittest.TestCase):
    def setUp(self):
        # Construct a synthetic mini-network representing I-35, Lindsey St, and Stadium
        self.G = nx.MultiDiGraph()
        
        # Nodes: 1 (I-35 North), 2 (I-35 & Lindsey Interchange), 3 (Stadium), 4 (Lloyd Noble)
        self.G.add_node(1, x=0.0, y=5000.0, lat=35.25, lon=-97.48)
        self.G.add_node(2, x=0.0, y=0.0, lat=35.20, lon=-97.48)
        self.G.add_node(3, x=3000.0, y=0.0, lat=35.20, lon=-97.44)
        self.G.add_node(4, x=3000.0, y=-2000.0, lat=35.18, lon=-97.44)

        # Edges
        # I-35 North -> Interchange (oneway)
        self.G.add_edge(1, 2, key="e1", attr=EdgeAttributes(
            edge_id="e1", u=1, v=2, name="I-35 S", highway_type="motorway",
            length_m=5000.0, lanes=3, free_speed_mps=29.0, free_speed_mph=65.0,
            free_flow_time_s=172.4, capacity_vph=6000.0, jam_storage_veh=667,
            is_oneway=True, is_reversible=False
        ))
        # Interchange -> I-35 North (oneway)
        self.G.add_edge(2, 1, key="e2", attr=EdgeAttributes(
            edge_id="e2", u=2, v=1, name="I-35 N", highway_type="motorway",
            length_m=5000.0, lanes=3, free_speed_mps=29.0, free_speed_mph=65.0,
            free_flow_time_s=172.4, capacity_vph=6000.0, jam_storage_veh=667,
            is_oneway=True, is_reversible=False
        ))
        # Lindsey St (bidirectional between Interchange and Stadium)
        self.G.add_edge(2, 3, key="e3", attr=EdgeAttributes(
            edge_id="e3", u=2, v=3, name="W Lindsey St", highway_type="primary",
            length_m=3000.0, lanes=2, free_speed_mps=15.6, free_speed_mph=35.0,
            free_flow_time_s=192.3, capacity_vph=2200.0, jam_storage_veh=400,
            is_oneway=False, is_reversible=True
        ))
        self.G.add_edge(3, 2, key="e4", attr=EdgeAttributes(
            edge_id="e4", u=3, v=2, name="W Lindsey St", highway_type="primary",
            length_m=3000.0, lanes=2, free_speed_mps=15.6, free_speed_mph=35.0,
            free_flow_time_s=192.3, capacity_vph=2200.0, jam_storage_veh=400,
            is_oneway=False, is_reversible=True
        ))
        # Jenkins Ave (bidirectional between Stadium and Lloyd Noble)
        self.G.add_edge(3, 4, key="e5", attr=EdgeAttributes(
            edge_id="e5", u=3, v=4, name="S Jenkins Ave", highway_type="tertiary",
            length_m=2000.0, lanes=1, free_speed_mps=11.2, free_speed_mph=25.0,
            free_flow_time_s=178.6, capacity_vph=700.0, jam_storage_veh=267,
            is_oneway=False, is_reversible=True
        ))
        self.G.add_edge(4, 3, key="e6", attr=EdgeAttributes(
            edge_id="e6", u=4, v=3, name="S Jenkins Ave", highway_type="tertiary",
            length_m=2000.0, lanes=1, free_speed_mps=11.2, free_speed_mph=25.0,
            free_flow_time_s=178.6, capacity_vph=700.0, jam_storage_veh=267,
            is_oneway=False, is_reversible=True
        ))

    def test_strongly_connected(self):
        """Graph must be strongly connected so all points are reachable and escapable."""
        self.assertTrue(nx.is_strongly_connected(self.G))

    def test_dijkstra_path_existence(self):
        """Gateway 1 must be able to reach Sink 3 (Stadium) and Sink 4 (Lloyd Noble)."""
        path_to_stadium = nx.shortest_path(self.G, source=1, target=3)
        self.assertEqual(path_to_stadium, [1, 2, 3])

        path_to_lnc = nx.shortest_path(self.G, source=1, target=4)
        self.assertEqual(path_to_lnc, [1, 2, 3, 4])

    def test_no_isolated_nodes(self):
        """No isolated nodes should exist in graph."""
        isolates = list(nx.isolates(self.G))
        self.assertEqual(len(isolates), 0)


if __name__ == "__main__":
    unittest.main()
