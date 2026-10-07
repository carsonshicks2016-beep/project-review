"""
Unit tests for route assignment and driver behavioral splits (Greedy vs. Local).
"""

import unittest
import networkx as nx
from soonergrid.physics.link_attributes import EdgeAttributes
from soonergrid.demand.route_assignment import RouteAssignmentEngine


class TestRouteAssignment(unittest.TestCase):
    def setUp(self):
        self.G = nx.MultiDiGraph()
        # Node 1 (Gateway) -> Node 2 (Lindsey St) -> Node 3 (Stadium)
        # Node 1 (Gateway) -> Node 4 (Highway 9 bypass) -> Node 3 (Stadium)
        self.G.add_node(1, x=0, y=2000, lat=35.22, lon=-97.48)
        self.G.add_node(2, x=2000, y=2000, lat=35.22, lon=-97.45)
        self.G.add_node(3, x=4000, y=2000, lat=35.22, lon=-97.44)
        self.G.add_node(4, x=2000, y=0, lat=35.19, lon=-97.45)

        # Primary route (1 -> 2 -> 3)
        self.G.add_edge(1, 2, key="e1", attr=EdgeAttributes(
            edge_id="e1", u=1, v=2, name="W Lindsey St", highway_type="primary",
            length_m=2000.0, lanes=2, free_speed_mps=15.0, free_speed_mph=35.0,
            free_flow_time_s=133.0, capacity_vph=2200.0, jam_storage_veh=300,
            is_oneway=True, is_reversible=True
        ))
        self.G.add_edge(2, 3, key="e2", attr=EdgeAttributes(
            edge_id="e2", u=2, v=3, name="W Lindsey St", highway_type="primary",
            length_m=2000.0, lanes=2, free_speed_mps=15.0, free_speed_mph=35.0,
            free_flow_time_s=133.0, capacity_vph=2200.0, jam_storage_veh=300,
            is_oneway=True, is_reversible=True
        ))

        # Secondary bypass route (1 -> 4 -> 3)
        self.G.add_edge(1, 4, key="e3", attr=EdgeAttributes(
            edge_id="e3", u=1, v=4, name="Highway 9 Bypass", highway_type="trunk",
            length_m=2800.0, lanes=2, free_speed_mps=24.0, free_speed_mph=55.0,
            free_flow_time_s=116.0, capacity_vph=3200.0, jam_storage_veh=400,
            is_oneway=True, is_reversible=False
        ))
        self.G.add_edge(4, 3, key="e4", attr=EdgeAttributes(
            edge_id="e4", u=4, v=3, name="S Jenkins Ave", highway_type="tertiary",
            length_m=2000.0, lanes=1, free_speed_mps=11.0, free_speed_mph=25.0,
            free_flow_time_s=181.0, capacity_vph=700.0, jam_storage_veh=260,
            is_oneway=True, is_reversible=True
        ))

        self.engine = RouteAssignmentEngine(self.G, greedy_ratio=0.75)

    def test_k_shortest_paths_diversity(self):
        """Must compute distinct paths between origin and destination."""
        paths = self.engine.get_k_shortest_paths(1, 3, k=2)
        self.assertGreaterEqual(len(paths), 2)
        self.assertNotEqual(paths[0]["node_path"], paths[1]["node_path"])

    def test_greedy_vs_local_assignment(self):
        """Greedy GPS user gets shortest free-flow path, local driver gets alternative bypass."""
        greedy_route = self.engine.assign_route(1, 3, is_local_driver=False)
        local_route = self.engine.assign_route(1, 3, is_local_driver=True)

        self.assertIsNotNone(greedy_route)
        self.assertIsNotNone(local_route)
        self.assertEqual(greedy_route["node_path"], [1, 2, 3])
        self.assertEqual(local_route["node_path"], [1, 4, 3])


if __name__ == "__main__":
    unittest.main()
