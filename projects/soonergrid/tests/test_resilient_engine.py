import unittest
import networkx as nx
from soonergrid.physics.link_attributes import EdgeAttributes
from soonergrid.data.game_day_zones import Gateway, ParkingSink
from soonergrid.sim.resilient_traffic_engine import ResilientTrafficSimulationEngine

class TestResilientTrafficEngine(unittest.TestCase):
    def setUp(self):
        self.graph = nx.MultiDiGraph()
        self.graph.add_node(1, x=0.0, y=0.0)
        self.graph.add_node(2, x=200.0, y=0.0)
        attr1 = EdgeAttributes(
            edge_id="e1", u=1, v=2, name="W Lindsey St", highway_type="primary",
            length_m=200.0, lanes=2, free_speed_mps=15.0, free_speed_mph=33.5,
            free_flow_time_s=13.3, capacity_vph=3600.0, jam_storage_veh=53,
            is_oneway=True, is_reversible=False
        )
        self.graph.add_edge(1, 2, key=0, attr=attr1)

        self.gateways = [
            Gateway(
                id="GW_TEST", name="Test Gateway", lat=35.295, lon=-97.485,
                corridor="Test", primary_origin="North", hourly_inflow_weight=1.0,
                nearest_node=1, dist_to_node_m=0.0
            )
        ]
        self.sinks = [
            ParkingSink(
                id="SINK_TEST", name="Test Sink", lat=35.2059, lon=-97.4423,
                capacity_stalls=5000, zone_type="stadium_core", shuttle_served=False,
                description="Test", nearest_node=2, dist_to_node_m=0.0
            )
        ]
        self.demand = {
            "timeline": [
                {"time_hr": 0.0, "phase": "Pre-Game", "gateway_inflows": {"GW_TEST": 500.0}}
            ]
        }

    def test_engine_initialization_and_run(self):
        engine = ResilientTrafficSimulationEngine(
            graph=self.graph,
            gateways=self.gateways,
            sinks=self.sinks,
            demand_dataset=self.demand,
            dt_s=5.0,
            cav_penetration=0.50,
            driver_compliance=0.80,
            incident_shock="SHOCK_COLLISION",
            enable_self_healing=True,
        )
        # Verify CAV capacity scaling: M_C(0.5) > 1.0
        self.assertGreater(engine.cav_capacity_multiplier, 1.2)
        # Verify compliance composite: 0.5 * 0.8 + 0.5 * 1.0 = 0.90
        self.assertAlmostEqual(engine.fleet_compliance, 0.90, places=2)

        # Run short simulation (15 minutes = 900s)
        results = engine.run_simulation(total_duration_s=900.0, playback_sample_interval_s=60.0)
        self.assertIn("summary", results)
        self.assertIn("playback_frames", results)
        self.assertGreater(len(results["playback_frames"]), 0)

if __name__ == '__main__':
    unittest.main()
