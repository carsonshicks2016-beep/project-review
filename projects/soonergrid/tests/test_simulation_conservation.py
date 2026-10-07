"""
Unit tests for network-wide vehicle conservation and queue propagation.
"""

import unittest
import networkx as nx
from soonergrid.physics.link_attributes import EdgeAttributes
from soonergrid.sim.link_transmission import LinkTransmissionModel
from soonergrid.sim.traffic_engine import TrafficSimulationEngine


class TestSimulationConservation(unittest.TestCase):
    def setUp(self):
        self.G = nx.MultiDiGraph()
        # 3-link linear pipeline: 1 -> 2 -> 3 -> 4
        self.G.add_node(1, x=0, y=0, lat=35.2, lon=-97.48)
        self.G.add_node(2, x=1000, y=0, lat=35.2, lon=-97.47)
        self.G.add_node(3, x=2000, y=0, lat=35.2, lon=-97.46)
        self.G.add_node(4, x=3000, y=0, lat=35.2, lon=-97.45)

        for u, v, eid in [(1, 2, "e1"), (2, 3, "e2"), (3, 4, "e3")]:
            self.G.add_edge(u, v, key=eid, attr=EdgeAttributes(
                edge_id=eid, u=u, v=v, name=f"Test Link {eid}", highway_type="primary",
                length_m=1000.0, lanes=2, free_speed_mps=20.0, free_speed_mph=45.0,
                free_flow_time_s=50.0, capacity_vph=2000.0, jam_storage_veh=250,
                is_oneway=True, is_reversible=False
            ))

        self.ltm = LinkTransmissionModel(backward_wave_speed_mps=5.0)

    def test_queue_spillback_propagation(self):
        """When downstream link e3 is blocked (zero receiving capacity), e2 must accumulate queue."""
        dt = 5.0
        # Put 100 vehicles on e2, 250 (jammed) on e3
        state_e2 = self.ltm.compute_link_state("e2", 100.0, 1000.0, 2, 20.0, 2000.0, 250)
        state_e3 = self.ltm.compute_link_state("e3", 250.0, 1000.0, 2, 20.0, 2000.0, 250)

        # Transfer flow from e2 to e3
        flow_to_e3 = self.ltm.transfer_flow(
            upstream_sending_vps=state_e2.sending_capacity_vps,
            downstream_receiving_vps=state_e3.receiving_capacity_vps,
            dt_s=dt,
            signal_green_ratio=1.0,
        )

        # Because e3 is 100% jammed, receiving capacity is 0, so flow transferred must be 0!
        self.assertEqual(flow_to_e3, 0.0)
        # Vehicles remain on e2, maintaining queue
        self.assertEqual(state_e2.vehicles_on_link, 100.0)


if __name__ == "__main__":
    unittest.main()
