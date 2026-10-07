"""Regression checks for conservation, boundary accounting and immutable demand."""
import copy
import unittest
from soonergrid.sim.conservation import allocate_movements, admit_boundary, validate_run
from soonergrid.sim.resilient_traffic_engine import ResilientTrafficSimulationEngine
import test_resilient_engine

class ResearchIntegrityTests(unittest.TestCase):
    def test_shared_merge_budget(self):
        flows=allocate_movements([('a','c',8),('b','c',8)],{'c':10})
        self.assertEqual(sum(v for _,_,v in flows),10)
        self.assertEqual([v for _,_,v in flows],[5,5])

    def test_zero_receiving_stops_all_feeders(self):
        self.assertEqual(sum(v for _,_,v in allocate_movements([('a','c',8),('b','c',2)],{'c':0})),0)

    def test_boundary_queue_is_not_lost(self):
        vehicles={'a':9};meta={'a':{'jam_storage_veh':10,'capacity_vps':2}};queue={}
        self.assertEqual(admit_boundary(vehicles,meta,queue,{'a':100},5),1)
        self.assertEqual(vehicles['a'],10);self.assertEqual(queue['a'],99)
        vehicles['a']=0
        self.assertEqual(admit_boundary(vehicles,meta,queue,{},5),10)
        self.assertEqual(queue['a'],89)

    def test_invalid_time_and_demand_rejected(self):
        for dt,duration,interval,timeline in [(0,10,1,[{}]),(3,10,1,[{}]),(1,10,1,[]),(1,10,1,[{'gateway_inflows':{'a':-1}}])]:
            with self.assertRaises(ValueError):validate_run(dt,duration,interval,timeline)

    def engine(self):
        fixture=test_resilient_engine.TestResilientTrafficEngine();fixture.setUp()
        return fixture,ResilientTrafficSimulationEngine(fixture.graph,fixture.gateways,fixture.sinks,fixture.demand,enable_self_healing=False)

    def test_sink_deadend_has_single_exit_and_exact_ledger(self):
        fixture,engine=self.engine();before=copy.deepcopy(fixture.demand)
        report=engine.run_simulation(900)['summary'];c=report['conservation']
        self.assertAlmostEqual(c['generated_veh'],500,places=7)
        self.assertAlmostEqual(c['generated_veh'],c['exited_veh']+c['remaining_veh']+c['boundary_queue_veh'],places=7)
        self.assertLess(c['max_absolute_residual_veh'],1e-7)
        self.assertEqual(fixture.demand,before)
        self.assertLessEqual(max(engine.vehicles_on_link.values()),53)
        with self.assertRaises(RuntimeError):engine.run_simulation(900)

    def test_reproducibility(self):
        _,first=self.engine();_,second=self.engine()
        self.assertEqual(first.run_simulation(120),second.run_simulation(120))

    def test_competing_feeders_preserve_storage_in_both_engines(self):
        from soonergrid.sim.traffic_engine import TrafficSimulationEngine
        from soonergrid.physics.link_attributes import EdgeAttributes
        import networkx as nx
        for cls in [TrafficSimulationEngine, ResilientTrafficSimulationEngine]:
            graph=nx.MultiDiGraph()
            for u,v,eid in [(1,3,'a'),(2,3,'b'),(3,4,'c')]:
                graph.add_edge(u,v,key=eid,attr=EdgeAttributes(
                    edge_id=eid,u=u,v=v,name='test',highway_type='primary',
                    length_m=100,lanes=1,free_speed_mps=20,free_speed_mph=44.7,
                    free_flow_time_s=5,capacity_vph=3600,jam_storage_veh=10,
                    is_oneway=True,is_reversible=False))
            engine=cls(graph,[],[],{'timeline':[{'gateway_inflows':{}}]},dt_s=5)
            engine.vehicles_on_link.update(a=9,b=9,c=9)
            result=engine.run_simulation(5)['summary']['conservation']
            self.assertLessEqual(engine.vehicles_on_link['c'],10)
            self.assertAlmostEqual(result['initial_veh'],27)
            self.assertAlmostEqual(result['initial_veh'],result['exited_veh']+result['remaining_veh'])

    def test_fractional_occupancy_is_not_truncated(self):
        from soonergrid.sim.metrics_tracker import MetricsTracker
        from soonergrid.sim.link_transmission import LinkFlowState
        states={str(i):LinkFlowState(str(i),.001,.6,10,22.3694,1,1,0,False) for i in range(10)}
        metadata={str(i):{'free_flow_time_s':10,'length_m':100,'name':'test','highway_type':'primary'} for i in range(10)}
        result=MetricsTracker().record_step(0,0,'test',states,metadata)
        self.assertAlmostEqual(result.active_vehicles_on_network,6)
