"""
Master Game Day Origin-Destination (OD) Matrix & Spatio-Temporal Demand Orchestrator.
"""

import os
import json
import math
import random
from typing import Dict, List, Tuple, Any, Optional
import networkx as nx

from soonergrid.data.game_day_zones import Gateway, ParkingSink, NORMAN_GATEWAYS, NORMAN_PARKING_SINKS
from soonergrid.demand.temporal_profile import TemporalDemandProfile
from soonergrid.demand.parking_choice import ParkingAllocationManager
from soonergrid.demand.route_assignment import RouteAssignmentEngine
from soonergrid.demand.transit_shuttle import LloydNobleShuttleFleet
from soonergrid.demand.pedestrian_conflicts import PedestrianConflictManager


class GameDayDemandEngine:
    """
    Orchestrates the complete game-day demand generation pipeline:
    integrates non-homogeneous Poisson arrivals, MNL parking selection,
    driver route choices, shuttle bus passenger abstraction, and pedestrian impedance.
    """

    def __init__(
        self,
        graph: nx.MultiDiGraph,
        gateways: List[Gateway],
        sinks: List[ParkingSink],
        total_vehicles: int = 34500,
        demand_scale: float = 1.0,
        greedy_gps_ratio: float = 0.75,
        random_seed: int = 42,
    ):
        self.graph = graph
        self.gateways = gateways
        self.sinks = sinks
        self.total_vehicles = int(total_vehicles * demand_scale)
        self.demand_scale = demand_scale
        self.greedy_ratio = greedy_gps_ratio
        random.seed(random_seed)

        # Initialize sub-engines
        self.temporal_profile = TemporalDemandProfile(total_attendee_vehicles=self.total_vehicles)
        self.parking_manager = ParkingAllocationManager(sinks)
        self.route_engine = RouteAssignmentEngine(graph, greedy_ratio=greedy_gps_ratio)
        self.shuttle_fleet = LloydNobleShuttleFleet()
        self.pedestrian_manager = PedestrianConflictManager()

        # Precompute gateway-to-sink Euclidean & road distances for the MNL choice model
        self.distances_km = self._precompute_gateway_sink_distances()

    def _precompute_gateway_sink_distances(self) -> Dict[str, Dict[str, float]]:
        """Computes metric road distances between each gateway and sink."""
        dist_matrix: Dict[str, Dict[str, float]] = {}
        for gw in self.gateways:
            dist_matrix[gw.id] = {}
            for sink in self.sinks:
                paths = self.route_engine.get_k_shortest_paths(gw.nearest_node, sink.nearest_node, k=1)
                if paths:
                    dist_matrix[gw.id][sink.id] = paths[0]["distance_km"]
                else:
                    # Fallback Euclidean distance in km
                    dx = (gw.lon - sink.lon) * 111.32 * math.cos(math.radians(35.2))
                    dy = (gw.lat - sink.lat) * 110.95
                    dist_matrix[gw.id][sink.id] = math.hypot(dx, dy)
        return dist_matrix

    def generate_spatio_temporal_demand(self, time_step_min: float = 15.0) -> Dict[str, Any]:
        """
        Generates discretized 15-minute time-slice demand arrays across the 8-hour window.
        """
        total_sim_hours = 8.0
        total_steps = int((total_sim_hours * 60.0) / time_step_min)
        dt_seconds = time_step_min * 60.0

        print(f"[DemandEngine] Generating game-day demand across {total_steps} time slices ({time_step_min} min intervals)...")
        print(f"[DemandEngine] Target game-day vehicle population: {self.total_vehicles:,} vehicles (scaling: {self.demand_scale:.2f}x)")

        timeline_data: List[Dict[str, Any]] = []
        cumulative_ingress_generated = 0
        cumulative_egress_discharged = 0

        # Gateway-specific cumulative trip counts
        gateway_totals: Dict[str, int] = {gw.id: 0 for gw in self.gateways}

        for step in range(total_steps):
            t_sim_s = step * dt_seconds
            t_hour = t_sim_s / 3600.0
            phase = self.temporal_profile.get_phase_name(t_sim_s)

            slice_ingress_trips = 0
            slice_egress_trips = 0
            gateway_inflows: Dict[str, int] = {}
            sink_arrivals: Dict[str, int] = {s.id: 0 for s in self.sinks}

            # 1. Ingress Generation (0h to 4.5h)
            if t_sim_s < (self.temporal_profile.t_kickoff + 1800.0):
                for gw in self.gateways:
                    rate_vps = self.temporal_profile.get_ingress_rate_vps(t_sim_s, gw.hourly_inflow_weight)
                    expected_veh = rate_vps * dt_seconds
                    actual_veh = int(expected_veh + random.uniform(0.0, 0.99))
                    gateway_inflows[gw.id] = actual_veh
                    slice_ingress_trips += actual_veh
                    gateway_totals[gw.id] += actual_veh

                    # Allocate parking sink via Multinomial Logit
                    gw_dists = self.distances_km[gw.id]
                    choice_probs = self.parking_manager.get_choice_probabilities(gw.id, gw_dists)

                    # Distribute vehicles according to choice probabilities
                    sink_ids = list(choice_probs.keys())
                    prob_weights = list(choice_probs.values())

                    for _ in range(actual_veh):
                        chosen_sink = random.choices(sink_ids, weights=prob_weights, k=1)[0]
                        self.parking_manager.allocate_vehicle(chosen_sink)
                        sink_arrivals[chosen_sink] += 1

            # 2. Egress Discharge (7.2h to 8.0h)
            if t_sim_s >= (self.temporal_profile.t_game_end - 600.0):
                for sink in self.sinks:
                    egress_rate_vps = self.temporal_profile.get_egress_rate_vps(t_sim_s, sink.capacity_stalls)
                    expected_egress = egress_rate_vps * dt_seconds
                    actual_egress = min(
                        self.parking_manager.occupancy[sink.id],
                        int(expected_egress + random.uniform(0.0, 0.99))
                    )
                    for _ in range(actual_egress):
                        self.parking_manager.discharge_vehicle(sink.id)
                    slice_egress_trips += actual_egress

            cumulative_ingress_generated += slice_ingress_trips
            cumulative_egress_discharged += slice_egress_trips

            # Snapshot parking lot occupancy percentages
            occupancy_pcts = {
                s.id: round(self.parking_manager.get_occupancy_ratio(s.id) * 100.0, 1)
                for s in self.sinks
            }

            # Update shuttle service
            is_peak = ("Peak" in phase) or ("Kickoff" in phase)
            shuttle_status = self.shuttle_fleet.update_shuttle_service(dt_seconds, is_peak)

            # Record time slice
            timeline_data.append({
                "step": step,
                "time_sec": t_sim_s,
                "time_hr": round(t_hour, 2),
                "phase": phase,
                "ingress_rate_vph": round(slice_ingress_trips * (3600.0 / dt_seconds), 0),
                "egress_rate_vph": round(slice_egress_trips * (3600.0 / dt_seconds), 0),
                "gateway_inflows": gateway_inflows,
                "sink_arrivals": sink_arrivals,
                "occupancy_percentages": occupancy_pcts,
                "pedestrian_capacity_derating": round(self.pedestrian_manager.get_effective_capacity_factor("W Lindsey St", t_sim_s), 2),
                "shuttle_pax_rate_hr": shuttle_status["pax_rate_hr"],
            })

        summary = {
            "total_vehicles_generated": cumulative_ingress_generated,
            "total_vehicles_discharged": cumulative_egress_discharged,
            "gateway_breakdown": {
                gw.id: {
                    "name": gw.name,
                    "total_trips": gateway_totals[gw.id],
                    "pct_share": round((gateway_totals[gw.id] / max(1, cumulative_ingress_generated)) * 100.0, 1)
                }
                for gw in self.gateways
            },
            "parking_sink_stats": {
                s.id: {
                    "name": s.name,
                    "capacity": s.capacity_stalls,
                    "peak_occupied": max(slice_data["occupancy_percentages"][s.id] * s.capacity_stalls / 100.0 for slice_data in timeline_data),
                    "peak_occupancy_pct": max(slice_data["occupancy_percentages"][s.id] for slice_data in timeline_data),
                }
                for s in self.sinks
            },
            "shuttle_impact": {
                "total_passengers_carried": self.shuttle_fleet.total_passengers_carried,
                "private_car_trips_avoided": self.shuttle_fleet.equivalent_car_trips_saved,
            }
        }

        dataset = {
            "summary": summary,
            "timeline": timeline_data,
        }
        return dataset

    def export_demand_dataset(self, output_filepath: str, time_step_min: float = 15.0) -> Dict[str, Any]:
        """Runs demand generation and saves JSON dataset to disk."""
        os.makedirs(os.path.dirname(output_filepath), exist_ok=True)
        data = self.generate_spatio_temporal_demand(time_step_min=time_step_min)
        with open(output_filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        print(f"[DemandEngine] Game-day demand dataset exported to: {output_filepath}")
        return data
