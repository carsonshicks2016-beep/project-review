"""
Resilient Vectorized Traffic Simulation Engine for Step 5:
Microscopic Human Factors, Mixed-Autonomy CAV Platooning, Dynamic Incident Injections,
and Adaptive Self-Healing Closed-Loop Response.
"""

import math
from soonergrid.sim.conservation import validate_run, allocate_movements, admit_boundary
from typing import Dict, List, Tuple, Any, Optional, Set
import networkx as nx

from soonergrid.data.game_day_zones import Gateway, ParkingSink
from soonergrid.physics.link_attributes import EdgeAttributes
from soonergrid.demand.pedestrian_conflicts import PedestrianConflictManager
from soonergrid.sim.link_transmission import LinkTransmissionModel, LinkFlowState
from soonergrid.sim.signal_controller import BaselineSignalController, BASELINE_INTERSECTIONS
from soonergrid.sim.metrics_tracker import MetricsTracker, SimulationStepMetrics
from soonergrid.policy.autonomous_coordinator import AutonomousCoordinator
from soonergrid.physics.cav_platooning import CAVPlatooningModel
from soonergrid.policy.human_factors import HumanFactorsManager
from soonergrid.sim.incident_manager import IncidentStressTester
from soonergrid.policy.self_healing_agent import SelfHealingCoordinator


class ResilientTrafficSimulationEngine:
    """
    Simulation engine capable of evaluating resilience under mixed autonomy,
    human behavioral imperfections, and severe crisis disruptions.
    """

    def __init__(
        self,
        graph: nx.MultiDiGraph,
        gateways: List[Gateway],
        sinks: List[ParkingSink],
        demand_dataset: Dict[str, Any],
        dt_s: float = 5.0,
        backward_wave_speed_mps: float = 5.0,
        coordinator: Optional[AutonomousCoordinator] = None,
        cav_penetration: float = 0.0,
        driver_compliance: float = 0.60,
        incident_shock: Optional[str] = None,
        enable_self_healing: bool = True,
    ):
        self.graph = graph
        self.gateways = gateways
        self.sinks = sinks
        self.demand_dataset = demand_dataset
        self.dt_s = dt_s
        self.coordinator = coordinator
        self.cav_penetration = cav_penetration
        self.driver_compliance = driver_compliance
        self.incident_shock = incident_shock
        self.enable_self_healing = enable_self_healing

        self.ltm = LinkTransmissionModel(backward_wave_speed_mps=backward_wave_speed_mps)
        self.signal_controller = BaselineSignalController()
        self.pedestrian_manager = PedestrianConflictManager()
        self.metrics_tracker = MetricsTracker(dt_s=dt_s)

        # Step 5 Specialized Modules
        self.cav_model = CAVPlatooningModel(cav_penetration_rate=cav_penetration)
        self.human_factors = HumanFactorsManager(base_compliance_rate=driver_compliance)
        self.incident_manager = IncidentStressTester()
        if incident_shock and incident_shock != "NOMINAL":
            self.incident_manager.activate_shock(incident_shock)

        self.self_healing = SelfHealingCoordinator() if enable_self_healing else None

        # Effective composite compliance
        self.fleet_compliance = self.cav_model.compute_fleet_compliance(
            human_compliance=driver_compliance,
            p=cav_penetration
        )
        # Capacity multiplier from CAV platooning
        self.cav_capacity_multiplier = self.cav_model.compute_capacity_multiplier(cav_penetration)
        self.cav_storage_multiplier = self.cav_model.compute_jam_storage_multiplier(cav_penetration)

        # Pre-indexed network structures
        self.edge_meta: Dict[str, Dict[str, Any]] = {}
        self.vehicles_on_link: Dict[str, float] = {}
        self.downstream_map: Dict[str, List[str]] = {}

        self.signal_fast_map: Dict[str, Tuple[bool, float, float, float, float, float, str]] = {}
        self.pedestrian_edge_set: Set[str] = set()
        self.lindsey_edge_ids: List[str] = []
        self.i35_ramp_edge_ids: List[str] = []
        self.tracked_corridor_ids: List[str] = []
        self.edge_dist_to_stadium: Dict[str, float] = {}

        self._initialize_network_structures()

    def _initialize_network_structures(self):
        """Precomputes lookup tables, CAV multipliers, signal parameters, and adjacency maps."""
        for u, v, key, data in self.graph.edges(keys=True, data=True):
            attr: EdgeAttributes = data["attr"]
            edge_id = attr.edge_id
            name = attr.name
            lower_name = name.lower()
            hwy = attr.highway_type

            # Node coordinates for distance to stadium (0, 0 is stadium core)
            u_data = self.graph.nodes[u]
            x_u, y_u = u_data.get("x", 0.0), u_data.get("y", 0.0)
            dist_stadium = math.sqrt(x_u * x_u + y_u * y_u)
            self.edge_dist_to_stadium[edge_id] = dist_stadium

            # Scaled base capacity and jam storage from CAV platooning
            base_cap_vph = attr.capacity_vph * self.cav_capacity_multiplier
            base_jam_veh = attr.jam_storage_veh * self.cav_storage_multiplier

            is_l = ("lindsey" in lower_name)
            is_r = ("motorway_link" in hwy and any(k in lower_name for k in ["i 35", "i-35", "interstate 35", "lindsey"]))

            self.edge_meta[edge_id] = {
                "edge_id": edge_id,
                "u": u,
                "v": v,
                "name": name,
                "highway_type": hwy,
                "length_m": attr.length_m,
                "lanes": attr.lanes,
                "free_speed_mps": attr.free_speed_mps,
                "free_speed_mph": attr.free_speed_mph,
                "free_flow_time_s": attr.free_flow_time_s,
                "capacity_vph": base_cap_vph,
                "capacity_vps": max(0.05, base_cap_vph / 3600.0),
                "jam_storage_veh": base_jam_veh,
                "is_reversible": attr.is_reversible,
                "is_lindsey": is_l,
                "is_ramp": is_r,
            }
            self.vehicles_on_link[edge_id] = 0.0

            if is_l:
                self.lindsey_edge_ids.append(edge_id)
            if is_r:
                self.i35_ramp_edge_ids.append(edge_id)
            if "lindsey" in lower_name or "boyd" in lower_name:
                self.pedestrian_edge_set.add(edge_id)

            # Pre-tag signals for fast lookup
            for sig in BASELINE_INTERSECTIONS:
                is_art = any(k in lower_name for k in sig.arterial_keywords)
                is_cr = any(k in lower_name for k in sig.cross_keywords)
                if is_art or is_cr:
                    self.signal_fast_map[edge_id] = (
                        is_art,
                        sig.offset_s,
                        sig.cycle_length_s,
                        sig.arterial_green_s,
                        sig.clearance_s,
                        sig.cross_green_s,
                        sig.id
                    )
                    break

            major_corridor_names = [
                "lindsey", "i 35", "i-35", "interstate 35", "jenkins", "berry",
                "mcgee", "chautauqua", "classen", "highway 9", "sh 9", "boyd",
                "main", "robinson", "flood", "alameda", "brooks", "elm", "asp",
                "24th", "12th", "ed noble"
            ]
            if (any(k in lower_name for k in major_corridor_names)
                    or hwy in ["motorway", "motorway_link", "trunk", "trunk_link", "primary", "primary_link"]
                    or edge_id in self.signal_fast_map):
                self.tracked_corridor_ids.append(edge_id)

        # Build downstream connectivity map
        for u, v, key, data in self.graph.edges(keys=True, data=True):
            edge_id = data["attr"].edge_id
            out_edges = [out_data["attr"].edge_id for _, _, _, out_data in self.graph.out_edges(v, keys=True, data=True)]
            self.downstream_map[edge_id] = out_edges

        # Preindex incident manager with network edges
        self.incident_manager.preindex_network(self.edge_meta)

        # Preindex sink and gateway arrival edges for vehicle completion
        sink_nodes = {s.nearest_node for s in self.sinks if s.nearest_node is not None}
        gw_nodes = {g.nearest_node for g in self.gateways if g.nearest_node is not None}
        self.sink_incoming_eids = {eid for eid, m in self.edge_meta.items() if m["v"] in sink_nodes}
        self.gw_incoming_eids = {eid for eid, m in self.edge_meta.items() if m["v"] in gw_nodes}

        if self.coordinator is not None:
            self.coordinator.initialize_network(self.edge_meta)

    def run_simulation(
        self,
        total_duration_s: float = 28800.0,
        playback_sample_interval_s: float = 60.0
    ) -> Dict[str, Any]:
        """
        Executes the resilient dynamic traffic simulation under mixed autonomy and incidents.
        """
        validate_run(self.dt_s, total_duration_s, playback_sample_interval_s,
                     self.demand_dataset.get("timeline", []))
        if getattr(self, "_has_run", False):
            raise RuntimeError("Create a fresh engine for each independent run")
        self._has_run = True
        boundary_queue = {}
        generated = admitted = exited = boundary_tstt = max_residual = 0.0
        initial = sum(self.vehicles_on_link.values())
        total_steps = round(total_duration_s / self.dt_s)
        sample_steps = max(1, int(playback_sample_interval_s / self.dt_s))

        timeline_slices = self.demand_dataset.get("timeline", [])
        slice_dt_s = 900.0

        gateway_links: Dict[str, List[str]] = {}
        for gw in self.gateways:
            if gw.nearest_node is not None:
                gw_out = [d["attr"].edge_id for _, _, _, d in self.graph.out_edges(gw.nearest_node, keys=True, data=True)]
                gateway_links[gw.id] = gw_out if gw_out else []

        playback_frames: List[Dict[str, Any]] = []

        # Local variables for high-speed execution
        dt = self.dt_s
        edge_meta = self.edge_meta
        vehicles = self.vehicles_on_link
        downstream_map = self.downstream_map
        signal_fast_map = self.signal_fast_map
        ped_set = self.pedestrian_edge_set
        w_mps = self.ltm.w_mps
        ped_mgr = self.pedestrian_manager
        metrics_tracker = self.metrics_tracker
        link_states: Dict[str, LinkFlowState] = {}
        sink_eids = self.sink_incoming_eids
        gw_eids = self.gw_incoming_eids

        total_vehicles_served = 0.0
        total_vehicles_injected = 0.0

        # Overtime shift offset
        ot_offset_s = self.incident_manager.overtime_shift_s

        for step in range(total_steps):
            t_sim_s = step * dt
            # Adjust effective game timeline if overtime is active
            effective_t = max(0.0, t_sim_s - ot_offset_s) if t_sim_s > 25000.0 else t_sim_s

            slice_idx = min(len(timeline_slices) - 1, int(effective_t / slice_dt_s))
            current_slice = timeline_slices[slice_idx]
            phase_name = current_slice.get("phase", "Game Day")

            # 1. Incident Stress Testing evaluation (O(1) fast lookup)
            incident_deratings, global_speed_mult, unscheduled_evac_vps = self.incident_manager.evaluate_step(
                t_sim_s=t_sim_s,
                edge_meta=None,
            )

            # 2. Calculate Lindsey density and speed for closed-loop self-healing
            lindsey_density = 0.0
            lindsey_speed_mph = 30.0
            if self.lindsey_edge_ids:
                tot_v = sum(vehicles[eid] for eid in self.lindsey_edge_ids)
                tot_jam = sum(edge_meta[eid]["jam_storage_veh"] for eid in self.lindsey_edge_ids)
                lindsey_density = tot_v / max(1.0, tot_jam)
                active_l_speeds = [link_states[eid].speed_mph for eid in self.lindsey_edge_ids if eid in link_states]
                if active_l_speeds:
                    lindsey_speed_mph = sum(active_l_speeds) / len(active_l_speeds)

            # Anomaly detection & self-healing response
            if self.self_healing is not None:
                self.self_healing.detect_anomalies(
                    t_sim_s=t_sim_s,
                    lindsey_density_ratio=lindsey_density,
                    chokepoint_speed_mph=lindsey_speed_mph,
                )

            # 3. Pedestrian, Coordinator, and Perimeter Routing
            ped_mult = ped_mgr.get_pedestrian_surge_multiplier(effective_t)
            raw_inflows = current_slice.get("gateway_inflows", {})

            if self.coordinator is not None:
                current_inflows, coord_ped_factor = self.coordinator.step(
                    t_sim_s=effective_t,
                    edge_meta=edge_meta,
                    link_states=link_states,
                    raw_gateway_inflows=raw_inflows,
                    ped_surge_multiplier=ped_mult,
                    lindsey_density_ratio=lindsey_density,
                )
                base_ped_factor = coord_ped_factor
            else:
                current_inflows = dict(raw_inflows)
                base_ped_factor = max(0.18, 1.0 - 0.82 * ped_mult)

            eff_ped_factor = self.human_factors.get_pedestrian_factor_with_jaywalking(base_ped_factor)

            # Apply Human Factors & CAV Fleet Compliance to Inflow Diversions
            if self.self_healing is not None and self.self_healing.is_incident_active:
                emergency_target = self.self_healing.get_emergency_diversion_target(0.40)
                eff_diversion = self.human_factors.compute_effective_diversion(
                    target_diversion_fraction=emergency_target,
                    compliance_override=self.fleet_compliance,
                )
                # Divert a fraction away from I-35 Lindsey gateway to East/South bypasses
                i35_inflow = current_inflows.get("GW_I35_NORTH", 0.0)
                diverted_veh = i35_inflow * eff_diversion * 0.50
                current_inflows["GW_I35_NORTH"] = max(0.0, i35_inflow - diverted_veh)
                current_inflows["GW_CLASSEN_NORTH"] = current_inflows.get("GW_CLASSEN_NORTH", 0.0) + (diverted_veh * 0.6)
                current_inflows["GW_SOONER_NORTH"] = current_inflows.get("GW_SOONER_NORTH", 0.0) + (diverted_veh * 0.4)

            # Boundary arrivals are retained until physical entry space is available.
            arrivals = {}
            for gw in self.gateways:
                volume = current_inflows.get(gw.id, 0.0) * dt / slice_dt_s
                out_links = gateway_links.get(gw.id, [])
                if volume and not out_links:
                    raise ValueError(f"Gateway {gw.id} has demand but no outgoing links")
                generated += volume
                for eid in out_links:
                    arrivals[eid] = arrivals.get(eid, 0.0) + volume / len(out_links)
            if unscheduled_evac_vps > 0 and self.lindsey_edge_ids:
                generated += unscheduled_evac_vps * dt
                for eid in self.lindsey_edge_ids:
                    arrivals[eid] = arrivals.get(eid, 0.0) + unscheduled_evac_vps * dt / len(self.lindsey_edge_ids)
            admitted += admit_boundary(vehicles, edge_meta, boundary_queue, arrivals, dt)
            boundary_tstt += sum(boundary_queue.values()) * dt / 3600.0

            # 5. Compute Link Flow States
            active_eids = [eid for eid, v in vehicles.items() if v > 0.0]
            link_states = {}

            for eid in active_eids:
                meta = edge_meta[eid]
                v_count = vehicles[eid]
                eff_len = meta["length_m"]
                density_vpm = v_count / eff_len

                # Incident & rubbernecking capacity adjustments
                inc_cap_mult = incident_deratings.get(eid, 1.0)
                dist_m = self.edge_dist_to_stadium[eid]
                is_game_active = (14400.0 <= effective_t <= 25920.0)
                rn_mult = self.human_factors.get_rubbernecking_multiplier(dist_m, is_event_active=is_game_active)

                eff_cap_vps = meta["capacity_vps"] * inc_cap_mult * rn_mult
                eff_v_free = meta["free_speed_mps"] * global_speed_mult
                jam_storage = meta["jam_storage_veh"]
                jam_density_vpm = jam_storage / eff_len

                rho_crit_vpm = eff_cap_vps / max(1.0, eff_v_free)
                sending_vps = min(eff_v_free * density_vpm, eff_cap_vps)
                remaining_space = max(0.0, jam_density_vpm - density_vpm)
                receiving_vps = min(eff_cap_vps, w_mps * remaining_space)

                if density_vpm <= rho_crit_vpm:
                    speed_mps = eff_v_free
                else:
                    speed_mps = max(0.8, sending_vps / max(1e-4, density_vpm))

                if density_vpm > rho_crit_vpm and jam_density_vpm > rho_crit_vpm:
                    q_frac = (density_vpm - rho_crit_vpm) / (jam_density_vpm - rho_crit_vpm)
                    queue_m = min(eff_len, eff_len * max(0.0, min(1.0, q_frac)))
                else:
                    queue_m = 0.0

                link_states[eid] = LinkFlowState(
                    edge_id=eid,
                    density_vpm=density_vpm,
                    vehicles_on_link=v_count,
                    speed_mps=speed_mps,
                    speed_mph=speed_mps * 2.23694,
                    sending_capacity_vps=sending_vps,
                    receiving_capacity_vps=receiving_vps,
                    queue_length_m=queue_m,
                    is_spillback=(queue_m >= eff_len * 0.70),
                )

            # 6. Inter-Link Transfers
            net_delta: Dict[str, float] = {}
            proposals = []
            receiving = {eid: min(
                (link_states[eid].receiving_capacity_vps if eid in link_states else m["capacity_vps"]) * dt,
                max(0.0, m["jam_storage_veh"] - vehicles[eid])) for eid, m in edge_meta.items()}

            for eid in active_eids:
                state = link_states[eid]
                ds_edges = downstream_map.get(eid)

                # One exit budget per source; absorption cannot also be forwarded.
                is_exit = (eid in sink_eids and effective_t < 25200.0) or (effective_t >= 25200.0 and eid in gw_eids) or not ds_edges
                if is_exit:
                    exit_flow = min(state.vehicles_on_link, state.sending_capacity_vps * dt)
                    net_delta[eid] = net_delta.get(eid, 0.0) - exit_flow
                    exited += exit_flow
                    total_vehicles_served += exit_flow
                    continue

                # Signal green calculation
                if self.coordinator is not None:
                    green_ratio = self.coordinator.get_signal_green_fraction(eid, effective_t)
                elif eid in signal_fast_map:
                    is_art, offset_s, cycle_s, art_green_s, clear_s, cross_green_s, node_id = signal_fast_map[eid]

                    # Upstream metering or downstream flush adjustments
                    if self.self_healing is not None and self.self_healing.is_incident_active:
                        if is_art:
                            art_green_s = self.self_healing.get_adaptive_signal_adjustment(node_id, art_green_s)

                    t_rel = (effective_t + offset_s) % cycle_s
                    if is_art:
                        green_ratio = 1.0 if t_rel < art_green_s else 0.05
                    else:
                        start_cr = art_green_s + (clear_s * 0.5)
                        end_cr = start_cr + cross_green_s
                        green_ratio = 1.0 if start_cr <= t_rel < end_cr else 0.05
                else:
                    green_ratio = 1.0

                eff_green = green_ratio * (eff_ped_factor if eid in ped_set else 1.0)

                total_rec = 0.0
                ds_rec_caps: List[Tuple[str, float]] = []
                for ds_id in ds_edges:
                    rcap = link_states[ds_id].receiving_capacity_vps if ds_id in link_states else edge_meta[ds_id]["capacity_vps"]
                    ds_rec_caps.append((ds_id, rcap))
                    total_rec += rcap

                if total_rec <= 0.0:
                    continue

                for ds_id, rcap in ds_rec_caps:
                    share = rcap / total_rec
                    quota_vps = state.sending_capacity_vps * share
                    flow_v = min(quota_vps, rcap) * eff_green * dt
                    flow_v = min(flow_v, state.vehicles_on_link * share)

                    proposals.append((eid, ds_id, flow_v))

            for source, target, volume in allocate_movements(proposals, receiving):
                net_delta[source] = net_delta.get(source, 0.0) - volume
                net_delta[target] = net_delta.get(target, 0.0) + volume

            # 7. Apply Flow Updates
            for eid, delta in net_delta.items():
                vehicles[eid] = max(0.0, vehicles[eid] + delta)

            residual = initial + generated - exited - sum(vehicles.values()) - sum(boundary_queue.values())
            max_residual = max(max_residual, abs(residual))
            if abs(residual) > 1e-5 * max(1.0, generated):
                raise ArithmeticError("Vehicle conservation failed")

            # 8. Metrics Recording
            metrics = metrics_tracker.record_step(
                step=step,
                time_sec=t_sim_s,
                phase_name=phase_name,
                link_states=link_states,
                edge_metadata=edge_meta,
            )

            # 9. Playback Frame (1-minute intervals)
            if step % sample_steps == 0:
                # Shockwave queue on Lindsey
                total_lindsey_queue_m = sum(link_states[eid].queue_length_m for eid in self.lindsey_edge_ids if eid in link_states)
                r_net = self.self_healing.compute_resilience_index(total_vehicles_served, max(1.0, generated)) if self.self_healing else 0.85

                frame_data = {
                    "time_hr": round(t_sim_s / 3600.0, 2),
                    "phase": phase_name,
                    "active_veh": metrics.active_vehicles_on_network,
                    "lindsey_speed_mph": metrics.lindsey_corridor_speed_mph,
                    "i35_ramp_queue_m": metrics.i35_ramp_queue_m,
                    "is_spillback": metrics.is_i35_spillback_hazard,
                    "incident_active": self.self_healing.is_incident_active if self.self_healing else (self.incident_shock is not None),
                    "resilience_index_r_net": r_net,
                    "lindsey_total_queue_m": round(total_lindsey_queue_m, 1),
                    "fleet_compliance_pct": round(self.fleet_compliance * 100.0, 1),
                    "cav_penetration_pct": round(self.cav_penetration * 100.0, 1),
                    "corridor_states": {
                        eid: {
                            "speed_mph": round(link_states[eid].speed_mph, 1),
                            "queue_m": round(link_states[eid].queue_length_m, 1),
                            "density_pct": round(min(100.0, (vehicles[eid] / max(1, edge_meta[eid]["jam_storage_veh"])) * 100.0), 1),
                        }
                        for eid in self.tracked_corridor_ids
                        if eid in link_states and (
                            vehicles[eid] > 0.05
                            or link_states[eid].queue_length_m > 0.1
                            or link_states[eid].speed_mph < edge_meta[eid]["free_speed_mph"] * 0.98
                            or edge_meta[eid]["is_lindsey"]
                        )
                    }
                }
                playback_frames.append(frame_data)

        report = metrics_tracker.get_summary_report()
        report["evidence_kind"] = "uncalibrated_network_simulation"
        report["conservation"] = {"generated_veh": generated, "admitted_veh": admitted,
            "exited_veh": exited, "remaining_veh": sum(vehicles.values()),
            "boundary_queue_veh": sum(boundary_queue.values()), "initial_veh": initial,
            "max_absolute_residual_veh": max_residual}
        report["boundary_wait_veh_hrs"] = boundary_tstt
        report["total_system_time_including_boundary_veh_hrs"] = report["total_system_travel_time_veh_hrs"] + boundary_tstt
        report["playback_frames_count"] = len(playback_frames)
        report["cav_penetration"] = self.cav_penetration
        report["driver_compliance"] = self.driver_compliance
        report["incident_shock"] = self.incident_shock or "NOMINAL"
        report["self_healing_enabled"] = self.enable_self_healing
        nominal_inj = max(1.0, generated)
        report["resilience_index_r_net"] = round(min(1.0, total_vehicles_served / nominal_inj), 3)

        if self.self_healing and self.self_healing.incident_detected_time_s:
            t_det = self.self_healing.incident_detected_time_s
            t_rec = self.self_healing.recovery_completed_time_s or total_duration_s
            report["recovery_time_min"] = round((t_rec - t_det) / 60.0, 1)
        else:
            report["recovery_time_min"] = 0.0

        return {
            "summary": report,
            "playback_frames": playback_frames,
        }
