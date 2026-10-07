"""
Vectorized, high-performance network traffic simulation engine implementing the Link Transmission Model.
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


class TrafficSimulationEngine:
    """
    Simulates network-wide macroscopic and mesoscopic traffic flow under
    unmanaged baseline or autonomous game-day conditions with pre-indexed high-speed vectorization.
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
    ):
        self.graph = graph
        self.gateways = gateways
        self.sinks = sinks
        self.demand_dataset = demand_dataset
        self.dt_s = dt_s
        self.coordinator = coordinator

        self.ltm = LinkTransmissionModel(backward_wave_speed_mps=backward_wave_speed_mps)
        self.signal_controller = BaselineSignalController()
        self.pedestrian_manager = PedestrianConflictManager()
        self.metrics_tracker = MetricsTracker(dt_s=dt_s)

        # Pre-indexed network structures
        self.edge_meta: Dict[str, Dict[str, Any]] = {}
        self.vehicles_on_link: Dict[str, float] = {}
        self.downstream_map: Dict[str, List[str]] = {}

        # Signal and pedestrian fast lookups (O(1) direct integer/tuple math)
        self.signal_fast_map: Dict[str, Tuple[bool, float, float, float, float, float]] = {}
        self.pedestrian_edge_set: Set[str] = set()
        self.lindsey_edge_ids: List[str] = []
        self.i35_ramp_edge_ids: List[str] = []
        self.tracked_corridor_ids: List[str] = []

        self._initialize_network_structures()

    def _initialize_network_structures(self):
        """Precomputes lookup tables, signal parameters, and adjacency maps for high-speed simulation."""
        for u, v, key, data in self.graph.edges(keys=True, data=True):
            attr: EdgeAttributes = data["attr"]
            edge_id = attr.edge_id
            name = attr.name
            lower_name = name.lower()
            hwy = attr.highway_type

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
                "capacity_vph": attr.capacity_vph,
                "capacity_vps": max(0.05, attr.capacity_vph / 3600.0),
                "jam_storage_veh": attr.jam_storage_veh,
                "is_reversible": attr.is_reversible,
            }
            self.vehicles_on_link[edge_id] = 0.0

            # Pre-tag Lindsey and I-35 ramp edges for metrics
            if "lindsey" in lower_name:
                self.lindsey_edge_ids.append(edge_id)
            if "motorway_link" in hwy and any(k in lower_name for k in ["i 35", "i-35", "interstate 35", "lindsey"]):
                self.i35_ramp_edge_ids.append(edge_id)

            # Pre-tag pedestrian conflict links
            if "lindsey" in lower_name or "boyd" in lower_name:
                self.pedestrian_edge_set.add(edge_id)

            # Pre-tag signals for O(1) mathematical lookup
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
                        sig.cross_green_s
                    )
                    break

            # Pre-tag critical corridors for interactive visualization
            if any(k in lower_name for k in ["lindsey", "i 35", "i-35", "interstate 35", "jenkins", "berry", "mcgee", "chautauqua", "classen", "highway 9", "sh 9"]):
                self.tracked_corridor_ids.append(edge_id)

        # Build downstream connectivity map
        for u, v, key, data in self.graph.edges(keys=True, data=True):
            edge_id = data["attr"].edge_id
            out_edges = [out_data["attr"].edge_id for _, _, _, out_data in self.graph.out_edges(v, keys=True, data=True)]
            self.downstream_map[edge_id] = out_edges

        print(f"[SimEngine] Fast indexing complete: {len(self.signal_fast_map)} signalized links, {len(self.tracked_corridor_ids)} tracked visualization corridors.")

        if self.coordinator is not None:
            self.coordinator.initialize_network(self.edge_meta)

    def run_simulation(self, total_duration_s: float = 28800.0, playback_sample_interval_s: float = 60.0) -> Dict[str, Any]:
        """
        Executes the full 8-hour dynamic simulation at high speed.
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

        print(f"[SimEngine] Executing baseline simulation: {total_steps} steps (dt = {self.dt_s}s, total time = {total_duration_s/3600:.1f} hrs)...")

        timeline_slices = self.demand_dataset.get("timeline", [])
        slice_dt_s = 900.0  # 15 minutes per slice

        # Gateway origin outgoing links
        gateway_links: Dict[str, List[str]] = {}
        for gw in self.gateways:
            if gw.nearest_node is not None:
                gw_out = [d["attr"].edge_id for _, _, _, d in self.graph.out_edges(gw.nearest_node, keys=True, data=True)]
                gateway_links[gw.id] = gw_out if gw_out else []

        playback_frames: List[Dict[str, Any]] = []

        # Local variable caching for loop speed
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

        for step in range(total_steps):
            t_sim_s = step * dt
            slice_idx = min(len(timeline_slices) - 1, int(t_sim_s / slice_dt_s))
            current_slice = timeline_slices[slice_idx]
            phase_name = current_slice.get("phase", "Game Day")

            # Evaluate Coordinator & Pedestrian factors
            ped_mult = ped_mgr.get_pedestrian_surge_multiplier(t_sim_s)

            if self.coordinator is not None:
                lindsey_density = 0.0
                if self.lindsey_edge_ids:
                    total_lindsey_veh = sum(vehicles[eid] for eid in self.lindsey_edge_ids)
                    total_lindsey_jam = sum(edge_meta[eid]["jam_storage_veh"] for eid in self.lindsey_edge_ids)
                    lindsey_density = total_lindsey_veh / max(1.0, total_lindsey_jam)

                current_inflows, ped_factor = self.coordinator.step(
                    t_sim_s=t_sim_s,
                    edge_meta=edge_meta,
                    link_states=link_states,
                    raw_gateway_inflows=current_slice.get("gateway_inflows", {}),
                    ped_surge_multiplier=ped_mult,
                    lindsey_density_ratio=lindsey_density,
                )
            else:
                current_inflows = dict(current_slice.get("gateway_inflows", {}))
                ped_factor = max(0.18, 1.0 - 0.82 * ped_mult)

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
            admitted += admit_boundary(vehicles, edge_meta, boundary_queue, arrivals, dt)
            boundary_tstt += sum(boundary_queue.values()) * dt / 3600.0

            # 2. Compute flow states only for active links and their destinations
            active_eids = [eid for eid, v in vehicles.items() if v > 0.0]
            link_states = {}

            # Evaluate states for active edges
            for eid in active_eids:
                meta = edge_meta[eid]
                v_count = vehicles[eid]
                eff_len = meta["length_m"]
                density_vpm = v_count / eff_len
                cap_vps = meta["capacity_vps"]
                v_free = meta["free_speed_mps"]
                jam_storage = meta["jam_storage_veh"]
                jam_density_vpm = jam_storage / eff_len

                rho_crit_vpm = cap_vps / max(1.0, v_free)
                sending_vps = min(v_free * density_vpm, cap_vps)
                remaining_space = max(0.0, jam_density_vpm - density_vpm)
                receiving_vps = min(cap_vps, w_mps * remaining_space)

                if density_vpm <= rho_crit_vpm:
                    speed_mps = v_free
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

            # 4. Inter-link transfers
            net_delta: Dict[str, float] = {}
            proposals = []
            receiving = {eid: min(
                (link_states[eid].receiving_capacity_vps if eid in link_states else m["capacity_vps"]) * dt,
                max(0.0, m["jam_storage_veh"] - vehicles[eid])) for eid, m in edge_meta.items()}

            for eid in active_eids:
                state = link_states[eid]
                ds_edges = downstream_map.get(eid)
                if not ds_edges:
                    # Destination exit
                    exit_flow = min(state.vehicles_on_link, state.sending_capacity_vps * dt)
                    net_delta[eid] = net_delta.get(eid, 0.0) - exit_flow
                    exited += exit_flow
                    continue

                # Signal green ratio (O(1) direct math)
                if self.coordinator is not None:
                    green_ratio = self.coordinator.get_signal_green_fraction(eid, t_sim_s)
                elif eid in signal_fast_map:
                    is_art, offset_s, cycle_s, art_green_s, clear_s, cross_green_s = signal_fast_map[eid]
                    t_rel = (t_sim_s + offset_s) % cycle_s
                    if is_art:
                        green_ratio = 1.0 if t_rel < art_green_s else 0.05
                    else:
                        start_cr = art_green_s + (clear_s * 0.5)
                        end_cr = start_cr + cross_green_s
                        green_ratio = 1.0 if start_cr <= t_rel < end_cr else 0.05
                else:
                    green_ratio = 1.0

                eff_green = green_ratio * (ped_factor if eid in ped_set else 1.0)

                # Sum receiving capacities of downstream options
                total_rec = 0.0
                ds_rec_caps: List[Tuple[str, float]] = []
                for ds_id in ds_edges:
                    if ds_id in link_states:
                        rcap = link_states[ds_id].receiving_capacity_vps
                    else:
                        # Downstream is empty -> receives at full capacity
                        rcap = edge_meta[ds_id]["capacity_vps"]
                    ds_rec_caps.append((ds_id, rcap))
                    total_rec += rcap

                if total_rec <= 0.0:
                    continue  # Gridlock stop

                for ds_id, rcap in ds_rec_caps:
                    share = rcap / total_rec
                    quota_vps = state.sending_capacity_vps * share
                    flow_v = min(quota_vps, rcap) * eff_green * dt
                    flow_v = min(flow_v, state.vehicles_on_link * share)

                    proposals.append((eid, ds_id, flow_v))

            for source, target, volume in allocate_movements(proposals, receiving):
                net_delta[source] = net_delta.get(source, 0.0) - volume
                net_delta[target] = net_delta.get(target, 0.0) + volume

            # 5. Apply flow updates
            for eid, delta in net_delta.items():
                vehicles[eid] = max(0.0, vehicles[eid] + delta)

            residual = initial + generated - exited - sum(vehicles.values()) - sum(boundary_queue.values())
            max_residual = max(max_residual, abs(residual))
            if abs(residual) > 1e-5 * max(1.0, generated):
                raise ArithmeticError("Vehicle conservation failed")

            # 6. Record step metrics
            metrics = metrics_tracker.record_step(
                step=step,
                time_sec=t_sim_s,
                phase_name=phase_name,
                link_states=link_states,
                edge_metadata=edge_meta,
            )

            # 7. Record playback frame (every minute)
            if step % sample_steps == 0:
                frame_data = {
                    "time_hr": round(t_sim_s / 3600.0, 2),
                    "phase": phase_name,
                    "contraflow_mode": self.coordinator.contraflow_mgr.current_mode if self.coordinator else "MODE_BALANCED",
                    "active_veh": metrics.active_vehicles_on_network,
                    "lindsey_speed_mph": metrics.lindsey_corridor_speed_mph,
                    "i35_ramp_queue_m": metrics.i35_ramp_queue_m,
                    "is_spillback": metrics.is_i35_spillback_hazard,
                    "corridor_states": {
                        eid: {
                            "speed_mph": round(link_states[eid].speed_mph, 1) if eid in link_states else edge_meta[eid]["free_speed_mph"],
                            "queue_m": round(link_states[eid].queue_length_m, 1) if eid in link_states else 0.0,
                            "density_pct": round(min(100.0, (vehicles[eid] / max(1, edge_meta[eid]["jam_storage_veh"])) * 100.0), 1),
                        }
                        for eid in self.tracked_corridor_ids
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

        return {
            "summary": report,
            "time_series": [
                {
                    "time_hr": m.time_hr,
                    "phase": m.phase_name,
                    "active_veh": m.active_vehicles_on_network,
                    "tstt_veh_hrs": m.cumulative_tstt_veh_hrs,
                    "delay_veh_hrs": m.cumulative_delay_veh_hrs,
                    "lindsey_speed_mph": m.lindsey_corridor_speed_mph,
                    "i35_queue_m": m.i35_ramp_queue_m,
                    "is_spillback": m.is_i35_spillback_hazard,
                }
                for m in metrics_tracker.time_series
            ],
            "playback_frames": playback_frames,
        }
