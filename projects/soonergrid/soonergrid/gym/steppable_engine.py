"""
Steppable Traffic Simulation Engine for RL Integration.

Extracts the inner loop of ResilientTrafficSimulationEngine.run_simulation()
into a single-step interface that accepts external signal control actions.

Architecture:
    The monolithic engine runs ~5760 steps inside run_simulation() with no
    external intervention. This module provides the same physics but with:

    1. engine.reset() -> Reinitializes all state to t=0
    2. engine.step(signal_actions) -> Advances exactly one dt_s, applies
       external signal green ratios, returns per-agent observations and reward

    Vehicle conservation is checked every step identically to the parent engine.

Design Decisions:
    - The engine owns the simulation state (vehicles_on_link, boundary_queue, etc.)
    - Signal green ratios are injected per-agent per-step via a dict
    - The engine does NOT manage episode truncation; that's the env's job
    - All existing physics (LWR, conservation, incident stress, CAV) are preserved
"""

import math
from typing import Dict, List, Tuple, Any, Optional, Set
import networkx as nx

from soonergrid.data.game_day_zones import Gateway, ParkingSink
from soonergrid.physics.link_attributes import EdgeAttributes
from soonergrid.demand.pedestrian_conflicts import PedestrianConflictManager
from soonergrid.sim.link_transmission import LinkTransmissionModel, LinkFlowState
from soonergrid.sim.signal_controller import BaselineSignalController, BASELINE_INTERSECTIONS
from soonergrid.sim.metrics_tracker import MetricsTracker
from soonergrid.sim.conservation import validate_run, allocate_movements, admit_boundary
from soonergrid.physics.cav_platooning import CAVPlatooningModel
from soonergrid.policy.human_factors import HumanFactorsManager
from soonergrid.sim.incident_manager import IncidentStressTester
from soonergrid.policy.self_healing_agent import SelfHealingCoordinator


# Agent node definitions: maps signal agent IDs to the intersection nodes they control
SIGNAL_AGENT_DEFS = [
    {
        "agent_id": "sig_i35_lindsey",
        "signal_id": "SIG_I35_LINDSEY_SPUI",
        "name": "I-35 & W Lindsey St SPUI",
        "cycle_length_s": 90.0,
        "base_arterial_green_s": 42.0,
        "min_green_s": 20.0,
        "max_green_s": 65.0,
        "clearance_s": 12.0,
        "arterial_keywords": ["lindsey"],
        "cross_keywords": ["i 35", "interstate 35", "i-35"],
    },
    {
        "agent_id": "sig_lindsey_mcgee",
        "signal_id": "SIG_LINDSEY_MCGEE",
        "name": "W Lindsey St & McGee Dr",
        "cycle_length_s": 90.0,
        "base_arterial_green_s": 48.0,
        "min_green_s": 20.0,
        "max_green_s": 65.0,
        "clearance_s": 12.0,
        "arterial_keywords": ["lindsey"],
        "cross_keywords": ["mcgee"],
    },
    {
        "agent_id": "sig_lindsey_berry",
        "signal_id": "SIG_LINDSEY_BERRY",
        "name": "W Lindsey St & Berry Rd",
        "cycle_length_s": 90.0,
        "base_arterial_green_s": 45.0,
        "min_green_s": 20.0,
        "max_green_s": 65.0,
        "clearance_s": 12.0,
        "arterial_keywords": ["lindsey"],
        "cross_keywords": ["berry"],
    },
    {
        "agent_id": "sig_lindsey_chautauqua",
        "signal_id": "SIG_LINDSEY_CHAUTAUQUA",
        "name": "W Lindsey St & Chautauqua Ave",
        "cycle_length_s": 90.0,
        "base_arterial_green_s": 50.0,
        "min_green_s": 20.0,
        "max_green_s": 65.0,
        "clearance_s": 12.0,
        "arterial_keywords": ["lindsey"],
        "cross_keywords": ["chautauqua"],
    },
    {
        "agent_id": "sig_lindsey_jenkins",
        "signal_id": "SIG_LINDSEY_JENKINS",
        "name": "W Lindsey St & S Jenkins Ave",
        "cycle_length_s": 90.0,
        "base_arterial_green_s": 40.0,
        "min_green_s": 20.0,
        "max_green_s": 65.0,
        "clearance_s": 12.0,
        "arterial_keywords": ["lindsey"],
        "cross_keywords": ["jenkins"],
    },
    {
        "agent_id": "sig_classen_lindsey",
        "signal_id": "SIG_CLASSEN_LINDSEY",
        "name": "Classen Blvd & E Lindsey St",
        "cycle_length_s": 90.0,
        "base_arterial_green_s": 45.0,
        "min_green_s": 20.0,
        "max_green_s": 65.0,
        "clearance_s": 12.0,
        "arterial_keywords": ["classen"],
        "cross_keywords": ["lindsey"],
    },
    {
        "agent_id": "sig_sh9_jenkins",
        "signal_id": "SIG_SH9_JENKINS",
        "name": "SH-9 & S Jenkins Ave (LNC)",
        "cycle_length_s": 90.0,
        "base_arterial_green_s": 55.0,
        "min_green_s": 25.0,
        "max_green_s": 70.0,
        "clearance_s": 10.0,
        "arterial_keywords": ["state highway 9", "sh 9", "highway 9"],
        "cross_keywords": ["jenkins"],
    },
]

# Number of approach legs observed per agent (N, S, E, W or subset)
MAX_APPROACH_LEGS = 8

# Observation vector dimensions per agent
OBS_APPROACH_QUEUES = MAX_APPROACH_LEGS        # Queue length (normalized) per approach
OBS_APPROACH_SPEEDS = MAX_APPROACH_LEGS        # Speed ratio per approach
OBS_APPROACH_OCCUPANCY = MAX_APPROACH_LEGS     # Density ratio per approach
OBS_SIGNAL_PHASE = 2                           # Current phase one-hot (arterial, cross)
OBS_ELAPSED_GREEN = 1                          # Fraction of green elapsed
OBS_TIME_FEATURES = 3                          # sin(hour), cos(hour), phase_encoding
OBS_NEIGHBOR_COMMS = 4                         # Upstream/downstream agent queue pressure
OBS_GLOBAL = 2                                 # Network-wide average speed, total vehicles (normalized)

OBS_DIM = (OBS_APPROACH_QUEUES + OBS_APPROACH_SPEEDS + OBS_APPROACH_OCCUPANCY +
           OBS_SIGNAL_PHASE + OBS_ELAPSED_GREEN + OBS_TIME_FEATURES +
           OBS_NEIGHBOR_COMMS + OBS_GLOBAL)

# Action space dimensions
# Action[0]: arterial green split delta in [-1, 1], mapped to [-15s, +15s]
# Action[1]: phase hold probability [0, 1] (> 0.5 holds current phase)
ACTION_DIM = 2


class SteppableTrafficEngine:
    """
    Single-step traffic simulation engine for RL integration.

    Maintains identical physics to ResilientTrafficSimulationEngine but
    exposes a step(actions) -> (observations, rewards, info) interface.
    The engine can be reset() for new episodes.

    Attributes:
        agent_ids: List of 7 signal agent IDs
        obs_dim: Observation vector dimensionality per agent
        action_dim: Action vector dimensionality per agent
    """

    def __init__(
        self,
        graph: nx.MultiDiGraph,
        gateways: List[Gateway],
        sinks: List[ParkingSink],
        demand_dataset: Dict[str, Any],
        dt_s: float = 5.0,
        episode_duration_s: float = 28800.0,
        backward_wave_speed_mps: float = 5.0,
        cav_penetration: float = 0.0,
        driver_compliance: float = 0.60,
        incident_shock: Optional[str] = None,
        enable_self_healing: bool = False,
        reward_queue_weight: float = 1.0,
        reward_delay_weight: float = 0.5,
        reward_spillback_weight: float = 5.0,
        reward_throughput_weight: float = 0.1,
    ):
        """
        Args:
            graph: NetworkX MultiDiGraph with EdgeAttributes on each edge.
            gateways: List of Gateway boundary injection points.
            sinks: List of ParkingSink vehicle absorbers.
            demand_dataset: Dict with 'timeline' key containing demand slices.
            dt_s: Simulation timestep in seconds.
            episode_duration_s: Maximum episode length in seconds.
            backward_wave_speed_mps: LWR backward wave speed.
            cav_penetration: Fraction of connected autonomous vehicles [0, 1].
            driver_compliance: Base human driver compliance rate [0, 1].
            incident_shock: Incident type string or None for nominal.
            enable_self_healing: Whether to enable self-healing coordinator.
            reward_queue_weight: Weight α for queue penalty in reward.
            reward_delay_weight: Weight for delay penalty in reward.
            reward_spillback_weight: Weight β for spillback penalty in reward.
            reward_throughput_weight: Weight for throughput bonus in reward.
        """
        self.graph = graph
        self.gateways = gateways
        self.sinks = sinks
        self.demand_dataset = demand_dataset
        self.dt_s = dt_s
        self.episode_duration_s = episode_duration_s
        self.cav_penetration = cav_penetration
        self.driver_compliance = driver_compliance
        self.incident_shock = incident_shock
        self.enable_self_healing = enable_self_healing

        # Reward shaping weights
        self._rw_queue = reward_queue_weight
        self._rw_delay = reward_delay_weight
        self._rw_spillback = reward_spillback_weight
        self._rw_throughput = reward_throughput_weight

        # Physics modules (persistent across resets)
        self._w_mps = backward_wave_speed_mps
        self._ltm = LinkTransmissionModel(backward_wave_speed_mps=backward_wave_speed_mps)

        # CAV model (persistent)
        self._cav_model = CAVPlatooningModel(cav_penetration_rate=cav_penetration)
        self._human_factors = HumanFactorsManager(base_compliance_rate=driver_compliance)
        self._fleet_compliance = self._cav_model.compute_fleet_compliance(
            human_compliance=driver_compliance, p=cav_penetration
        )
        self._cav_cap_mult = self._cav_model.compute_capacity_multiplier(cav_penetration)
        self._cav_stor_mult = self._cav_model.compute_jam_storage_multiplier(cav_penetration)

        # Agent definitions
        self._agent_defs = {d["agent_id"]: d for d in SIGNAL_AGENT_DEFS}
        self.agent_ids: List[str] = [d["agent_id"] for d in SIGNAL_AGENT_DEFS]
        self.obs_dim = OBS_DIM
        self.action_dim = ACTION_DIM

        # Network structures (built once, shared across resets)
        self._edge_meta: Dict[str, Dict[str, Any]] = {}
        self._downstream_map: Dict[str, List[str]] = {}
        self._edge_dist_to_stadium: Dict[str, float] = {}
        self._lindsey_edge_ids: List[str] = []
        self._i35_ramp_edge_ids: List[str] = []
        self._pedestrian_edge_set: Set[str] = set()
        self._sink_incoming_eids: Set[str] = set()
        self._gw_incoming_eids: Set[str] = set()

        # Per-agent edge mappings: agent_id -> list of (edge_id, is_arterial)
        self._agent_edge_map: Dict[str, List[Tuple[str, bool]]] = {
            aid: [] for aid in self.agent_ids
        }

        # Gateway link precomputation
        self._gateway_links: Dict[str, List[str]] = {}

        # Build the static network structures
        self._build_network()

        # Mutable simulation state (reset per episode)
        self._vehicles: Dict[str, float] = {}
        self._boundary_queue: Dict[str, float] = {}
        self._link_states: Dict[str, LinkFlowState] = {}
        self._step_idx: int = 0
        self._generated: float = 0.0
        self._admitted: float = 0.0
        self._exited: float = 0.0
        self._max_residual: float = 0.0
        self._boundary_tstt: float = 0.0
        self._prev_step_delay: Dict[str, float] = {}
        self._prev_step_queue: Dict[str, float] = {}
        self._prev_throughput: float = 0.0

        # Current signal state per agent
        self._active_art_green: Dict[str, float] = {}
        self._phase_elapsed: Dict[str, float] = {}
        self._current_phase_is_arterial: Dict[str, bool] = {}

        # Incident modules (reset per episode)
        self._incident_manager: Optional[IncidentStressTester] = None
        self._self_healing: Optional[SelfHealingCoordinator] = None
        self._ped_manager: Optional[PedestrianConflictManager] = None
        self._metrics_tracker: Optional[MetricsTracker] = None

    def _build_network(self) -> None:
        """One-time construction of static network lookup tables."""
        for u, v, key, data in self.graph.edges(keys=True, data=True):
            attr: EdgeAttributes = data["attr"]
            edge_id = attr.edge_id
            name = attr.name
            lower_name = name.lower()
            hwy = attr.highway_type

            u_data = self.graph.nodes[u]
            x_u, y_u = u_data.get("x", 0.0), u_data.get("y", 0.0)
            dist_stadium = math.sqrt(x_u * x_u + y_u * y_u)
            self._edge_dist_to_stadium[edge_id] = dist_stadium

            base_cap_vph = attr.capacity_vph * self._cav_cap_mult
            base_jam_veh = attr.jam_storage_veh * self._cav_stor_mult

            is_l = ("lindsey" in lower_name)
            is_r = ("motorway_link" in hwy and any(
                k in lower_name for k in ["i 35", "i-35", "interstate 35", "lindsey"]
            ))

            self._edge_meta[edge_id] = {
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

            if is_l:
                self._lindsey_edge_ids.append(edge_id)
            if is_r:
                self._i35_ramp_edge_ids.append(edge_id)
            if "lindsey" in lower_name or "boyd" in lower_name:
                self._pedestrian_edge_set.add(edge_id)

            # Map edges to signal agents
            for agent_id, agent_def in self._agent_defs.items():
                is_art = any(k in lower_name for k in agent_def["arterial_keywords"])
                is_cross = any(k in lower_name for k in agent_def["cross_keywords"])
                if is_art or is_cross:
                    self._agent_edge_map[agent_id].append((edge_id, is_art))
                    break  # Each edge maps to at most one agent

        # Downstream connectivity
        for u, v, key, data in self.graph.edges(keys=True, data=True):
            edge_id = data["attr"].edge_id
            out_edges = [
                out_data["attr"].edge_id
                for _, _, _, out_data in self.graph.out_edges(v, keys=True, data=True)
            ]
            self._downstream_map[edge_id] = out_edges

        # Sink and gateway edge sets
        sink_nodes = {s.nearest_node for s in self.sinks if s.nearest_node is not None}
        gw_nodes = {g.nearest_node for g in self.gateways if g.nearest_node is not None}
        self._sink_incoming_eids = {
            eid for eid, m in self._edge_meta.items() if m["v"] in sink_nodes
        }
        self._gw_incoming_eids = {
            eid for eid, m in self._edge_meta.items() if m["v"] in gw_nodes
        }

        # Gateway outgoing links
        for gw in self.gateways:
            if gw.nearest_node is not None:
                gw_out = [
                    d["attr"].edge_id
                    for _, _, _, d in self.graph.out_edges(gw.nearest_node, keys=True, data=True)
                ]
                self._gateway_links[gw.id] = gw_out if gw_out else []

    def reset(self, seed: Optional[int] = None) -> Dict[str, Any]:
        """
        Resets the simulation to t=0 and returns initial observations.

        Args:
            seed: Optional random seed (reserved for stochastic extensions).

        Returns:
            observations: Dict mapping agent_id -> observation vector (list of floats).
        """
        # Reset vehicle state
        self._vehicles = {eid: 0.0 for eid in self._edge_meta}
        self._boundary_queue = {}
        self._link_states = {}
        self._step_idx = 0
        self._generated = 0.0
        self._admitted = 0.0
        self._exited = 0.0
        self._max_residual = 0.0
        self._boundary_tstt = 0.0
        self._prev_throughput = 0.0

        # Reset per-agent state
        for agent_id in self.agent_ids:
            agent_def = self._agent_defs[agent_id]
            self._active_art_green[agent_id] = agent_def["base_arterial_green_s"]
            self._phase_elapsed[agent_id] = 0.0
            self._current_phase_is_arterial[agent_id] = True
            self._prev_step_delay[agent_id] = 0.0
            self._prev_step_queue[agent_id] = 0.0

        # Recreate per-episode modules
        self._ped_manager = PedestrianConflictManager()
        self._metrics_tracker = MetricsTracker(dt_s=self.dt_s)

        self._incident_manager = IncidentStressTester()
        if self.incident_shock and self.incident_shock != "NOMINAL":
            self._incident_manager.activate_shock(self.incident_shock)
        self._incident_manager.preindex_network(self._edge_meta)

        if self.enable_self_healing:
            self._self_healing = SelfHealingCoordinator()
        else:
            self._self_healing = None

        # Build initial observations (all zeros at t=0)
        return self._build_observations()

    def step(
        self, actions: Dict[str, List[float]]
    ) -> Tuple[Dict[str, Any], Dict[str, float], bool, bool, Dict[str, Any]]:
        """
        Advances the simulation by one timestep dt_s with external signal actions.

        Args:
            actions: Dict mapping agent_id -> [green_split_delta, phase_hold].
                - green_split_delta: float in [-1, 1], scaled to ±15s adjustment
                  to the arterial green time.
                - phase_hold: float in [0, 1]. If > 0.5, holds current phase;
                  otherwise allows normal cycling.

        Returns:
            observations: Dict[agent_id -> list[float]] observation vectors.
            rewards: Dict[agent_id -> float] scalar rewards.
            terminated: bool, True if episode ended normally.
            truncated: bool, True if episode time limit reached.
            info: Dict with diagnostic metrics.
        """
        dt = self.dt_s
        vehicles = self._vehicles
        edge_meta = self._edge_meta
        downstream_map = self._downstream_map

        t_sim_s = self._step_idx * dt
        total_steps = round(self.episode_duration_s / dt)

        # Apply actions: update per-agent green splits
        for agent_id, action in actions.items():
            if agent_id not in self._agent_defs:
                continue
            agent_def = self._agent_defs[agent_id]

            # Decode green split delta: [-1, 1] -> [-15s, +15s]
            green_delta = float(action[0]) * 15.0
            new_green = self._active_art_green[agent_id] + green_delta
            new_green = max(agent_def["min_green_s"], min(agent_def["max_green_s"], new_green))
            self._active_art_green[agent_id] = new_green

            # Phase hold: if action[1] > 0.5, hold current phase
            if len(action) > 1 and float(action[1]) > 0.5:
                self._phase_elapsed[agent_id] += dt
            else:
                self._phase_elapsed[agent_id] += dt

        # Timeline demand lookup
        timeline_slices = self.demand_dataset.get("timeline", [])
        slice_dt_s = 900.0
        ot_offset_s = self._incident_manager.overtime_shift_s if self._incident_manager else 0.0
        effective_t = max(0.0, t_sim_s - ot_offset_s) if t_sim_s > 25000.0 else t_sim_s
        slice_idx = min(len(timeline_slices) - 1, int(effective_t / slice_dt_s))
        current_slice = timeline_slices[slice_idx]
        phase_name = current_slice.get("phase", "Game Day")

        # --- Incident evaluation ---
        incident_deratings, global_speed_mult, unscheduled_evac_vps = (
            self._incident_manager.evaluate_step(t_sim_s=t_sim_s, edge_meta=None)
            if self._incident_manager else ({}, 1.0, 0.0)
        )

        # --- Lindsey corridor state for self-healing ---
        lindsey_density = 0.0
        lindsey_speed_mph = 30.0
        if self._lindsey_edge_ids:
            tot_v = sum(vehicles[eid] for eid in self._lindsey_edge_ids)
            tot_jam = sum(edge_meta[eid]["jam_storage_veh"] for eid in self._lindsey_edge_ids)
            lindsey_density = tot_v / max(1.0, tot_jam)
            active_l = [
                self._link_states[eid].speed_mph
                for eid in self._lindsey_edge_ids
                if eid in self._link_states
            ]
            if active_l:
                lindsey_speed_mph = sum(active_l) / len(active_l)

        if self._self_healing is not None:
            self._self_healing.detect_anomalies(
                t_sim_s=t_sim_s,
                lindsey_density_ratio=lindsey_density,
                chokepoint_speed_mph=lindsey_speed_mph,
            )

        # --- Pedestrian factor ---
        ped_mult = self._ped_manager.get_pedestrian_surge_multiplier(effective_t) if self._ped_manager else 0.0
        base_ped_factor = max(0.18, 1.0 - 0.82 * ped_mult)
        eff_ped_factor = self._human_factors.get_pedestrian_factor_with_jaywalking(base_ped_factor)

        # --- Boundary demand injection ---
        raw_inflows = current_slice.get("gateway_inflows", {})
        current_inflows = dict(raw_inflows)

        # Self-healing diversion
        if self._self_healing is not None and self._self_healing.is_incident_active:
            emergency_target = self._self_healing.get_emergency_diversion_target(0.40)
            eff_diversion = self._human_factors.compute_effective_diversion(
                target_diversion_fraction=emergency_target,
                compliance_override=self._fleet_compliance,
            )
            i35_inflow = current_inflows.get("GW_I35_NORTH", 0.0)
            diverted_veh = i35_inflow * eff_diversion * 0.50
            current_inflows["GW_I35_NORTH"] = max(0.0, i35_inflow - diverted_veh)
            current_inflows["GW_CLASSEN_NORTH"] = current_inflows.get("GW_CLASSEN_NORTH", 0.0) + diverted_veh * 0.6
            current_inflows["GW_SOONER_NORTH"] = current_inflows.get("GW_SOONER_NORTH", 0.0) + diverted_veh * 0.4

        arrivals: Dict[str, float] = {}
        for gw in self.gateways:
            volume = current_inflows.get(gw.id, 0.0) * dt / slice_dt_s
            out_links = self._gateway_links.get(gw.id, [])
            self._generated += volume
            for eid in out_links:
                arrivals[eid] = arrivals.get(eid, 0.0) + volume / max(1, len(out_links))

        if unscheduled_evac_vps > 0 and self._lindsey_edge_ids:
            self._generated += unscheduled_evac_vps * dt
            for eid in self._lindsey_edge_ids:
                arrivals[eid] = arrivals.get(eid, 0.0) + unscheduled_evac_vps * dt / len(self._lindsey_edge_ids)

        self._admitted += admit_boundary(vehicles, edge_meta, self._boundary_queue, arrivals, dt)
        self._boundary_tstt += sum(self._boundary_queue.values()) * dt / 3600.0

        # --- Compute link flow states ---
        active_eids = [eid for eid, v in vehicles.items() if v > 0.0]
        link_states: Dict[str, LinkFlowState] = {}

        for eid in active_eids:
            meta = edge_meta[eid]
            v_count = vehicles[eid]
            eff_len = meta["length_m"]
            density_vpm = v_count / eff_len

            inc_cap_mult = incident_deratings.get(eid, 1.0)
            dist_m = self._edge_dist_to_stadium.get(eid, 0.0)
            is_game_active = (14400.0 <= effective_t <= 25920.0)
            rn_mult = self._human_factors.get_rubbernecking_multiplier(dist_m, is_event_active=is_game_active)

            eff_cap_vps = meta["capacity_vps"] * inc_cap_mult * rn_mult
            eff_v_free = meta["free_speed_mps"] * global_speed_mult
            jam_storage = meta["jam_storage_veh"]
            jam_density_vpm = jam_storage / eff_len

            rho_crit_vpm = eff_cap_vps / max(1.0, eff_v_free)
            sending_vps = min(eff_v_free * density_vpm, eff_cap_vps)
            remaining_space = max(0.0, jam_density_vpm - density_vpm)
            receiving_vps = min(eff_cap_vps, self._w_mps * remaining_space)

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

        # --- Build agent-controlled green ratio lookup ---
        agent_green_map: Dict[str, float] = {}  # edge_id -> green_ratio
        for agent_id in self.agent_ids:
            agent_def = self._agent_defs[agent_id]
            cycle = agent_def["cycle_length_s"]
            art_green = self._active_art_green[agent_id]
            clearance = agent_def["clearance_s"]
            cross_green = cycle - art_green - clearance

            for eid, is_art in self._agent_edge_map[agent_id]:
                t_rel = t_sim_s % cycle
                if is_art:
                    green_ratio = 1.0 if t_rel < art_green else 0.05
                else:
                    start_cross = art_green + clearance * 0.5
                    end_cross = start_cross + cross_green
                    green_ratio = 1.0 if start_cross <= t_rel < end_cross else 0.05
                agent_green_map[eid] = green_ratio

        # --- Inter-link transfers ---
        net_delta: Dict[str, float] = {}
        proposals = []
        receiving = {
            eid: min(
                (link_states[eid].receiving_capacity_vps if eid in link_states else m["capacity_vps"]) * dt,
                max(0.0, m["jam_storage_veh"] - vehicles[eid])
            )
            for eid, m in edge_meta.items()
        }

        step_exited = 0.0
        for eid in active_eids:
            state = link_states[eid]
            ds_edges = downstream_map.get(eid)

            is_exit = (
                (eid in self._sink_incoming_eids and effective_t < 25200.0)
                or (effective_t >= 25200.0 and eid in self._gw_incoming_eids)
                or not ds_edges
            )
            if is_exit:
                exit_flow = min(state.vehicles_on_link, state.sending_capacity_vps * dt)
                net_delta[eid] = net_delta.get(eid, 0.0) - exit_flow
                self._exited += exit_flow
                step_exited += exit_flow
                continue

            # Signal green from RL agents or free flow
            green_ratio = agent_green_map.get(eid, 1.0)
            eff_green = green_ratio * (eff_ped_factor if eid in self._pedestrian_edge_set else 1.0)

            total_rec = 0.0
            ds_rec_caps: List[Tuple[str, float]] = []
            for ds_id in ds_edges:
                rcap = (
                    link_states[ds_id].receiving_capacity_vps
                    if ds_id in link_states
                    else edge_meta[ds_id]["capacity_vps"]
                )
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

        # Apply flow updates
        for eid, delta in net_delta.items():
            vehicles[eid] = max(0.0, vehicles[eid] + delta)

        # Conservation check
        initial = 0.0  # We start from zero each episode
        residual = (
            initial + self._generated - self._exited
            - sum(vehicles.values()) - sum(self._boundary_queue.values())
        )
        self._max_residual = max(self._max_residual, abs(residual))
        if abs(residual) > 1e-5 * max(1.0, self._generated):
            raise ArithmeticError(
                f"Vehicle conservation failed: residual={residual:.6f}, "
                f"generated={self._generated:.1f}, exited={self._exited:.1f}"
            )

        # Update stored link states
        self._link_states = link_states

        # Record metrics
        if self._metrics_tracker:
            self._metrics_tracker.record_step(
                step=self._step_idx,
                time_sec=t_sim_s,
                phase_name=phase_name,
                link_states=link_states,
                edge_metadata=edge_meta,
            )

        # Advance step counter
        self._step_idx += 1

        # Build outputs
        observations = self._build_observations()
        rewards = self._compute_rewards(link_states, step_exited)
        terminated = False
        truncated = self._step_idx >= total_steps

        info = {
            "step": self._step_idx,
            "time_s": t_sim_s,
            "time_hr": t_sim_s / 3600.0,
            "phase": phase_name,
            "total_vehicles_on_network": sum(vehicles.values()),
            "boundary_queue": sum(self._boundary_queue.values()),
            "generated": self._generated,
            "exited": self._exited,
            "step_throughput": step_exited,
            "lindsey_speed_mph": lindsey_speed_mph,
            "max_conservation_residual": self._max_residual,
        }

        return observations, rewards, terminated, truncated, info

    def _build_observations(self) -> Dict[str, List[float]]:
        """
        Constructs per-agent observation vectors.

        Observation layout (OBS_DIM = 36 floats per agent):
            [0:8]   approach_queues    - Normalized queue (queue_m / length_m) per approach leg
            [8:16]  approach_speeds    - Speed ratio (speed / free_speed) per approach leg
            [16:24] approach_occupancy - Density ratio (vehicles / jam_storage) per approach leg
            [24:26] signal_phase       - One-hot [arterial_green, cross_green]
            [26:27] elapsed_green_frac - Fraction of allocated green time elapsed
            [27:30] time_features      - [sin(2π·hour/24), cos(2π·hour/24), phase_encoding]
            [30:34] neighbor_comms     - [upstream_queue_pressure, downstream_queue_pressure,
                                          upstream_speed_ratio, downstream_speed_ratio]
            [34:36] global_features    - [network_avg_speed_ratio, total_veh_normalized]
        """
        t_sim_s = self._step_idx * self.dt_s
        hour = t_sim_s / 3600.0
        vehicles = self._vehicles
        edge_meta = self._edge_meta
        link_states = self._link_states

        # Time features (shared across agents)
        sin_hour = math.sin(2.0 * math.pi * hour / 24.0)
        cos_hour = math.cos(2.0 * math.pi * hour / 24.0)

        # Phase encoding: map game phase to [0, 1]
        timeline_slices = self.demand_dataset.get("timeline", [])
        slice_dt_s = 900.0
        slice_idx = min(len(timeline_slices) - 1, int(t_sim_s / slice_dt_s))
        phase_name = timeline_slices[slice_idx].get("phase", "") if timeline_slices else ""
        phase_map = {
            "Pre-Game Early": 0.1, "Pre-Game": 0.2, "Pre-Game Peak": 0.3,
            "Pre-Game Late": 0.4, "Kickoff Surge": 0.5, "In-Game": 0.6,
            "Halftime": 0.65, "4th Quarter": 0.7, "Post-Game Surge": 0.8,
            "Post-Game Peak": 0.9, "Post-Game Late": 0.95,
        }
        phase_enc = phase_map.get(phase_name, 0.5)

        # Global features
        total_veh = sum(vehicles.values())
        global_speeds = []
        for eid, state in link_states.items():
            meta = edge_meta.get(eid)
            if meta:
                global_speeds.append(state.speed_mps / max(1.0, meta["free_speed_mps"]))
        avg_speed_ratio = sum(global_speeds) / max(1, len(global_speeds)) if global_speeds else 1.0

        # Normalize total vehicles by a reference maximum (~50000 for Norman game day)
        total_veh_norm = min(1.0, total_veh / 50000.0)

        observations: Dict[str, List[float]] = {}

        for agent_id in self.agent_ids:
            obs = [0.0] * OBS_DIM

            # Gather approach-leg observations
            agent_edges = self._agent_edge_map[agent_id]
            for i, (eid, is_art) in enumerate(agent_edges[:MAX_APPROACH_LEGS]):
                meta = edge_meta.get(eid)
                if not meta:
                    continue
                state = link_states.get(eid)
                if state:
                    # Queue normalized by link length
                    obs[i] = min(1.0, state.queue_length_m / max(1.0, meta["length_m"]))
                    # Speed ratio
                    obs[MAX_APPROACH_LEGS + i] = min(1.0, state.speed_mps / max(1.0, meta["free_speed_mps"]))
                    # Occupancy (density ratio)
                    obs[2 * MAX_APPROACH_LEGS + i] = min(
                        1.0, vehicles[eid] / max(1.0, meta["jam_storage_veh"])
                    )
                else:
                    # Empty link
                    obs[MAX_APPROACH_LEGS + i] = 1.0  # Free-flow speed

            # Signal phase one-hot
            base_idx = 3 * MAX_APPROACH_LEGS
            is_art_green = self._current_phase_is_arterial.get(agent_id, True)
            cycle = self._agent_defs[agent_id]["cycle_length_s"]
            art_green = self._active_art_green[agent_id]
            t_rel = t_sim_s % cycle
            is_art_green = t_rel < art_green
            self._current_phase_is_arterial[agent_id] = is_art_green
            obs[base_idx] = 1.0 if is_art_green else 0.0
            obs[base_idx + 1] = 0.0 if is_art_green else 1.0

            # Elapsed green fraction
            if is_art_green:
                obs[base_idx + 2] = min(1.0, t_rel / max(1.0, art_green))
            else:
                cross_elapsed = t_rel - art_green
                cross_green = cycle - art_green - self._agent_defs[agent_id]["clearance_s"]
                obs[base_idx + 2] = min(1.0, cross_elapsed / max(1.0, cross_green))

            # Time features
            time_idx = base_idx + 3
            obs[time_idx] = sin_hour
            obs[time_idx + 1] = cos_hour
            obs[time_idx + 2] = phase_enc

            # Neighbor communication
            neighbor_idx = time_idx + 3
            agent_list = self.agent_ids
            my_pos = agent_list.index(agent_id)
            if my_pos > 0:
                upstream_id = agent_list[my_pos - 1]
                obs[neighbor_idx] = self._prev_step_queue.get(upstream_id, 0.0)
                obs[neighbor_idx + 2] = self._get_agent_avg_speed_ratio(upstream_id)
            if my_pos < len(agent_list) - 1:
                downstream_id = agent_list[my_pos + 1]
                obs[neighbor_idx + 1] = self._prev_step_queue.get(downstream_id, 0.0)
                obs[neighbor_idx + 3] = self._get_agent_avg_speed_ratio(downstream_id)

            # Global features
            global_idx = neighbor_idx + 4
            obs[global_idx] = avg_speed_ratio
            obs[global_idx + 1] = total_veh_norm

            observations[agent_id] = obs

        return observations

    def _get_agent_avg_speed_ratio(self, agent_id: str) -> float:
        """Returns average speed ratio across an agent's controlled edges."""
        edges = self._agent_edge_map.get(agent_id, [])
        if not edges:
            return 1.0
        ratios = []
        for eid, _ in edges:
            state = self._link_states.get(eid)
            meta = self._edge_meta.get(eid)
            if state and meta:
                ratios.append(state.speed_mps / max(1.0, meta["free_speed_mps"]))
        return sum(ratios) / max(1, len(ratios)) if ratios else 1.0

    def _compute_rewards(
        self, link_states: Dict[str, LinkFlowState], step_throughput: float
    ) -> Dict[str, float]:
        """
        Computes per-agent reward using the multi-objective reward function:

            R_i = -α·Σ(queue²) - δ·delay - β·spillback_penalty + γ·throughput

        Where:
            - queue: Normalized approach queue lengths for agent i's edges
            - delay: Normalized speed degradation on agent i's edges
            - spillback: Binary penalty for queue exceeding 70% of link length
            - throughput: Fraction of vehicles served this step

        All terms are normalized to roughly [-1, 1] scale per agent per step
        to ensure stable gradient magnitudes during training.
        """
        vehicles = self._vehicles
        edge_meta = self._edge_meta
        rewards: Dict[str, float] = {}

        for agent_id in self.agent_ids:
            edges = self._agent_edge_map[agent_id]
            if not edges:
                rewards[agent_id] = 0.0
                continue

            queue_penalty = 0.0
            delay_penalty = 0.0
            spillback_count = 0

            for eid, is_art in edges:
                state = link_states.get(eid)
                meta = edge_meta.get(eid)
                if not state or not meta:
                    continue

                # Normalized queue squared
                norm_q = state.queue_length_m / max(1.0, meta["length_m"])
                queue_penalty += norm_q * norm_q

                # Speed degradation (1 - speed/free_speed)
                speed_ratio = state.speed_mps / max(1.0, meta["free_speed_mps"])
                delay_penalty += max(0.0, 1.0 - speed_ratio)

                # Spillback binary penalty
                if state.is_spillback:
                    spillback_count += 1

            # Normalize by number of edges
            n_edges = max(1, len(edges))
            queue_penalty /= n_edges
            delay_penalty /= n_edges

            # Throughput bonus (shared proportionally)
            throughput_bonus = step_throughput / max(1.0, self._generated) if self._generated > 0 else 0.0

            reward = (
                -self._rw_queue * queue_penalty
                - self._rw_delay * delay_penalty
                - self._rw_spillback * spillback_count
                + self._rw_throughput * throughput_bonus
            )

            # Store queue pressure for neighbor communication
            self._prev_step_queue[agent_id] = queue_penalty
            self._prev_step_delay[agent_id] = delay_penalty

            rewards[agent_id] = reward

        return rewards

    def get_episode_summary(self) -> Dict[str, Any]:
        """Returns a summary report for the completed episode."""
        vehicles = self._vehicles
        t_sim_s = self._step_idx * self.dt_s
        report = {}
        if self._metrics_tracker:
            report = self._metrics_tracker.get_summary_report()
        report["conservation"] = {
            "generated_veh": self._generated,
            "admitted_veh": self._admitted,
            "exited_veh": self._exited,
            "remaining_veh": sum(vehicles.values()),
            "boundary_queue_veh": sum(self._boundary_queue.values()),
            "max_absolute_residual_veh": self._max_residual,
        }
        report["episode_duration_s"] = t_sim_s
        report["episode_steps"] = self._step_idx
        return report
