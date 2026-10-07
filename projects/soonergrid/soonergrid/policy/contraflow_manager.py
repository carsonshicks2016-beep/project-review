"""
Dynamic contraflow (reversible lane) manager for game-day arterial reconfiguration.
Implements safety clearance buffers, tidal lane capacity reallocation, and dedicated transit corridors.
"""

from typing import Dict, List, Tuple, Any, Optional, Set
from dataclasses import dataclass


@dataclass
class ContraflowCorridor:
    name: str
    inbound_keyword: str      # E.g. "east" or heading towards campus
    outbound_keyword: str     # E.g. "west" or heading towards I-35
    total_physical_lanes: int
    nominal_lane_cap_vph: float
    twin_edge_pairs: List[Tuple[str, str]]  # (inbound_edge_id, outbound_edge_id)


class DynamicContraflowManager:
    """
    Manages macro-level dynamic road re-striping across the 7-hour game-day arc:
    - MODE_BALANCED: Standard bidirectional lanes (2:2 Lindsey St)
    - MODE_INGRESS_TIDAL: 3:1 Inbound towards stadium during peak arrival
    - MODE_CLEARANCE_BUFFER: 10-minute safety buffer clearing tidal lanes before reversal
    - MODE_EGRESS_FLUSH: 0:4 Outbound towards I-35 during post-game evacuation
    """

    def __init__(
        self,
        kickoff_time_s: float = 14400.0,
        game_duration_s: float = 11520.0,
        clearance_duration_s: float = 600.0,  # 10 minutes safety buffer
    ):
        self.t_kickoff = kickoff_time_s
        self.t_game_end = kickoff_time_s + game_duration_s
        self.clearance_s = clearance_duration_s

        # Mode definitions
        self.current_mode = "MODE_BALANCED"
        self.mode_history: List[Tuple[float, str]] = []

        # Track registered twin edges
        self.inbound_edges: Set[str] = set()
        self.outbound_edges: Set[str] = set()
        self.twin_map: Dict[str, str] = {}  # edge_id -> twin_edge_id

    def register_contraflow_twins(self, edge_meta: Dict[str, Dict[str, Any]]):
        """Identifies and registers reversible twin pairs along Lindsey St and Jenkins Ave."""
        for eid, meta in edge_meta.items():
            if not meta["is_reversible"]:
                continue
            name = meta["name"].lower()
            u, v = meta["u"], meta["v"]

            # Lindsey St: Inbound is Eastbound (x increases towards stadium), Outbound is Westbound (x decreases towards I-35)
            if "lindsey" in name:
                # Find reciprocal edge v -> u
                for other_id, other_meta in edge_meta.items():
                    if other_meta["name"].lower() == name and other_meta["u"] == v and other_meta["v"] == u:
                        self.twin_map[eid] = other_id
                        self.twin_map[other_id] = eid
                        # In our metric projection, Stadium is at x ≈ 0, I-35 is at x ≈ -3500m
                        # If edge moves towards positive x -> Inbound (Eastbound)
                        u_node = meta["u"]
                        # Assign direction based on edge ID or registration
                        if "east" in eid or "inbound" in eid or eid < other_id:
                            self.inbound_edges.add(eid)
                            self.outbound_edges.add(other_id)
                        else:
                            self.outbound_edges.add(eid)
                            self.inbound_edges.add(other_id)
                        break

        print(f"[ContraflowManager] Registered {len(self.twin_map)//2} reversible corridor pairs ({len(self.inbound_edges)} inbound, {len(self.outbound_edges)} outbound).")

    def evaluate_mode(self, t_sim_s: float) -> str:
        """Determines the optimal macro contraflow mode based on game-day timeline."""
        # 1. Ingress Tidal Mode: T - 2.5h to T - 0.25h (Peak fan arrival)
        t_ingress_start = self.t_kickoff - 9000.0   # T - 2.5h (5,400s)
        t_ingress_end = self.t_kickoff - 900.0      # T - 15m (13,500s)

        # 2. Clearance Buffer Mode: 10 minutes prior to game end (T_end - 10m to T_end)
        t_clearance_start = self.t_game_end - self.clearance_s  # 25,320s
        t_clearance_end = self.t_game_end                       # 25,920s

        # 3. Egress Flush Mode: Final whistle to T_end + 1.25h
        t_egress_end = self.t_game_end + 4500.0     # 30,420s

        if t_ingress_start <= t_sim_s < t_ingress_end:
            target_mode = "MODE_INGRESS_TIDAL"
        elif t_clearance_start <= t_sim_s < t_clearance_end:
            target_mode = "MODE_CLEARANCE_BUFFER"
        elif t_clearance_end <= t_sim_s < t_egress_end:
            target_mode = "MODE_EGRESS_FLUSH"
        else:
            target_mode = "MODE_BALANCED"

        if target_mode != self.current_mode:
            self.current_mode = target_mode
            self.mode_history.append((t_sim_s, target_mode))
            print(f"[ContraflowManager] Mode transition at t={t_sim_s/3600:.2f}h -> {target_mode}")

        return self.current_mode

    def apply_contraflow_modifications(
        self,
        edge_meta: Dict[str, Dict[str, Any]],
        mode: str,
    ):
        """
        Dynamically adjusts lane allocations and saturation capacities in-place.
        """
        for in_id in self.inbound_edges:
            out_id = self.twin_map.get(in_id)
            if not out_id or out_id not in edge_meta or in_id not in edge_meta:
                continue

            in_meta = edge_meta[in_id]
            out_meta = edge_meta[out_id]
            total_lanes = 4  # Standard Lindsey 4-lane cross-section
            lane_cap_vph = 1100.0

            if mode == "MODE_INGRESS_TIDAL":
                # 3 lanes inbound (East) / 1 lane outbound (West)
                in_meta["lanes"] = 3
                in_meta["capacity_vph"] = 3 * lane_cap_vph
                in_meta["capacity_vps"] = in_meta["capacity_vph"] / 3600.0

                out_meta["lanes"] = 1
                out_meta["capacity_vph"] = 1 * lane_cap_vph
                out_meta["capacity_vps"] = out_meta["capacity_vph"] / 3600.0

            elif mode == "MODE_CLEARANCE_BUFFER":
                # Flush phase: incoming traffic throttled, remaining vehicles clear
                in_meta["lanes"] = 1
                in_meta["capacity_vph"] = 500.0
                in_meta["capacity_vps"] = 500.0 / 3600.0

                out_meta["lanes"] = 2
                out_meta["capacity_vph"] = 2 * lane_cap_vph
                out_meta["capacity_vps"] = out_meta["capacity_vph"] / 3600.0

            elif mode == "MODE_EGRESS_FLUSH":
                # Full 0:4 Outbound flush towards I-35
                in_meta["lanes"] = 0
                in_meta["capacity_vph"] = 0.0
                in_meta["capacity_vps"] = 0.0

                out_meta["lanes"] = 4
                out_meta["capacity_vph"] = 4 * lane_cap_vph
                out_meta["capacity_vps"] = out_meta["capacity_vph"] / 3600.0

            else:
                # MODE_BALANCED: 2:2 default
                in_meta["lanes"] = 2
                in_meta["capacity_vph"] = 2 * lane_cap_vph
                in_meta["capacity_vps"] = in_meta["capacity_vph"] / 3600.0

                out_meta["lanes"] = 2
                out_meta["capacity_vph"] = 2 * lane_cap_vph
                out_meta["capacity_vps"] = out_meta["capacity_vph"] / 3600.0
