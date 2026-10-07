"""
Master Autonomous Coordinator for Game-Day Traffic Optimization.
Hierarchically manages dynamic contraflow, MARL signal green waves, and perimeter route diversion.
"""

from typing import Dict, List, Tuple, Any, Optional
from soonergrid.policy.contraflow_manager import DynamicContraflowManager
from soonergrid.policy.signal_marl_agent import MARLSignalController
from soonergrid.policy.perimeter_router import PerimeterRouter


class AutonomousCoordinator:
    """
    Two-Tier Hierarchical Controller:
    1. Strategic Tier (Macro):
       - Dynamic Contraflow Re-Striping along Lindsey St & Jenkins Ave
       - Perimeter Inflow Balancing (diverting Moore/OKC & Noble traffic via bypasses)
       - Pulsed Pedestrian Scramble Gating at stadium crosswalks
    2. Tactical Tier (Micro):
       - Multi-Agent Reinforcement Learning (MARL) Green Wave Phase Coordination
       - Real-time adaptive green split expansion during queue surges
    """

    def __init__(
        self,
        kickoff_time_s: float = 14400.0,
        game_duration_s: float = 11520.0,
        progression_speed_mps: float = 13.41,  # 30 mph
    ):
        self.contraflow_mgr = DynamicContraflowManager(
            kickoff_time_s=kickoff_time_s,
            game_duration_s=game_duration_s,
            clearance_duration_s=600.0,
        )
        self.marl_signals = MARLSignalController(progression_speed_mps=progression_speed_mps)
        self.perimeter_router = PerimeterRouter()

        self.last_mode = "MODE_BALANCED"
        self.is_initialized = False

    def initialize_network(self, edge_meta: Dict[str, Dict[str, Any]]):
        """Initializes corridor mappings and pre-indexes signal and contraflow links."""
        self.contraflow_mgr.register_contraflow_twins(edge_meta)
        self.marl_signals.register_signal_edges(edge_meta)
        # Default green wave progression for pre-game ingress (eastbound towards stadium)
        self.marl_signals.synchronize_green_waves(is_egress=False)
        self.is_initialized = True
        print("[AutonomousCoordinator] Initialized network with contraflow corridors and MARL signal agents.")

    def step(
        self,
        t_sim_s: float,
        edge_meta: Dict[str, Dict[str, Any]],
        link_states: Dict[str, Any],
        raw_gateway_inflows: Dict[str, float],
        ped_surge_multiplier: float,
        lindsey_density_ratio: float = 0.0,
    ) -> Tuple[Dict[str, float], float]:
        """
        Executes one hierarchical control step:
        - Updates contraflow mode & modifies link capacities if mode transitioned
        - Synchronizes green wave offsets (inbound vs outbound)
        - Adapts MARL signal green splits
        - Diverts gateway inflows at network perimeter
        - Evaluates pulsed pedestrian scramble gating factor
        Returns:
            (diverted_gateway_inflows, pulsed_pedestrian_factor)
        """
        # 1. Macro Contraflow Mode Evaluation
        mode = self.contraflow_mgr.evaluate_mode(t_sim_s)
        if mode != self.last_mode:
            self.contraflow_mgr.apply_contraflow_modifications(edge_meta, mode)
            # When switching to egress, reverse green wave progression direction to westbound towards I-35
            is_egress = (mode == "MODE_EGRESS_FLUSH")
            self.marl_signals.synchronize_green_waves(is_egress=is_egress)
            self.last_mode = mode

        # 2. Tactical Signal Green Split Adaptation
        is_surge = (mode in ("MODE_INGRESS_TIDAL", "MODE_EGRESS_FLUSH"))
        self.marl_signals.adapt_green_splits(link_states, is_peak_surge=is_surge)

        # 3. Perimeter Gateway Diversion
        diverted_inflows = self.perimeter_router.divert_gateway_inflows(
            t_sim_s=t_sim_s,
            raw_inflows=raw_gateway_inflows,
            is_surge_phase=is_surge,
            lindsey_density_ratio=lindsey_density_ratio,
        )

        # 4. Pulsed Pedestrian Scramble Gating
        ped_factor = self.perimeter_router.get_pedestrian_scramble_factor(
            t_sim_s=t_sim_s,
            baseline_ped_surge_mult=ped_surge_multiplier,
        )

        return diverted_inflows, ped_factor

    def get_signal_green_fraction(self, edge_id: str, t_sim_s: float) -> float:
        """O(1) lookup of dynamic signal green state."""
        return self.marl_signals.get_signal_green_fraction(edge_id, t_sim_s)
