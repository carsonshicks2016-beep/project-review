"""
Perimeter Dynamic Route Diversion and Pulsed Pedestrian Scramble Coordinator.
Diverts regional gateway inflows around gridlocked arterials and groups pedestrian crossings into pulsed windows.
"""

from typing import Dict, List, Tuple, Any, Optional
import math


class PerimeterRouter:
    """
    Manages regional inflow diversion at Norman gateways and pulsed pedestrian scramble gating:
    1. Gateway Diversion:
       - Intercepts I-35 North (GW_I35_NORTH) traffic and diverts up to 40% via 12th Ave NE / Sooner Rd bypass (GW_SOONER_NORTH)
       - Intercepts US-77 South (GW_US77_SOUTH) traffic and diverts up to 50% via Highway 9 to Lloyd Noble Center (GW_SH9_EAST)
       - Driver compliance model: 45% compliance rate for VMS detour recommendations.
    2. Pulsed Pedestrian Scramble:
       - Replaces continuous pedestrian trickle (which causes 82% turning capacity loss) with synchronized
         scramble windows (90s uninterrupted vehicle flow -> 45s exclusive pedestrian all-walk scramble).
    """

    def __init__(
        self,
        i35_diversion_target: float = 0.40,
        us77_diversion_target: float = 0.50,
        compliance_rate: float = 0.45,
        scramble_cycle_s: float = 135.0,
        scramble_walk_s: float = 45.0,
    ):
        self.i35_target = i35_diversion_target
        self.us77_target = us77_diversion_target
        self.compliance = compliance_rate
        self.scramble_cycle = scramble_cycle_s
        self.scramble_walk = scramble_walk_s

        # Effective diversion fractions = target * compliance
        self.effective_i35_divert = self.i35_target * self.compliance   # 0.40 * 0.45 = 18.0% of total I-35 traffic
        self.effective_us77_divert = self.us77_target * self.compliance # 0.50 * 0.45 = 22.5% of total US-77 traffic

        # Diversion activation thresholds
        self.density_threshold = 0.55  # Activate when Lindsey corridor density exceeds 55% jam

    def divert_gateway_inflows(
        self,
        t_sim_s: float,
        raw_inflows: Dict[str, float],
        is_surge_phase: bool = True,
        lindsey_density_ratio: float = 0.0,
    ) -> Dict[str, float]:
        """
        Dynamically redistributes gateway inflows among network perimeter entry points.
        Conserves total vehicle inflow count strictly.
        """
        diverted_inflows = dict(raw_inflows)

        # Diversion active during surge periods or when corridor density is elevated
        should_divert = is_surge_phase or (lindsey_density_ratio >= self.density_threshold)

        if not should_divert:
            return diverted_inflows

        # 1. Divert I-35 North -> 12th Ave NE / Sooner Rd bypass
        i35_vol = diverted_inflows.get("GW_I35_NORTH", 0.0)
        if i35_vol > 0.0 and "GW_SOONER_NORTH" in diverted_inflows:
            divert_amount = i35_vol * self.effective_i35_divert
            diverted_inflows["GW_I35_NORTH"] -= divert_amount
            diverted_inflows["GW_SOONER_NORTH"] += divert_amount

        # 2. Divert US-77 South (Noble) -> SH-9 East (Direct to Lloyd Noble Center)
        us77_vol = diverted_inflows.get("GW_US77_SOUTH", 0.0)
        if us77_vol > 0.0 and "GW_SH9_EAST" in diverted_inflows:
            divert_amount = us77_vol * self.effective_us77_divert
            diverted_inflows["GW_US77_SOUTH"] -= divert_amount
            diverted_inflows["GW_SH9_EAST"] += divert_amount

        return diverted_inflows

    def get_pedestrian_scramble_factor(
        self,
        t_sim_s: float,
        baseline_ped_surge_mult: float,
    ) -> float:
        """
        Computes effective vehicle capacity factor under pulsed pedestrian scramble control.
        - Outside peak pedestrian surge (baseline_mult < 0.1): factor = 1.0 (no conflicts)
        - During peak pedestrian surge:
          - During vehicle green phase (t_rel < 90s): factor = 1.0 (sidewalk gates closed, 0 pedestrian conflicts)
          - During pedestrian scramble phase (90s <= t_rel < 135s): factor = 0.05 (all-direction pedestrian crossing)
        Average capacity during vehicle phase is 1.0 instead of baseline 0.18, eliminating continuous bottlenecking.
        """
        if baseline_ped_surge_mult < 0.15:
            return 1.0

        t_rel = t_sim_s % self.scramble_cycle
        vehicle_phase_s = self.scramble_cycle - self.scramble_walk  # 90s

        if t_rel < vehicle_phase_s:
            # Vehicle phase: pedestrian crossing prohibited, turning capacity completely uninhibited
            return 1.0
        else:
            # Pedestrian scramble phase: crosswalk active in all directions, vehicle movement stopped
            return 0.05
