"""
Dynamic Incident Generator & Real-Time Stress-Testing Engine.
Simulates arterial collisions, severe Oklahoma thunderstorm squalls, and overtime demand shifts.
"""

from typing import Dict, List, Tuple, Any, Optional, Set
from dataclasses import dataclass, field
import math


@dataclass
class IncidentEvent:
    id: str
    name: str
    incident_type: str  # "collision", "weather_squall", "overtime_delay"
    start_time_s: float
    duration_s: float
    target_corridor_keywords: List[str]
    capacity_multiplier: float
    speed_multiplier: float
    unscheduled_evac_rate_vph: float = 0.0
    description: str = ""
    affected_edges: Set[str] = field(default_factory=set)

    @property
    def end_time_s(self) -> float:
        return self.start_time_s + self.duration_s

    def is_active(self, t_sim_s: float) -> bool:
        return self.start_time_s <= t_sim_s < self.end_time_s


class IncidentStressTester:
    """
    Injects and manages calibrated environmental disturbances and crisis shocks:
    - SHOCK_COLLISION: 2-lane blockage at Lindsey & Berry during peak ingress.
    - SHOCK_THUNDERSTORM: Flash severe weather squall at halftime causing roadway wetting,
      capacity drops, and sudden tailgater crowd evacuation.
    - SHOCK_OVERTIME: Double overtime game delay (+45 min), shifting the egress surge.
    """

    def __init__(self, kickoff_time_s: float = 14400.0, game_duration_s: float = 11520.0):
        self.t_kickoff = kickoff_time_s
        self.t_game_end = kickoff_time_s + game_duration_s

        self.registered_incidents: Dict[str, IncidentEvent] = {}
        self.active_incidents: Set[str] = set()

        # Overtime delay parameter
        self.overtime_shift_s = 0.0

        # Pre-configure research-grade benchmark shocks
        self._setup_default_shocks()

    def _setup_default_shocks(self):
        """Initializes calibrated research-grade disturbance scenarios."""
        # 1. Chokepoint collision: T - 1.5h (9,000s), lasts 35 min (2,100s)
        self.registered_incidents["SHOCK_COLLISION"] = IncidentEvent(
            id="SHOCK_COLLISION",
            name="Major Multi-Vehicle Collision at W Lindsey St & Berry Rd",
            incident_type="collision",
            start_time_s=self.t_kickoff - 5400.0,  # 9,000s (2.5h)
            duration_s=2100.0,                     # 35 minutes
            target_corridor_keywords=["lindsey", "berry"],
            capacity_multiplier=0.33,              # 2 of 3 inbound lanes blocked
            speed_multiplier=0.40,
            unscheduled_evac_rate_vph=0.0,
            description="2-vehicle crash chokes Lindsey/Berry intersection; inbound capacity collapses by 67%."
        )

        # 2. Oklahoma Thunderstorm Squall: Halftime T + 1.5h (19,800s), lasts 45 min (2,700s)
        self.registered_incidents["SHOCK_THUNDERSTORM"] = IncidentEvent(
            id="SHOCK_THUNDERSTORM",
            name="Flash Oklahoma Thunderstorm Squall & Roadway Wetting",
            incident_type="weather_squall",
            start_time_s=self.t_kickoff + 5400.0,  # 19,800s (5.5h)
            duration_s=2700.0,                     # 45 minutes
            target_corridor_keywords=["all"],      # City-wide impact
            capacity_multiplier=0.60,              # 40% capacity drop from standing water
            speed_multiplier=0.65,                 # 35% speed drop from torrential rain
            unscheduled_evac_rate_vph=8000.0,      # Tailgaters evacuate immediately
            description="Sudden severe thunderstorm hits Norman; speeds drop by 35%, 15k tailgaters evacuate."
        )

        # 3. Double Overtime Extension: Shifts egress by +45 minutes
        self.registered_incidents["SHOCK_OVERTIME"] = IncidentEvent(
            id="SHOCK_OVERTIME",
            name="Double Overtime Game Extension (+45 min)",
            incident_type="overtime_delay",
            start_time_s=self.t_game_end - 600.0,  # 25,320s
            duration_s=2700.0,                     # 45 minutes delay
            target_corridor_keywords=[],
            capacity_multiplier=1.0,
            speed_multiplier=1.0,
            unscheduled_evac_rate_vph=0.0,
            description="Game locked in tie; egress postponed by 45 minutes, creating compressed dark evacuation."
        )

    def preindex_network(self, edge_meta: Dict[str, Dict[str, Any]]):
        """Precomputes exact edge IDs affected by incidents for O(1) runtime evaluation."""
        for event in self.registered_incidents.values():
            if event.incident_type == "collision":
                event.affected_edges = {
                    eid for eid, meta in edge_meta.items()
                    if any(k in meta["name"].lower() for k in event.target_corridor_keywords)
                }

    def activate_shock(self, shock_id: str):
        """Manually enables a specific shock event."""
        if shock_id in self.registered_incidents:
            self.active_incidents.add(shock_id)
            if shock_id == "SHOCK_OVERTIME":
                self.overtime_shift_s = 2700.0  # +45 minutes
            print(f"[IncidentManager] Activated crisis shock: {shock_id}")

    def deactivate_shock(self, shock_id: str):
        """Deactivates a specific shock event."""
        if shock_id in self.active_incidents:
            self.active_incidents.remove(shock_id)
            if shock_id == "SHOCK_OVERTIME":
                self.overtime_shift_s = 0.0
            print(f"[IncidentManager] Deactivated crisis shock: {shock_id}")

    def evaluate_step(
        self,
        t_sim_s: float,
        edge_meta: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> Tuple[Dict[str, float], float, float]:
        """
        Evaluates active incidents for timestep t_sim_s.
        Returns:
            (edge_capacity_derating_factors, global_speed_factor, unscheduled_evacuation_vps)
        """
        if not self.active_incidents:
            return {}, 1.0, 0.0

        edge_deratings: Dict[str, float] = {}
        global_speed_factor = 1.0
        unscheduled_evac_vps = 0.0

        for shock_id in self.active_incidents:
            event = self.registered_incidents[shock_id]
            if not event.is_active(t_sim_s):
                continue

            # Weather squall: global impact
            if event.incident_type == "weather_squall":
                global_speed_factor = min(global_speed_factor, event.speed_multiplier)
                unscheduled_evac_vps += (event.unscheduled_evac_rate_vph / 3600.0)
                if edge_meta is not None:
                    cap_mult = event.capacity_multiplier
                    for eid in edge_meta:
                        edge_deratings[eid] = min(edge_deratings.get(eid, 1.0), cap_mult)

            # Localized arterial crash: use pre-indexed set
            elif event.incident_type == "collision":
                if event.affected_edges:
                    for eid in event.affected_edges:
                        edge_deratings[eid] = min(edge_deratings.get(eid, 1.0), event.capacity_multiplier)
                elif edge_meta is not None:
                    # Fallback if not preindexed
                    for eid, meta in edge_meta.items():
                        if any(k in meta["name"].lower() for k in event.target_corridor_keywords):
                            edge_deratings[eid] = min(edge_deratings.get(eid, 1.0), event.capacity_multiplier)

        return edge_deratings, global_speed_factor, unscheduled_evac_vps
