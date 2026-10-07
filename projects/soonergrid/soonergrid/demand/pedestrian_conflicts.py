"""
Pedestrian crosswalk conflict surges and dynamic arterial turning capacity derating.
"""

import math
from typing import Dict, List, Tuple, Any

# Primary stadium pedestrian conflict intersections in Norman
PEDESTRIAN_HOTSPOTS = [
    {
        "id": "HOTSPOT_LINDSEY_JENKINS",
        "name": "W Lindsey St & S Jenkins Ave (Stadium SE Corner)",
        "lat": 35.2035,
        "lon": -97.4420,
        "peak_pedestrians_per_hr": 7500.0,
        "impacted_corridor": "W Lindsey St",
    },
    {
        "id": "HOTSPOT_LINDSEY_ASP",
        "name": "W Lindsey St & S Asp Ave (Greek Row Crossing)",
        "lat": 35.2035,
        "lon": -97.4455,
        "peak_pedestrians_per_hr": 4500.0,
        "impacted_corridor": "W Lindsey St",
    },
    {
        "id": "HOTSPOT_BOYD_ASP",
        "name": "W Boyd St & S Asp Ave (Campus Corner Pedestrian Plaza)",
        "lat": 35.2105,
        "lon": -97.4455,
        "peak_pedestrians_per_hr": 6000.0,
        "impacted_corridor": "W Boyd St",
    },
]


class PedestrianConflictManager:
    """
    Simulates the surge of foot traffic across major arterials and calculates
    the resulting capacity degradation on conflicting vehicular links.
    """

    def __init__(self, kickoff_time_s: float = 14400.0, game_duration_s: float = 11520.0):
        self.t_kickoff = kickoff_time_s
        self.t_game_end = kickoff_time_s + game_duration_s

    def get_pedestrian_surge_multiplier(self, t_sim_s: float) -> float:
        """
        Returns normalized pedestrian intensity [0.0, 1.0] across time.
        Peaks heavily at:
        1. Pre-game stadium entry: T - 45 min (t ≈ 11,700s)
        2. Post-game stadium exit: T_end + 10 min (t ≈ 26,520s)
        """
        # Ingress foot surge
        t_pre = self.t_kickoff - 2700.0
        diff_pre = (t_sim_s - t_pre) / 1800.0
        val_pre = math.exp(-0.5 * diff_pre * diff_pre)

        # Egress foot surge (even sharper)
        t_post = self.t_game_end + 600.0
        diff_post = (t_sim_s - t_post) / 1200.0
        val_post = math.exp(-0.5 * diff_post * diff_post)

        return min(1.0, max(0.05, max(val_pre, val_post)))

    def get_effective_capacity_factor(self, road_name: str, t_sim_s: float) -> float:
        """
        Calculates the capacity modifier for a roadway:
        C_eff = C_nominal * factor, where factor in [0.15, 1.0].
        A factor of 0.20 means pedestrian crosswalk crowds have choked 80% of vehicle flow.
        """
        is_impacted = any(
            h["impacted_corridor"].lower() in road_name.lower()
            for h in PEDESTRIAN_HOTSPOTS
        )
        if not is_impacted:
            return 1.0

        ped_intensity = self.get_pedestrian_surge_multiplier(t_sim_s)
        # At peak foot traffic, vehicle capacity drops to 18% of nominal
        factor = 1.0 - (0.82 * ped_intensity)
        return max(0.18, factor)
