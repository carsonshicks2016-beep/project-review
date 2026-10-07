"""
Baseline fixed-time (pre-timed) traffic signal controllers for critical Norman intersections.
"""

from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass


@dataclass
class IntersectionSignal:
    id: str
    name: str
    cycle_length_s: float
    arterial_green_s: float
    cross_green_s: float
    clearance_s: float
    offset_s: float          # Uncoordinated offset relative to t=0
    arterial_keywords: List[str]
    cross_keywords: List[str]


# Real-World Baseline Signal Inventory in Norman, OK
BASELINE_INTERSECTIONS: List[IntersectionSignal] = [
    IntersectionSignal(
        id="SIG_I35_LINDSEY_SPUI",
        name="I-35 & W Lindsey St Single-Point Interchange",
        cycle_length_s=90.0,
        arterial_green_s=42.0,
        cross_green_s=36.0,
        clearance_s=12.0,
        offset_s=0.0,
        arterial_keywords=["lindsey"],
        cross_keywords=["i 35", "interstate 35", "i-35"],
    ),
    IntersectionSignal(
        id="SIG_LINDSEY_MCGEE",
        name="W Lindsey St & McGee Dr",
        cycle_length_s=90.0,
        arterial_green_s=48.0,
        cross_green_s=30.0,
        clearance_s=12.0,
        offset_s=22.0,       # Uncoordinated offset
        arterial_keywords=["lindsey"],
        cross_keywords=["mcgee"],
    ),
    IntersectionSignal(
        id="SIG_LINDSEY_BERRY",
        name="W Lindsey St & Berry Rd (Major Chokepoint)",
        cycle_length_s=90.0,
        arterial_green_s=45.0,
        cross_green_s=33.0,
        clearance_s=12.0,
        offset_s=45.0,       # Clashes with McGee
        arterial_keywords=["lindsey"],
        cross_keywords=["berry"],
    ),
    IntersectionSignal(
        id="SIG_LINDSEY_CHAUTAUQUA",
        name="W Lindsey St & Chautauqua Ave",
        cycle_length_s=90.0,
        arterial_green_s=50.0,
        cross_green_s=28.0,
        clearance_s=12.0,
        offset_s=15.0,
        arterial_keywords=["lindsey"],
        cross_keywords=["chautauqua"],
    ),
    IntersectionSignal(
        id="SIG_LINDSEY_JENKINS",
        name="W Lindsey St & S Jenkins Ave (Stadium SE Corner)",
        cycle_length_s=90.0,
        arterial_green_s=40.0,
        cross_green_s=38.0,
        clearance_s=12.0,
        offset_s=60.0,
        arterial_keywords=["lindsey"],
        cross_keywords=["jenkins"],
    ),
    IntersectionSignal(
        id="SIG_CLASSEN_LINDSEY",
        name="Classen Blvd & E Lindsey St",
        cycle_length_s=90.0,
        arterial_green_s=45.0,
        cross_green_s=33.0,
        clearance_s=12.0,
        offset_s=30.0,
        arterial_keywords=["classen"],
        cross_keywords=["lindsey"],
    ),
    IntersectionSignal(
        id="SIG_SH9_JENKINS",
        name="State Highway 9 & S Jenkins Ave (Lloyd Noble Entrance)",
        cycle_length_s=90.0,
        arterial_green_s=55.0,
        cross_green_s=25.0,
        clearance_s=10.0,
        offset_s=10.0,
        arterial_keywords=["state highway 9", "sh 9", "highway 9"],
        cross_keywords=["jenkins"],
    ),
]


class BaselineSignalController:
    """
    Evaluates fixed-time signal phases across Norman.
    Without adaptive coordination, arterial platoons encounter arbitrary red phases.
    """

    def __init__(self, intersections: Optional[List[IntersectionSignal]] = None):
        self.intersections = intersections or BASELINE_INTERSECTIONS

    def get_signal_green_fraction(self, road_name: str, t_sim_s: float) -> float:
        """
        Determines the instantaneous effective green signal status [0.0, 1.0] for a road link.
        If link does not pass through a signalized chokepoint, returns 1.0 (unrestricted free flow).
        """
        lower_name = road_name.lower()

        # Find matching signal
        for sig in self.intersections:
            is_arterial = any(k in lower_name for k in sig.arterial_keywords)
            is_cross = any(k in lower_name for k in sig.cross_keywords)

            if is_arterial or is_cross:
                # Calculate cycle position
                t_rel = (t_sim_s + sig.offset_s) % sig.cycle_length_s

                if is_arterial:
                    # Arterial green phase
                    if t_rel < sig.arterial_green_s:
                        return 1.0
                    else:
                        return 0.05  # Red phase with minor permitted right turns
                elif is_cross:
                    # Cross street green phase
                    start_cross = sig.arterial_green_s + (sig.clearance_s / 2.0)
                    end_cross = start_cross + sig.cross_green_s
                    if start_cross <= t_rel < end_cross:
                        return 1.0
                    else:
                        return 0.05

        return 1.0  # Non-signalized link
