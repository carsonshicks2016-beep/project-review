"""
Microscopic Human Factors, Behavioral Heterogeneity, and Compliance Engine.
Models stochastic driver route adherence, rubbernecking/distraction capacity deratings,
and pedestrian jaywalking leakage across pulsed scramble crosswalks.
"""

from typing import Dict, List, Tuple, Any, Optional
import math


class HumanFactorsManager:
    """
    Manages human behavioral imperfections and stochastic driver compliance:
    1. Driver Route Compliance (β_comp):
       - Models compliance with Variable Message Sign (VMS) perimeter diversion advisories.
       - Drivers with lower compliance stubbornly follow default navigation into gridlocked arterials.
    2. Rubbernecking & Event Venue Distraction:
       - Slower reaction times and visual distraction around Memorial Stadium and tailgate plazas.
       - Induces a 10% to 25% capacity reduction on adjacent links during high-crowd phases.
    3. Pedestrian Jaywalking Leakage (κ_jay):
       - Impatient pedestrians crossing during vehicle green phases, creating minor friction
         even with pulsed scramble gating.
    """

    def __init__(
        self,
        base_compliance_rate: float = 0.45,
        rubbernecking_max_penalty: float = 0.25,
        jaywalking_leakage_rate: float = 0.10,
    ):
        self.compliance_rate = base_compliance_rate
        self.rubbernecking_penalty = rubbernecking_max_penalty
        self.jaywalking_leakage = jaywalking_leakage_rate

    def set_compliance_rate(self, compliance_rate: float):
        """Sets the driver adherence fraction [0.0, 1.0]."""
        self.compliance_rate = max(0.0, min(1.0, compliance_rate))

    def compute_effective_diversion(
        self,
        target_diversion_fraction: float,
        compliance_override: Optional[float] = None,
    ) -> float:
        """
        Computes the realized diversion fraction based on driver compliance:
        effective_diversion = target * compliance
        """
        comp = self.compliance_rate if compliance_override is None else compliance_override
        return max(0.0, min(1.0, target_diversion_fraction * comp))

    def get_rubbernecking_multiplier(
        self,
        dist_to_stadium_m: float,
        is_event_active: bool = True,
    ) -> float:
        """
        Computes capacity multiplier due to driver distraction and rubbernecking.
        - Within 800m of stadium: capacity reduces by up to 25% (multiplier = 0.75).
        - Decays smoothly with distance from the stadium core.
        """
        if not is_event_active or dist_to_stadium_m > 1500.0:
            return 1.0

        # Linear ramp from max penalty at stadium (d=0) to 1.0 at 1500m
        proximity_factor = max(0.0, 1.0 - (dist_to_stadium_m / 1500.0))
        penalty = self.rubbernecking_penalty * proximity_factor
        return max(0.70, 1.0 - penalty)

    def get_pedestrian_factor_with_jaywalking(
        self,
        base_ped_factor: float,
        jaywalking_leakage_override: Optional[float] = None,
    ) -> float:
        """
        Applies pedestrian jaywalking leakage to the pulsed scramble capacity factor.
        Even when vehicle gates are green, impatient pedestrians reduce theoretical capacity.
        """
        leakage = self.jaywalking_leakage if jaywalking_leakage_override is None else jaywalking_leakage_override
        if base_ped_factor >= 0.95:
            # During vehicle green phase, jaywalking creates minor friction
            return max(0.75, base_ped_factor * (1.0 - leakage))
        else:
            # During pedestrian all-walk scramble phase
            return base_ped_factor
