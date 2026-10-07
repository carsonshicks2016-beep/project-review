"""
Connected & Autonomous Vehicle (CAV) Fleet Penetration & Platooning Model.
Implements Cooperative Adaptive Cruise Control (CACC) micro-headway scaling,
nonlinear capacity expansion, and fleet-wide compliance boosting under mixed autonomy.
"""

from typing import Dict, List, Tuple, Any, Optional
import math


class CAVPlatooningModel:
    """
    Models the macroscopic impact of Connected & Autonomous Vehicles (CAVs)
    operating with Cooperative Adaptive Cruise Control (CACC) platooning:
    - HDV-HDV Time Headway: h_hdv = 1.50 s
    - Mixed HDV-CAV Time Headway: h_mixed = 1.10 s
    - CAV-CAV Platooning Headway: h_cav = 0.60 s
    """

    def __init__(
        self,
        cav_penetration_rate: float = 0.0,
        h_hdv_s: float = 1.50,
        h_mixed_s: float = 1.10,
        h_cav_s: float = 0.60,
    ):
        self.p = max(0.0, min(1.0, cav_penetration_rate))
        self.h_hdv = h_hdv_s
        self.h_mixed = h_mixed_s
        self.h_cav = h_cav_s

    def set_penetration_rate(self, cav_penetration: float):
        """Sets the market penetration fraction p in [0.0, 1.0]."""
        self.p = max(0.0, min(1.0, cav_penetration))

    def compute_mean_headway_s(self, p: Optional[float] = None) -> float:
        """
        Computes the theoretical mean saturation headway h_bar(p) across the mixed fleet:
        h_bar(p) = (1-p)^2 * h_hdv + 2*p*(1-p) * h_mixed + p^2 * h_cav
        """
        pen = self.p if p is None else max(0.0, min(1.0, p))
        p_hdv_hdv = (1.0 - pen) ** 2
        p_mixed = 2.0 * pen * (1.0 - pen)
        p_cav_cav = pen ** 2

        mean_h = (p_hdv_hdv * self.h_hdv) + (p_mixed * self.h_mixed) + (p_cav_cav * self.h_cav)
        return round(mean_h, 3)

    def compute_capacity_multiplier(self, p: Optional[float] = None) -> float:
        """
        Computes the arterial saturation capacity scaling factor:
        M_C(p) = h_hdv / h_bar(p)
        - At p = 0.0: M_C = 1.00 (nominal capacity)
        - At p = 0.5: M_C ≈ 1.40 (+40% throughput)
        - At p = 1.0: M_C = 2.50 (+150% throughput)
        """
        mean_h = self.compute_mean_headway_s(p)
        return round(self.h_hdv / max(0.1, mean_h), 3)

    def compute_jam_storage_multiplier(self, p: Optional[float] = None) -> float:
        """
        Computes jam storage expansion due to shorter standstill bumper-to-bumper gaps.
        At p = 1.0, standstill distance drops from 7.5m to 6.0m (+25% storage capacity).
        """
        pen = self.p if p is None else max(0.0, min(1.0, p))
        return round(1.0 + 0.25 * pen, 3)

    def compute_fleet_compliance(
        self,
        human_compliance: float,
        p: Optional[float] = None,
    ) -> float:
        """
        Computes the composite network compliance rate.
        CAVs exhibit 100% deterministic adherence to AI diversion advisories and green waves.
        C_fleet(p) = (1 - p) * C_human + p * 1.0
        """
        pen = self.p if p is None else max(0.0, min(1.0, p))
        c_hum = max(0.0, min(1.0, human_compliance))
        fleet_c = ((1.0 - pen) * c_hum) + (pen * 1.0)
        return round(fleet_c, 3)
