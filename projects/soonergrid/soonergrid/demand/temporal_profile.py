"""
Temporal demand curves and non-homogeneous Poisson arrival processes
for University of Oklahoma game-day traffic.
"""

import math
from typing import Dict, List, Tuple


class TemporalDemandProfile:
    """
    Simulates the 8-hour game-day temporal arc:
    - 0 to 4 hours: Gradual Ingress (Tailgate setup -> Peak Ingress Surge -> Kickoff rush)
    - 4 to 7.2 hours: In-Game Quiescence (Background city traffic only)
    - 7.2 to 8.0 hours: Post-Game Flash Egress (95% outbound burst)
    """

    def __init__(
        self,
        kickoff_time_s: float = 14400.0,       # T = 4.0 hours (Kickoff)
        game_duration_s: float = 11520.0,      # 3.2 hours game time -> Game ends at T = 7.2 hours (25,920s)
        total_attendee_vehicles: int = 34500,  # ~86,000 stadium capacity @ 2.5 pax/veh
        background_rate_vph: float = 4000.0,   # Baseline ambient city vehicle flow
    ):
        self.t_kickoff = kickoff_time_s
        self.t_game_end = kickoff_time_s + game_duration_s
        self.total_vehicles = total_attendee_vehicles
        self.background_rate_vps = background_rate_vph / 3600.0

        # Pre-compute normalization factors for ingress and egress surge kernels
        self._norm_ingress = self._integrate_kernel(self._ingress_kernel, 0.0, self.t_kickoff + 1800.0, 500)
        self._norm_egress = self._integrate_kernel(self._egress_kernel, self.t_game_end - 900.0, self.t_game_end + 7200.0, 500)

    def _integrate_kernel(self, kernel_fn, t_start: float, t_end: float, steps: int = 500) -> float:
        dt = (t_end - t_start) / steps
        total = 0.0
        for i in range(steps):
            t = t_start + (i + 0.5) * dt
            total += kernel_fn(t) * dt
        return max(1e-6, total)

    def _ingress_kernel(self, t: float) -> float:
        """
        Ingress probability density kernel:
        A skewed bell curve peaking at T - 1.25 hours (t ≈ 9,900s).
        """
        t_peak = self.t_kickoff - 4500.0  # 1.25 hours before kickoff
        if t < 0.0 or t > (self.t_kickoff + 1800.0):
            return 0.0

        # Asymmetric spread: wider before peak (tailgating arrivals), sharper drop after kickoff
        sigma = 3800.0 if t <= t_peak else 1800.0
        diff = (t - t_peak) / sigma
        return math.exp(-0.5 * diff * diff)

    def _egress_kernel(self, t: float) -> float:
        """
        Egress probability density kernel:
        A sharp burst beginning at game conclusion (t ≈ 25,920s),
        peaking 15 minutes post-game, and dissipating over 45 minutes.
        """
        t_peak = self.t_game_end + 900.0  # 15 minutes after final whistle
        if t < (self.t_game_end - 600.0):
            return 0.0

        # Very sharp rise, rapid decay
        sigma = 600.0 if t <= t_peak else 1500.0
        diff = (t - t_peak) / sigma
        return math.exp(-0.5 * diff * diff)

    def get_ingress_rate_vps(self, t_sim_s: float, gateway_weight: float = 1.0) -> float:
        """
        Returns instantaneous vehicle ingress arrival rate (vehicles/second)
        for a given gateway at simulation time t.
        """
        kernel_val = self._ingress_kernel(t_sim_s)
        ingress_vps = (self.total_vehicles / self._norm_ingress) * kernel_val * gateway_weight
        # Add baseline ambient traffic proportion
        return ingress_vps + (self.background_rate_vps * gateway_weight)

    def get_egress_rate_vps(self, t_sim_s: float, lot_capacity: int) -> float:
        """
        Returns instantaneous vehicle discharge rate (vehicles/second)
        from a specific parking lot during post-game evacuation.
        """
        kernel_val = self._egress_kernel(t_sim_s)
        # Fraction of vehicles leaving this specific lot
        lot_total = self.total_vehicles * (lot_capacity / max(1, self.total_vehicles))
        egress_vps = (lot_total / self._norm_egress) * kernel_val
        return egress_vps

    def get_phase_name(self, t_sim_s: float) -> str:
        """Returns human-readable game-day phase name for telemetry."""
        if t_sim_s < (self.t_kickoff - 9000.0):
            return "Phase I: Early Tailgate Arrival"
        elif t_sim_s < (self.t_kickoff - 1800.0):
            return "Phase II: Peak Ingress Surge"
        elif t_sim_s < (self.t_kickoff + 1800.0):
            return "Phase III: Kickoff & Late Push"
        elif t_sim_s < (self.t_game_end - 900.0):
            return "Phase IV: In-Game Quiescence"
        elif t_sim_s < (self.t_game_end + 3600.0):
            return "Phase V: Peak Post-Game Egress"
        else:
            return "Phase VI: Post-Game Dispersion"
