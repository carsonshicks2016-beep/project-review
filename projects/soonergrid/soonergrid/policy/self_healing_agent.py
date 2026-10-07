"""
Adaptive Self-Healing Controller for Dynamic Incident Mitigation.
Detects anomalous shockwaves and orchestrates upstream perimeter metering,
downstream emergency flush green waves, and dynamic bypass diversion overrides.
"""

from typing import Dict, List, Tuple, Any, Optional
import math


class SelfHealingCoordinator:
    """
    Closed-loop resilience and incident mitigation controller:
    1. Automated Shockwave Anomaly Detection:
       - Monitors density gradients and unexpected queue spikes along monitored arterials.
    2. Adaptive Incident Response Policies:
       - Perimeter Diversion Surge: Diverts up to 75% of incoming traffic away from the blocked corridor.
       - Upstream Inflow Metering: Throttles green splits upstream to prevent gridlock spillback into the incident.
       - Downstream Bottleneck Flushing: Extends green splits downstream to 80s/90s to drain queues rapidly.
    3. Resilience Metrics Tracking:
       - Computes Time to Recovery (T_recov) and Network Resilience Index (R_net).
    """

    def __init__(
        self,
        density_anomaly_threshold: float = 0.70,  # 70% of jam density
        recovery_speed_threshold_mph: float = 24.0,
    ):
        self.anomaly_threshold = density_anomaly_threshold
        self.recovery_threshold = recovery_speed_threshold_mph

        # State tracking
        self.is_incident_active = False
        self.incident_detected_time_s: Optional[float] = None
        self.incident_cleared_time_s: Optional[float] = None
        self.recovery_completed_time_s: Optional[float] = None

        # Emergency override multipliers
        self.emergency_diversion_boost = 0.35  # Boosts diversion by +35% (e.g. 40% -> 75%)
        self.upstream_metering_factor = 0.45   # Reduces upstream green to 45%
        self.downstream_flush_factor = 1.35    # Expands downstream green by +35%

    def detect_anomalies(
        self,
        t_sim_s: float,
        lindsey_density_ratio: float,
        chokepoint_speed_mph: float,
    ) -> bool:
        """
        Detects anomalous shockwave formation indicating a severe incident or lane blockage.
        """
        # Trigger when density spikes above threshold and corridor speed drops below 12 mph
        anomaly_present = (lindsey_density_ratio >= self.anomaly_threshold) and (chokepoint_speed_mph < 12.0)

        if anomaly_present and not self.is_incident_active:
            self.is_incident_active = True
            self.incident_detected_time_s = t_sim_s
            print(f"[SelfHealing] 🚨 ANOMALY DETECTED at t={t_sim_s/3600:.2f}h! Activating Emergency Resilience Protocol.")

        elif not anomaly_present and self.is_incident_active:
            # Check if corridor has recovered
            if chokepoint_speed_mph >= self.recovery_threshold:
                self.is_incident_active = False
                self.recovery_completed_time_s = t_sim_s
                recov_duration = (t_sim_s - self.incident_detected_time_s) / 60.0
                print(f"[SelfHealing] ✅ CORRIDOR RECOVERED at t={t_sim_s/3600:.2f}h (Recovery duration: {recov_duration:.1f} min).")

        return self.is_incident_active

    def get_emergency_diversion_target(self, base_diversion_target: float) -> float:
        """Surges perimeter detour recommendations during active incidents."""
        if not self.is_incident_active:
            return base_diversion_target
        return min(0.85, base_diversion_target + self.emergency_diversion_boost)

    def get_adaptive_signal_adjustment(self, node_id: str, base_green_s: float) -> float:
        """
        Meters upstream nodes and flushes downstream nodes:
        - Upstream (SPUI, McGee): Throttles green to hold vehicles safely outside the bottleneck.
        - Downstream (Berry, Jenkins): Expands green to clear queued vehicles rapidly.
        """
        if not self.is_incident_active:
            return base_green_s

        if any(k in node_id for k in ["SPUI", "MCGEE"]):
            # Upstream metering: reduce arterial green
            return max(20.0, base_green_s * self.upstream_metering_factor)
        elif any(k in node_id for k in ["BERRY", "JENKINS", "CHAUTAUQUA"]):
            # Downstream flush: expand arterial green up to 75s
            return min(75.0, base_green_s * self.downstream_flush_factor)
        else:
            return base_green_s

    def compute_resilience_index(
        self,
        served_throughput_veh: float,
        nominal_target_throughput_veh: float,
    ) -> float:
        """
        Computes the Network Resilience Index R_net in [0.0, 1.0].
        R_net = Q_served / Q_nominal
        """
        if nominal_target_throughput_veh <= 0.0:
            return 1.0
        return round(min(1.0, served_throughput_veh / nominal_target_throughput_veh), 3)
