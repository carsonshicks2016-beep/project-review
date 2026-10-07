import unittest
from soonergrid.policy.self_healing_agent import SelfHealingCoordinator

class TestSelfHealingResilience(unittest.TestCase):
    """
    Unit test suite verifying the Adaptive Self-Healing Controller,
    shockwave anomaly detection, upstream metering, downstream flushing,
    and Network Resilience Index (R_net).
    """

    def setUp(self):
        self.coordinator = SelfHealingCoordinator(
            density_anomaly_threshold=0.70,
            recovery_speed_threshold_mph=24.0
        )

    def test_anomaly_detection_trigger(self):
        """Verifies that high density and low speed triggers an emergency anomaly state."""
        # Normal traffic: density 0.4, speed 30 mph -> No anomaly
        active = self.coordinator.detect_anomalies(t_sim_s=7200.0, lindsey_density_ratio=0.40, chokepoint_speed_mph=30.0)
        self.assertFalse(active)
        self.assertFalse(self.coordinator.is_incident_active)

        # Severe shockwave: density 0.75, speed 8.5 mph -> Anomaly triggered
        active = self.coordinator.detect_anomalies(t_sim_s=7500.0, lindsey_density_ratio=0.75, chokepoint_speed_mph=8.5)
        self.assertTrue(active)
        self.assertTrue(self.coordinator.is_incident_active)
        self.assertEqual(self.coordinator.incident_detected_time_s, 7500.0)

    def test_emergency_diversion_boost(self):
        """Active incident must boost base perimeter diversion recommendation."""
        base_diversion = 0.40
        # Inactive state: base diversion unchanged
        self.assertEqual(self.coordinator.get_emergency_diversion_target(base_diversion), 0.40)

        # Activate incident
        self.coordinator.detect_anomalies(t_sim_s=7500.0, lindsey_density_ratio=0.80, chokepoint_speed_mph=6.0)
        # Active state: diversion boosted by +0.35 -> 0.75
        boosted = self.coordinator.get_emergency_diversion_target(base_diversion)
        self.assertAlmostEqual(boosted, 0.75, places=3)

    def test_signal_metering_and_flush(self):
        """
        Upstream intersections must be metered (green reduced),
        and downstream intersections flushed (green extended).
        """
        # Inactive state: baseline green unchanged
        base_green = 50.0
        self.assertEqual(self.coordinator.get_adaptive_signal_adjustment("node_SPUI", base_green), 50.0)

        # Activate incident
        self.coordinator.detect_anomalies(t_sim_s=7500.0, lindsey_density_ratio=0.85, chokepoint_speed_mph=5.0)

        # Upstream metering: node_SPUI green throttled by 0.45 -> 22.5s
        metered_green = self.coordinator.get_adaptive_signal_adjustment("node_SPUI", base_green)
        self.assertAlmostEqual(metered_green, 22.5, places=2)

        # Downstream flush: node_BERRY green expanded by 1.35 -> 67.5s
        flushed_green = self.coordinator.get_adaptive_signal_adjustment("node_BERRY", base_green)
        self.assertAlmostEqual(flushed_green, 67.5, places=2)

    def test_recovery_and_resilience_index(self):
        """Tests incident resolution detection and Network Resilience Index R_net."""
        # Trigger incident
        self.coordinator.detect_anomalies(t_sim_s=7500.0, lindsey_density_ratio=0.80, chokepoint_speed_mph=8.0)
        self.assertTrue(self.coordinator.is_incident_active)

        # Traffic normalizes: speed reaches 25 mph >= 24 mph recovery threshold
        self.coordinator.detect_anomalies(t_sim_s=8700.0, lindsey_density_ratio=0.45, chokepoint_speed_mph=25.0)
        self.assertFalse(self.coordinator.is_incident_active)
        self.assertEqual(self.coordinator.recovery_completed_time_s, 8700.0)

        # Test R_net: 45,000 served out of 50,000 nominal -> R_net = 0.90
        r_net = self.coordinator.compute_resilience_index(served_throughput_veh=45000, nominal_target_throughput_veh=50000)
        self.assertEqual(r_net, 0.90)

if __name__ == '__main__':
    unittest.main()
