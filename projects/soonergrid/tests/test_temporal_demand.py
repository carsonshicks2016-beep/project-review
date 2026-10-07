"""
Unit tests for game day temporal demand surge curves and mass conservation.
"""

import unittest
from soonergrid.demand.temporal_profile import TemporalDemandProfile


class TestTemporalDemand(unittest.TestCase):
    def setUp(self):
        self.profile = TemporalDemandProfile(
            kickoff_time_s=14400.0,
            game_duration_s=11520.0,
            total_attendee_vehicles=34500,
            background_rate_vph=0.0  # Zero background for pure attendee mass conservation test
        )

    def test_mass_conservation_ingress(self):
        """Integrating ingress rate across time must equal total attendee vehicle count (±2%)."""
        total_integrated = 0.0
        dt = 30.0  # 30-second integration step
        t_max = self.profile.t_kickoff + 1800.0
        
        t = 0.0
        while t < t_max:
            rate = self.profile.get_ingress_rate_vps(t, gateway_weight=1.0)
            total_integrated += rate * dt
            t += dt

        err_pct = abs(total_integrated - self.profile.total_vehicles) / self.profile.total_vehicles
        self.assertLess(err_pct, 0.02, f"Ingress mass conservation failed: {total_integrated} vs {self.profile.total_vehicles}")

    def test_ingress_peak_timing(self):
        """Peak ingress arrival rate must occur between T-2.0h and T-1.0h."""
        peak_time = 0.0
        max_rate = 0.0
        
        for t in range(0, int(self.profile.t_kickoff), 60):
            rate = self.profile.get_ingress_rate_vps(float(t))
            if rate > max_rate:
                max_rate = rate
                peak_time = float(t)

        peak_hours_before_kickoff = (self.profile.t_kickoff - peak_time) / 3600.0
        self.assertGreaterEqual(peak_hours_before_kickoff, 1.0)
        self.assertLessEqual(peak_hours_before_kickoff, 2.0)

    def test_egress_peak_timing(self):
        """Egress peak must occur within 30 minutes after final whistle."""
        peak_time = 0.0
        max_rate = 0.0
        t_start = int(self.profile.t_game_end)
        t_stop = t_start + 7200
        
        for t in range(t_start, t_stop, 30):
            rate = self.profile.get_egress_rate_vps(float(t), lot_capacity=34500)
            if rate > max_rate:
                max_rate = rate
                peak_time = float(t)

        minutes_after_game = (peak_time - self.profile.t_game_end) / 60.0
        self.assertGreaterEqual(minutes_after_game, 5.0)
        self.assertLessEqual(minutes_after_game, 30.0)


if __name__ == "__main__":
    unittest.main()
