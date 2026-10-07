"""
Unit tests for drift scoring, trick detection, and combo chaining.
"""

import unittest
import numpy as np
from drift_rl.dynamics import VehicleState
from drift_rl.tracks import build_touge_track
from drift_rl.scoring import DriftScorer, TrickEvent


class TestDriftScoring(unittest.TestCase):

    def setUp(self):
        self.scorer = DriftScorer()
        self.track = build_touge_track()

    def test_straight_driving_no_drift_points(self):
        """Driving straight with 0 slip angle should earn 0 drift points."""
        state = VehicleState(x=0.0, y=0.0, yaw=0.0, vx=12.0, vy=0.0)
        corners = np.zeros((4, 2))
        reward, tricks, spinout = self.scorer.update(state, self.track, corners, dt=0.05)

        self.assertEqual(reward, 0.0)
        self.assertEqual(len(tricks), 0)
        self.assertFalse(spinout)
        self.assertFalse(self.scorer.is_drifting)

    def test_sustained_drift_increases_multiplier(self):
        """Holding a slide (> 20 deg) at speed should build combo multiplier."""
        # 30 degree slide at 12 m/s
        vx = 12.0 * np.cos(np.radians(30))
        vy = 12.0 * np.sin(np.radians(30))
        state = VehicleState(x=0.0, y=0.0, yaw=0.0, vx=vx, vy=vy)
        corners = np.zeros((4, 2))

        init_mult = self.scorer.multiplier
        for _ in range(40):  # 2.0 seconds of drifting
            reward, tricks, spinout = self.scorer.update(state, self.track, corners, dt=0.05)
            self.assertGreater(reward, 0.0)

        self.assertTrue(self.scorer.is_drifting)
        self.assertGreater(self.scorer.multiplier, init_mult)
        self.assertGreater(self.scorer.active_combo_score, 0.0)

    def test_manji_transition_detection(self):
        """Switching slide direction from left to right should trigger a Manji transition."""
        corners = np.zeros((4, 2))
        # 1. Slide left for 0.8 seconds
        s_left = VehicleState(x=0.0, y=0.0, yaw=0.0, vx=10.0, vy=-5.0)
        for _ in range(16):
            self.scorer.update(s_left, self.track, corners, dt=0.05)

        # 2. Slide right (positive vy)
        s_right = VehicleState(x=0.0, y=0.0, yaw=0.0, vx=10.0, vy=5.0)
        tricks_detected = []
        for _ in range(5):
            _, tricks, _ = self.scorer.update(s_right, self.track, corners, dt=0.05)
            tricks_detected.extend(tricks)

        manji_events = [t for t in tricks_detected if "MANJI" in t.name]
        self.assertGreaterEqual(len(manji_events), 1, "Manji transition trick must be recognized")
        self.assertGreaterEqual(self.scorer.manji_chains, 1)

    def test_backward_entry_detection(self):
        """Drifting at > 75 deg slip angle and recovering should trigger BACKWARD ENTRY."""
        corners = np.zeros((4, 2))
        # 1. Normal slide
        s1 = VehicleState(x=0.0, y=0.0, yaw=0.0, vx=12.0, vy=5.0)
        for _ in range(10):
            self.scorer.update(s1, self.track, corners, dt=0.05)

        # 2. Extreme backward slide (80 deg) at 10 m/s
        vx_back = 10.0 * np.cos(np.radians(80))
        vy_back = 10.0 * np.sin(np.radians(80))
        s_back = VehicleState(x=0.0, y=0.0, yaw=0.0, vx=vx_back, vy=vy_back)
        for _ in range(10):
            self.scorer.update(s_back, self.track, corners, dt=0.05)

        # 3. Recovery to 35 degrees
        vx_rec = 8.0 * np.cos(np.radians(35))
        vy_rec = 8.0 * np.sin(np.radians(35))
        s_rec = VehicleState(x=0.0, y=0.0, yaw=0.0, vx=vx_rec, vy=vy_rec)
        tricks_detected = []
        for _ in range(5):
            _, tricks, _ = self.scorer.update(s_rec, self.track, corners, dt=0.05)
            tricks_detected.extend(tricks)

        backward_events = [t for t in tricks_detected if "BACKWARD ENTRY" in t.name]
        self.assertGreaterEqual(len(backward_events), 1, "Backward entry must be recognized upon recovery")

    def test_spinout_penalty(self):
        """Extreme angle with near zero speed triggers spinout detection and breaks combo."""
        corners = np.zeros((4, 2))
        # Build up combo first
        s_drift = VehicleState(x=0.0, y=0.0, yaw=0.0, vx=10.0, vy=5.0)
        for _ in range(20):
            self.scorer.update(s_drift, self.track, corners, dt=0.05)

        self.assertGreater(self.scorer.active_combo_score, 0.0)

        # Spinout state: angle 120 deg, speed 1.0 m/s
        s_spin = VehicleState(x=0.0, y=0.0, yaw=0.0, vx=-0.5, vy=1.0)
        reward, tricks, spinout = self.scorer.update(s_spin, self.track, corners, dt=0.05)

        self.assertTrue(spinout, "Spinout should be flagged")
        self.assertLess(reward, 0.0, "Spinout should apply negative reward")
        self.assertEqual(self.scorer.multiplier, 1.0, "Multiplier should reset")


if __name__ == "__main__":
    unittest.main()
