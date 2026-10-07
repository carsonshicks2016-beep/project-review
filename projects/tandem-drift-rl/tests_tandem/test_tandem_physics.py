"""
Unit and integration tests for Tandem Drift Physics and Judging.
"""

import os
import sys

# Ensure repository root is on sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from drift_tandem_rl.dynamics import VehicleState, VehicleParams, TandemDynamics
from drift_tandem_rl.tracks import get_track
from drift_tandem_rl.scoring import TandemJudge


def test_wake_cone_geometry():
    """Verify that wake intensity is highest directly behind leader and zero outside cone."""
    dyn = TandemDynamics()
    leader = VehicleState(x=50.0, y=0.0, vx=15.0, vy=0.0, yaw=0.0)

    # Follower directly behind (3 meters back)
    follower_behind = VehicleState(x=47.0, y=0.0, vx=15.0, vy=0.0, yaw=0.0)
    wake_center, _, _ = dyn.compute_wake_interaction(leader, follower_behind)
    assert wake_center > 0.3, f"Expected strong wake in center, got {wake_center}"

    # Follower far laterally (outside wake cone)
    follower_outside = VehicleState(x=47.0, y=10.0, vx=15.0, vy=0.0, yaw=0.0)
    wake_outside, _, _ = dyn.compute_wake_interaction(leader, follower_outside)
    assert wake_outside == 0.0, f"Expected 0 wake outside cone, got {wake_outside}"

    # Follower ahead of leader
    follower_ahead = VehicleState(x=55.0, y=0.0, vx=15.0, vy=0.0, yaw=0.0)
    wake_ahead, _, _ = dyn.compute_wake_interaction(leader, follower_ahead)
    assert wake_ahead == 0.0, f"Expected 0 wake ahead of leader, got {wake_ahead}"


def test_sat_collision_detection():
    """Verify Separating Axis Theorem correctly identifies overlapping car boxes."""
    dyn = TandemDynamics()
    car1 = VehicleState(x=0.0, y=0.0, yaw=0.0)
    car2_far = VehicleState(x=10.0, y=0.0, yaw=0.0)
    car2_overlap = VehicleState(x=1.5, y=0.5, yaw=0.2)

    c1 = dyn.get_car_corners(car1)
    c2_f = dyn.get_car_corners(car2_far)
    c2_o = dyn.get_car_corners(car2_overlap)

    col_far, _, _ = dyn.check_car_collision_sat(c1, c2_f)
    assert not col_far, "Cars 10m apart should not collide"

    col_overlap, pen, normal = dyn.check_car_collision_sat(c1, c2_o)
    assert col_overlap, "Overlapping cars should be detected as collision"
    assert pen > 0.0, "Penetration depth must be positive"


def test_smoke_friction_reduction():
    """Verify tire smoke plumes reduce surface friction coefficient."""
    dyn = TandemDynamics()
    car = VehicleState(x=10.0, y=5.0)

    # No smoke
    mu_clean = dyn.compute_smoke_friction_loss(car, [])
    assert mu_clean == 1.0, "Clean surface should have 1.0 friction scale"

    # Smoke cloud right on car
    smoke = [(10.0, 5.0, 4.0, 0.9)]
    mu_dirty = dyn.compute_smoke_friction_loss(car, smoke)
    assert mu_dirty < 1.0, f"Smoke should reduce friction, got {mu_dirty}"
    assert mu_dirty >= 0.70, "Friction loss should be physically capped"


def test_tandem_judging_sweet_spot():
    """Verify Formula Drift judging rewards sweet-spot door-to-door proximity."""
    judge = TandemJudge()
    track = get_track("touge")

    lead = VehicleState(x=100.0, y=0.0, vx=12.0, vy=3.0, yaw=0.0)
    # Sweet spot clearance (approx 1.5m clearance -> 5.9m CG gap)
    chase_sweet = VehicleState(x=94.1, y=0.0, vx=12.0, vy=3.0, yaw=0.0)
    # Distant clearance (approx 8.0m clearance -> 12.4m CG gap)
    chase_far = VehicleState(x=87.6, y=0.0, vx=12.0, vy=3.0, yaw=0.0)

    corners = np.zeros((4, 2))
    r_lead1, r_chase_sweet, info_sweet = judge.evaluate_step(
        lead, chase_sweet, track, corners, corners, False, 0.0
    )

    judge.reset()
    r_lead2, r_chase_far, info_far = judge.evaluate_step(
        lead, chase_far, track, corners, corners, False, 0.0
    )

    assert r_chase_sweet > r_chase_far, (
        f"Sweet spot reward ({r_chase_sweet}) should exceed distant reward ({r_chase_far})"
    )
    assert info_sweet["in_prox_sweet_spot"], "Sweet spot flag must be True"
    assert not info_far["in_prox_sweet_spot"], "Distant car should not be in sweet spot"


if __name__ == "__main__":
    test_wake_cone_geometry()
    test_sat_collision_detection()
    test_smoke_friction_reduction()
    test_tandem_judging_sweet_spot()
    print("ALL TANDEM DRIFT TESTS PASSED SUCCESSFULLY!")
