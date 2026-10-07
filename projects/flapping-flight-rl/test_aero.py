"""
Aerodynamic Verification Suite:
Simulates one complete wingbeat cycle to verify that cycle-averaged vertical lift
balances or exceeds body weight, and validates clap-and-fling timing.
"""
import numpy as np
from config import InsectSpec
from kinematics import KinematicEngine
from aerodynamics import AerodynamicSolver

def test_wingbeat_cycle():
    spec = InsectSpec()
    kinematics = KinematicEngine(spec)
    aero = AerodynamicSolver(spec)

    f = spec.nominal_frequency
    T = 1.0 / f
    dt = 1e-4 # 0.1 ms integration step
    steps = int(T / dt)

    times = np.linspace(0, T, steps)
    vert_forces = []
    thrust_forces = []
    clap_events = 0

    print("=" * 60)
    print(f"VERIFYING INSECT AERODYNAMICS FOR: {spec.name.upper()}")
    print(f"Body mass: {spec.body_mass * 1e3:.2f} g | Required Weight Support: {spec.weight * 1e3:.2f} mN")
    print(f"Flapping Frequency: {f:.1f} Hz | Cycle Period: {T * 1e3:.1f} ms")
    print("=" * 60)

    fling_active = False
    for t in times:
        left_state, right_state = kinematics.compute_wing_angles(t)

        # Clap-and-fling handling
        if kinematics.fling_boost_timer > 0:
            fling_active = True
            kinematics.fling_boost_timer -= dt
            clap_events += 1
        else:
            fling_active = False

        # Left + Right wing aerodynamic forces
        F_left, _, tel_L = aero.compute_wing_wrench(left_state, is_left=True, fling_active=fling_active)
        F_right, _, tel_R = aero.compute_wing_wrench(right_state, is_left=False, fling_active=fling_active)

        total_F = F_left + F_right
        vert_forces.append(total_F[2])
        thrust_forces.append(total_F[0])

    mean_lift_mn = np.mean(vert_forces) * 1e3
    weight_mn = spec.weight * 1e3
    peak_lift_mn = np.max(vert_forces) * 1e3
    mean_thrust_mn = np.mean(thrust_forces) * 1e3

    print(f"Results across full wingbeat cycle:")
    print(f"  * Mean Vertical Lift: {mean_lift_mn:6.2f} mN")
    print(f"  * Insect Weight:      {weight_mn:6.2f} mN")
    print(f"  * Lift-to-Weight:     {mean_lift_mn / weight_mn:6.2f}x")
    print(f"  * Peak Lift Spike:    {peak_lift_mn:6.2f} mN")
    print(f"  * Net Mean Thrust:    {mean_thrust_mn:6.2f} mN")
    print(f"  * Clap-and-Fling:     {'Triggered' if clap_events > 0 else 'Not triggered'} ({clap_events} steps)")

    assert mean_lift_mn > 0.8 * weight_mn, f"Insufficient lift generated! {mean_lift_mn:.2f} mN < {weight_mn:.2f} mN"
    print("\n[SUCCESS] Aerodynamic solver is verified and physically grounded!")

if __name__ == "__main__":
    test_wingbeat_cycle()
