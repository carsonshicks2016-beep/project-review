"""
Orbital mechanics simulation using Clohessy-Wiltshire (CW / Hill) equations.
Coordinates: Local-Vertical Local-Horizontal (LVLH) frame centered on target satellite.
  +X: Radial (away from Earth center)
  +Y: In-track (along target velocity vector, V-bar)
  +Z: Cross-track (normal to orbital plane)
"""

import numpy as np
from dataclasses import dataclass

# Physical constants
MU_EARTH = 3.986004418e14  # Earth gravitational parameter [m^3/s^2]
R_EARTH = 6378137.0        # Earth equatorial radius [m]
G0 = 9.80665               # Standard gravity [m/s^2]


@dataclass
class OrbitParams:
    altitude_m: float = 400_000.0  # 400 km LEO
    mu: float = MU_EARTH
    r_earth: float = R_EARTH

    @property
    def r0(self) -> float:
        """Orbital radius of target satellite [m]."""
        return self.r_earth + self.altitude_m

    @property
    def omega(self) -> float:
        """Mean orbital angular velocity [rad/s]."""
        return np.sqrt(self.mu / (self.r0 ** 3))

    @property
    def period(self) -> float:
        """Orbital period [s]."""
        return 2.0 * np.pi / self.omega

    @property
    def orbital_speed(self) -> float:
        """Circular orbital velocity [m/s]."""
        return np.sqrt(self.mu / self.r0)


@dataclass
class SpacecraftParams:
    dry_mass_kg: float = 450.0        # Spacecraft dry mass
    propellant_mass_kg: float = 50.0  # Initial propellant mass
    max_thrust_n: float = 10.0        # Max thrust per axis [N]
    isp_s: float = 220.0              # Specific impulse [s] (hydrazine/cold gas RCS)

    @property
    def total_mass_kg(self) -> float:
        return self.dry_mass_kg + self.propellant_mass_kg


class CWDynamics:
    """
    Clohessy-Wiltshire Relative Dynamics Simulator.
    State vector: [x, y, z, vx, vy, vz]
    Units: meters [m] and meters per second [m/s].
    """

    def __init__(self, orbit: OrbitParams = None, spacecraft: SpacecraftParams = None):
        self.orbit = orbit or OrbitParams()
        self.spacecraft = spacecraft or SpacecraftParams()
        self.omega = self.orbit.omega

    def derivative(self, state: np.ndarray, thrust_force: np.ndarray, mass: float) -> np.ndarray:
        """
        Compute dx/dt for state [x, y, z, vx, vy, vz].
        Thrust force: [Fx, Fy, Fz] in Newtons.
        Mass: current mass in kg.
        """
        x, y, z, vx, vy, vz = state
        w = self.omega
        w2 = w * w

        ax_thrust = thrust_force[0] / mass
        ay_thrust = thrust_force[1] / mass
        az_thrust = thrust_force[2] / mass

        # Clohessy-Wiltshire relative equations of motion
        ax = 3.0 * w2 * x + 2.0 * w * vy + ax_thrust
        ay = -2.0 * w * vx + ay_thrust
        az = -w2 * z + az_thrust

        return np.array([vx, vy, vz, ax, ay, az], dtype=np.float64)

    def step_rk4(
        self,
        state: np.ndarray,
        thrust_force: np.ndarray,
        propellant_mass: float,
        dt: float
    ) -> tuple[np.ndarray, float, float]:
        """
        Integrates state forward by dt using 4th order Runge-Kutta.
        Returns:
            new_state: [x, y, z, vx, vy, vz]
            new_propellant_mass: remaining propellant [kg]
            delta_v: velocity change equivalent expended [m/s]
        """
        current_mass = self.spacecraft.dry_mass_kg + max(0.0, propellant_mass)

        # Thruster clipping and fuel availability
        if propellant_mass <= 1e-6:
            actual_thrust = np.zeros(3, dtype=np.float64)
        else:
            actual_thrust = np.clip(
                thrust_force,
                -self.spacecraft.max_thrust_n,
                self.spacecraft.max_thrust_n
            ).astype(np.float64)

        thrust_magnitude = float(np.linalg.norm(actual_thrust))
        mass_flow_rate = thrust_magnitude / (self.spacecraft.isp_s * G0)
        dm = mass_flow_rate * dt

        # Fuel consumption
        actual_dm = min(dm, max(0.0, propellant_mass))
        new_propellant_mass = max(0.0, propellant_mass - actual_dm)
        delta_v = (thrust_magnitude / current_mass) * dt if current_mass > 0 else 0.0

        # RK4 Integration
        k1 = self.derivative(state, actual_thrust, current_mass)
        k2 = self.derivative(state + 0.5 * dt * k1, actual_thrust, current_mass)
        k3 = self.derivative(state + 0.5 * dt * k2, actual_thrust, current_mass)
        k4 = self.derivative(state + dt * k3, actual_thrust, current_mass)

        new_state = state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

        return new_state, new_propellant_mass, delta_v

    def state_transition_matrix(self, dt: float) -> np.ndarray:
        """
        Analytical State Transition Matrix Phi(dt) for CW equations (unforced).
        """
        w = self.omega
        wt = w * dt
        s = np.sin(wt)
        c = np.cos(wt)

        phi = np.zeros((6, 6), dtype=np.float64)

        # rr block (position to position)
        phi[0, 0] = 4.0 - 3.0 * c
        phi[1, 0] = 6.0 * (s - wt)
        phi[1, 1] = 1.0
        phi[2, 2] = c

        # rv block (velocity to position)
        phi[0, 3] = s / w
        phi[0, 4] = 2.0 * (1.0 - c) / w
        phi[1, 3] = 2.0 * (c - 1.0) / w
        phi[1, 4] = (4.0 * s - 3.0 * wt) / w
        phi[2, 5] = s / w

        # vr block (position to velocity)
        phi[3, 0] = 3.0 * w * s
        phi[4, 0] = 6.0 * w * (c - 1.0)
        phi[5, 2] = -w * s

        # vv block (velocity to velocity)
        phi[3, 3] = c
        phi[3, 4] = 2.0 * s
        phi[4, 3] = -2.0 * s
        phi[4, 4] = 4.0 * c - 3.0
        phi[5, 5] = c

        return phi
