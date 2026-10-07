"""
Gymnasium environment for Autonomous Orbital Rendezvous and Satellite Docking.
Features:
  - Clohessy-Wiltshire relative orbital mechanics (LVLH frame).
  - V-bar approach cone / corridor.
  - Keep-Out Zones (KOZ) and target satellite geometry.
  - Soft-capture docking envelope (distance, velocity, alignment tolerances).
  - Realistic fuel depletion and thruster constraints.
  - Multi-stage curriculum support (docking -> proximity -> full rendezvous).
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Optional, Dict, Any

from satellite_rl.dynamics import CWDynamics, OrbitParams, SpacecraftParams


class SatelliteDockingEnv(gym.Env):
    """
    Autonomous Satellite Rendezvous & Docking Environment.

    Target satellite is at origin [0, 0, 0] of LVLH frame.
    Docking port is on the +Y face at [0, 0, 0].
    Chaser approaches from +Y (along V-bar corridor) towards [0, 0, 0].
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 20}

    def __init__(
        self,
        stage: str = "curriculum",  # "docking", "proximity", "rendezvous", or "curriculum"
        curriculum_level: float = 0.0,  # 0.0 = close docking, 1.0 = full rendezvous
        dt: float = 1.0,               # Simulation step size [s]
        max_steps: int = 400,          # Max steps per episode
        orbit_params: Optional[OrbitParams] = None,
        spacecraft_params: Optional[SpacecraftParams] = None,
    ):
        super().__init__()

        self.orbit = orbit_params or OrbitParams()
        self.spacecraft = spacecraft_params or SpacecraftParams()
        self.dyn = CWDynamics(self.orbit, self.spacecraft)

        self.stage = stage
        self.curriculum_level = float(np.clip(curriculum_level, 0.0, 1.0))
        self.dt = dt
        self.max_steps = max_steps

        # Docking Corridor Parameters (V-bar cone along +Y)
        self.corridor_half_angle_rad = np.radians(25.0)  # 25 deg approach cone
        self.corridor_tan = np.tan(self.corridor_half_angle_rad)

        # Docking Tolerances (Soft Capture)
        self.dock_pos_tol = 0.30        # [m] (30 cm capture envelope)
        self.dock_max_speed = 0.12      # [m/s] (12 cm/s safe soft latch)
        self.dock_max_lateral_speed = 0.06  # [m/s] (6 cm/s lateral drift)

        # Safety & Limits
        self.target_radius = 2.0        # [m] collision sphere around target body
        self.max_distance = 3000.0      # [m] boundary limit

        # Action Space: 3D continuous thruster command [-1.0, 1.0] in [X, Y, Z]
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(3,),
            dtype=np.float32
        )

        # Observation Space (10 dimensions):
        # 0: x / 200.0 (radial position normalized)
        # 1: y / 200.0 (along-track position normalized)
        # 2: z / 200.0 (cross-track position normalized)
        # 3: vx / 1.0 (radial velocity normalized)
        # 4: vy / 1.0 (along-track velocity normalized)
        # 5: vz / 1.0 (cross-track velocity normalized)
        # 6: distance / 200.0
        # 7: corridor_alignment (1.0 inside corridor, < 0.0 outside)
        # 8: glideslope_speed_error (closing speed vs nominal curve)
        # 9: fuel_ratio (propellant remaining / initial propellant)
        self.observation_space = spaces.Box(
            low=-20.0,
            high=20.0,
            shape=(10,),
            dtype=np.float32
        )

        # Episode state variables
        self.state = np.zeros(6, dtype=np.float64)
        self.propellant = self.spacecraft.propellant_mass_kg
        self.total_delta_v = 0.0
        self.step_count = 0
        self.prev_distance = 0.0

    def set_curriculum_level(self, level: float):
        """Set curriculum difficulty [0.0 (docking) to 1.0 (full rendezvous)]."""
        self.stage = "curriculum"
        self.curriculum_level = float(np.clip(level, 0.0, 1.0))

    def _get_obs(self) -> np.ndarray:
        x, y, z, vx, vy, vz = self.state
        dist = float(np.linalg.norm(self.state[:3]))
        lateral_dist = float(np.sqrt(x * x + z * z))

        # Corridor alignment metric: positive inside cone, negative outside
        if y > 0.5:
            allowed_radius = y * self.corridor_tan
            corridor_alignment = 1.0 - (lateral_dist / (allowed_radius + 1e-4))
        else:
            corridor_alignment = 1.0 if dist < self.dock_pos_tol else -1.0

        # Glideslope nominal velocity (soft closing curve)
        # At 50m: ~0.4 m/s, at 10m: ~0.18 m/s, at 1m: ~0.05 m/s
        nominal_speed = 0.05 + 0.05 * np.sqrt(max(0.0, dist))
        actual_speed = float(np.linalg.norm(self.state[3:6]))
        speed_error = (actual_speed - nominal_speed) / 2.0

        fuel_ratio = self.propellant / max(1e-4, self.spacecraft.propellant_mass_kg)

        obs = np.array([
            x / 100.0,
            y / 200.0,
            z / 100.0,
            vx / 1.5,
            vy / 1.5,
            vz / 1.5,
            dist / 200.0,
            float(np.clip(corridor_alignment, -2.0, 1.0)),
            float(np.clip(speed_error, -2.0, 2.0)),
            float(fuel_ratio)
        ], dtype=np.float32)

        return obs

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)
        self.step_count = 0
        self.propellant = self.spacecraft.propellant_mass_kg
        self.total_delta_v = 0.0

        # Determine spawn distance and uncertainty based on stage / curriculum
        level = self.curriculum_level
        if self.stage == "docking":
            level = 0.0
        elif self.stage == "proximity":
            level = 0.4
        elif self.stage == "rendezvous":
            level = 1.0

        # Spawn parameters:
        # Docking (level 0): y in [20m, 40m], lateral offset +/- 2m, small closing velocity
        # Rendezvous (level 1): y in [250m, 600m], lateral offset +/- 50m
        min_y = 20.0 + level * 230.0
        max_y = 40.0 + level * 560.0
        y0 = float(self.np_random.uniform(min_y, max_y))

        max_lateral = 2.0 + level * 40.0
        x0 = float(self.np_random.uniform(-max_lateral, max_lateral))
        z0 = float(self.np_random.uniform(-max_lateral, max_lateral))

        max_v = 0.05 + level * 0.25
        vx0 = float(self.np_random.uniform(-max_v, max_v))
        vy0 = float(self.np_random.uniform(-max_v, -0.01))
        vz0 = float(self.np_random.uniform(-max_v, max_v))

        self.state = np.array([x0, y0, z0, vx0, vy0, vz0], dtype=np.float64)
        self.prev_distance = float(np.linalg.norm(self.state[:3]))

        info = {
            "initial_distance": self.prev_distance,
            "curriculum_level": self.curriculum_level,
            "propellant_kg": self.propellant
        }
        return self._get_obs(), info

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        self.step_count += 1
        action = np.clip(action, -1.0, 1.0).astype(np.float64)

        # Scale action to thruster force [N]
        thrust_force = action * self.spacecraft.max_thrust_n

        # Physics simulation step via CW dynamics (RK4)
        new_state, new_fuel, delta_v = self.dyn.step_rk4(
            self.state, thrust_force, self.propellant, self.dt
        )
        self.state = new_state
        self.propellant = new_fuel
        self.total_delta_v += delta_v

        # Geometry metrics
        x, y, z, vx, vy, vz = self.state
        dist = float(np.linalg.norm(self.state[:3]))
        speed = float(np.linalg.norm(self.state[3:6]))
        lateral_dist = float(np.sqrt(x * x + z * z))
        lateral_speed = float(np.sqrt(vx * vx + vz * vz))

        # Check corridor status
        in_corridor = (y > 0.0) and (lateral_dist <= y * self.corridor_tan + 0.3)

        # Termination & Truncation checks
        terminated = False
        truncated = False
        info = {
            "success": False,
            "collision": False,
            "out_of_bounds": False,
            "out_of_fuel": False,
            "distance": dist,
            "speed": speed,
            "delta_v": self.total_delta_v,
            "propellant_remaining": self.propellant,
            "in_corridor": in_corridor
        }

        # Bounded, well-conditioned reward formulation
        # 1. Distance potential reward
        dist_progress = self.prev_distance - dist
        r_progress = 3.0 * dist_progress

        # 2. Continuous lateral alignment penalty (bounded)
        r_lateral = -0.3 * min(5.0, lateral_dist) - 0.2 * min(2.0, lateral_speed)

        # 3. Terminal docking funnel guidance (extra encouragement inside 3m)
        if dist < 4.0 and y > 0:
            r_terminal = 0.5 * (4.0 - dist)
        else:
            r_terminal = 0.0

        # 4. Glideslope speed constraint & terminal braking
        target_speed = 0.08 + 0.08 * np.sqrt(max(0.0, dist))
        speed_excess = max(0.0, speed - target_speed)

        if dist < 6.0:
            # Active deceleration zone: strong penalty for coming in too fast
            r_speed = -4.0 * speed_excess
            if speed <= self.dock_max_speed:
                r_speed += 0.8  # bonus for approaching at safe docking speed
        else:
            r_speed = -1.0 * min(3.0, speed_excess)

        # 5. Corridor alignment bonus
        r_corridor = 0.1 if in_corridor else -0.3

        # 6. Propellant efficiency penalty
        thrust_norm = float(np.linalg.norm(action))
        r_fuel = -0.003 * thrust_norm

        reward = r_progress + r_lateral + r_terminal + r_speed + r_corridor + r_fuel

        # --- Terminal Conditions ---
        # Case A: Successful Soft Docking!
        if dist <= self.dock_pos_tol and y >= -0.05:
            if speed <= self.dock_max_speed and lateral_speed <= self.dock_max_lateral_speed:
                terminated = True
                fuel_bonus = 20.0 * (self.propellant / self.spacecraft.propellant_mass_kg)
                reward += 100.0 + fuel_bonus
                info["success"] = True
            else:
                # Arrived at port too fast! Hard capture crash
                terminated = True
                reward -= 30.0
                info["collision"] = True

        # Case B: Physical Collision with Target Satellite
        elif abs(z) > 2.0 and abs(y) < 1.5 and abs(x) < 1.5:
            terminated = True
            reward -= 40.0
            info["collision"] = True
        elif (y <= 0.0 and dist < self.target_radius) or (y < 0.25 and lateral_dist > 0.6):
            terminated = True
            reward -= 40.0
            info["collision"] = True

        # Case C: Out of bounds (slingshot or lost in orbit)
        elif dist > self.max_distance:
            terminated = True
            reward -= 50.0
            info["out_of_bounds"] = True

        # Case D: Out of propellant
        elif self.propellant <= 1e-4:
            terminated = True
            reward -= 25.0
            info["out_of_fuel"] = True

        # Truncation: Max episode steps reached
        if self.step_count >= self.max_steps and not terminated:
            truncated = True
            # Penalty proportional to remaining distance
            reward -= min(20.0, dist * 0.1)

        self.prev_distance = dist
        return self._get_obs(), float(reward), terminated, truncated, info
