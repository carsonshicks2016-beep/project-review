"""
Sensor Suite for Off-Road Rover:
16-Beam Raycast Terrain LIDAR with Dynamic Obstacle / Slope Detection,
6-Axis Inertial Measurement Unit (IMU), Suspension Articulation Sensors,
and Course Heading Telemetry.
"""
import numpy as np
from config import (
    RAY_MAX_DIST, FORWARD_FAN_ANGLES, FORWARD_FAN_PITCH,
    NEAR_BUMPER_ANGLES, NEAR_BUMPER_PITCH, WHEELBASE, HALF_TRACK,
    SUSP_REST_LENGTH, SUSP_MIN_LENGTH
)
from vehicle import quat_to_matrix

class RoverSensorSuite:
    def __init__(self, terrain):
        self.terrain = terrain
        self.num_rays = 16
        self.ray_origins_local = []
        self.ray_dirs_local = []
        self._setup_ray_geometry()

    def _setup_ray_geometry(self):
        """Constructs the 16 raycast laser directions in rover body coordinates."""
        # 1. 10 Forward Fan Rays (Mounted on Roof Rack / Front Hood)
        # Azimuth: -55 to +55 deg, Elevation: -18 deg down
        hood_origin = np.array([WHEELBASE / 2.0 + 0.3, 0.0, 0.45])
        p_fan = FORWARD_FAN_PITCH
        cp, sp = np.cos(p_fan), np.sin(p_fan)

        for az in FORWARD_FAN_ANGLES:
            ca, sa = np.cos(az), np.sin(az)
            d = np.array([ca * cp, sa * cp, sp])
            d = d / np.linalg.norm(d)
            self.ray_origins_local.append(hood_origin)
            self.ray_dirs_local.append(d)

        # 2. 4 Near-Bumper Boulder Proximity Rays
        # Azimuth: -30, -10, 10, 30 deg, Elevation: -42 deg down
        bumper_origin = np.array([WHEELBASE / 2.0 + 0.55, 0.0, 0.15])
        p_bump = NEAR_BUMPER_PITCH
        cp_b, sp_b = np.cos(p_bump), np.sin(p_bump)

        for az in NEAR_BUMPER_ANGLES:
            ca, sa = np.cos(az), np.sin(az)
            d = np.array([ca * cp_b, sa * cp_b, sp_b])
            d = d / np.linalg.norm(d)
            self.ray_origins_local.append(bumper_origin)
            self.ray_dirs_local.append(d)

        # 3. 2 Wheel-Track Predictive Ground Scanners (Left & Right wheel tracks)
        left_track_origin = np.array([WHEELBASE / 2.0, HALF_TRACK, 0.2])
        right_track_origin = np.array([WHEELBASE / 2.0, -HALF_TRACK, 0.2])
        down_dir = np.array([0.707, 0.0, -0.707])  # 45 degrees forward-down
        self.ray_origins_local.append(left_track_origin)
        self.ray_dirs_local.append(down_dir)
        self.ray_origins_local.append(right_track_origin)
        self.ray_dirs_local.append(down_dir)

    def scan(self, vehicle):
        """
        Executes all 16 raycasts in world coordinates against the procedural terrain.
        Returns:
            ray_distances: normalized [0, 1]
            ray_endpoints_world: list of 3D hit positions for 3D visualizer
            ray_hit_flags: list of bools
            ray_normals_world: list of 3D normals at hit points
        """
        R = quat_to_matrix(vehicle.quat)
        pos = vehicle.pos

        ray_distances = np.zeros(self.num_rays, dtype=float)
        ray_endpoints_world = []
        ray_hit_flags = []
        ray_normals_world = []

        for i in range(self.num_rays):
            origin_world = pos + R @ self.ray_origins_local[i]
            dir_world = R @ self.ray_dirs_local[i]

            hit_pos, dist, norm, hit = self.terrain.raycast(origin_world, dir_world, max_dist=RAY_MAX_DIST)
            ray_distances[i] = dist / RAY_MAX_DIST
            ray_endpoints_world.append(hit_pos)
            ray_hit_flags.append(hit)
            ray_normals_world.append(norm)

        return ray_distances, ray_endpoints_world, ray_hit_flags, ray_normals_world

    def get_observation(self, vehicle):
        """
        Constructs the complete 28-dimensional normalized observation vector for the AI policy.
        """
        ray_dists, _, _, _ = self.scan(vehicle)

        # Kinematic states in body frame
        R = quat_to_matrix(vehicle.quat)
        v_body = R.T @ vehicle.vel
        omega_body = vehicle.omega
        roll, pitch, yaw = vehicle.get_euler_angles()

        # Suspension travel normalized [0 = fully extended, 1 = fully compressed]
        max_travel = SUSP_REST_LENGTH - SUSP_MIN_LENGTH
        susp_norm = np.clip((SUSP_REST_LENGTH - vehicle.susp_lengths) / max_travel, 0.0, 1.0)

        # Lateral track deviation from centerline (y = 0)
        track_y = np.clip(vehicle.pos[1] / 6.0, -1.0, 1.0)

        # Heading alignment with course (+X direction)
        heading_error = -yaw  # Target heading is 0 rad (facing +X)
        heading_cos = np.cos(heading_error)
        heading_sin = np.sin(heading_error)

        # Normalized features
        obs = np.concatenate([
            ray_dists,                                         # 16 features: [0, 1]
            np.clip(v_body / 8.0, -1.0, 1.0),                  # 3 features: [vx, vy, vz]
            np.clip(omega_body / 3.0, -1.0, 1.0),              # 3 features: [p, q, r]
            [np.clip(roll / (np.pi / 3), -1.0, 1.0)],          # 1 feature: roll
            [np.clip(pitch / (np.pi / 3), -1.0, 1.0)],         # 1 feature: pitch
            susp_norm,                                         # 4 features: suspension
            [track_y],                                         # 1 feature: track offset
            [heading_sin]                                      # 1 feature: heading error
        ])

        return obs.astype(np.float32)
