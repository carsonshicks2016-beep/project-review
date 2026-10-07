"""
Gymnasium-compatible Environment for 3D Off-Road Rover Terrain Traversal.
Supports both Reinforcement Learning (PPO / SAC via Stable-Baselines3)
and Neuroevolution Genetic Algorithms.
"""
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from config import (
    SIM_DT, CONTROL_FREQ, SUB_STEPS, MAX_STEPS_PER_TRIAL,
    TERRAIN_LENGTH
)
from terrain import ProceduralTerrain
from vehicle import RoverVehicle
from sensors import RoverSensorSuite

class RoverTerrainEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": CONTROL_FREQ}

    def __init__(self, seed=42, render_mode=None):
        super().__init__()
        self.render_mode = render_mode
        self.terrain = ProceduralTerrain(seed=seed)
        self.vehicle = RoverVehicle(self.terrain, initial_pos=[2.0, 0.0])
        self.sensors = RoverSensorSuite(self.terrain)

        # Action Space: [steering (-1 to +1), throttle/brake (-1 to +1)]
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(2,), dtype=np.float32
        )

        # Observation Space: 30 continuous features
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(30,), dtype=np.float32
        )

        self.current_step = 0
        self.prev_x = 2.0
        self.stuck_counter = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.terrain = ProceduralTerrain(seed=seed)
            self.sensors = RoverSensorSuite(self.terrain)

        # Reset vehicle with subtle initial lateral jitter for robust generalization
        init_y = np.random.uniform(-0.5, 0.5) if seed is None else 0.0
        self.vehicle.reset(initial_pos=[2.0, init_y])
        self.current_step = 0
        self.prev_x = self.vehicle.pos[0]
        self.stuck_counter = 0

        obs = self.sensors.get_observation(self.vehicle)
        info = self._get_info()
        return obs, info

    def step(self, action):
        self.current_step += 1
        steer_cmd = float(np.clip(action[0], -1.0, 1.0))
        throttle_cmd = float(np.clip(action[1], -1.0, 1.0))

        # Integrate internal physics sub-steps (100 Hz simulation -> 20 Hz control)
        for _ in range(SUB_STEPS):
            self.vehicle.step(steer_cmd, throttle_cmd, dt=SIM_DT)
            if self.vehicle.is_flipped():
                break

        curr_x = self.vehicle.pos[0]
        dx = curr_x - self.prev_x
        self.prev_x = curr_x

        # Reward formulation
        # 1. Forward progress along track
        r_progress = dx * 5.0
        # 2. Forward velocity bonus
        r_speed = max(0.0, float(self.vehicle.vel[0])) * 0.15
        # 3. Upright stability reward & tilt penalties
        roll, pitch, _ = self.vehicle.get_euler_angles()
        r_stability = 0.05 - 0.25 * (roll**2 + pitch**2)
        # 4. Centerline tracking penalty (avoid drifting off course bounds)
        r_lane = -0.05 * (self.vehicle.pos[1] ** 2)
        # 5. Belly dragging penalty
        r_belly = -0.2 if self.vehicle.belly_contact else 0.0

        step_reward = r_progress + r_speed + r_stability + r_lane + r_belly

        # Termination & Truncation checks
        terminated = False
        truncated = False

        # Catastrophic Rollover / Flip
        if self.vehicle.is_flipped():
            step_reward -= 30.0
            terminated = True

        # Course Victory
        if curr_x >= (TERRAIN_LENGTH - 10.0):
            step_reward += 150.0
            terminated = True

        # Stuck detection (vehicle trapped against boulders or beached)
        if abs(dx) < 0.02 and self.current_step > 20:
            self.stuck_counter += 1
            if self.stuck_counter > (CONTROL_FREQ * 3):  # Stuck for 3 seconds
                step_reward -= 10.0
                terminated = True
        else:
            self.stuck_counter = max(0, self.stuck_counter - 1)

        # Max episode length
        if self.current_step >= MAX_STEPS_PER_TRIAL:
            truncated = True

        obs = self.sensors.get_observation(self.vehicle)
        info = self._get_info()

        return obs, float(step_reward), terminated, truncated, info

    def _get_info(self):
        telem = self.vehicle.get_telemetry()
        dists, endpoints, hits, normals = self.sensors.scan(self.vehicle)
        telem["rays"] = {
            "distances": [round(float(d), 3) for d in dists],
            "endpoints": [[round(float(c), 2) for c in pt] for pt in endpoints],
            "hits": [bool(h) for h in hits]
        }
        telem["step"] = self.current_step
        return telem
