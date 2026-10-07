"""
Gymnasium environment for reinforcement learning of high-speed drifting and trick chaining.
"""

from typing import Optional, Dict, Any, Tuple
import gymnasium as gym
from gymnasium import spaces
import numpy as np

from drift_rl.dynamics import VehicleDynamics, VehicleParams, VehicleState
from drift_rl.tracks import Track, get_track
from drift_rl.scoring import DriftScorer, TrickEvent


class DriftGymkhanaEnv(gym.Env):
    """
    Continuous control environment for teaching autonomous drift cars
    to maximize angle, speed, and combo chaining.
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50}

    def __init__(
        self,
        track_name: str = "touge",
        params: Optional[VehicleParams] = None,
        max_steps: int = 1200,
        render_mode: Optional[str] = None,
    ):
        super().__init__()
        self.track_name = track_name
        self.track = get_track(track_name)
        self.params = params or VehicleParams()
        self.dyn = VehicleDynamics(self.params)
        self.scorer = DriftScorer()
        self.max_steps = max_steps
        self.render_mode = render_mode
        self.dt = 0.02  # 50 Hz physics

        # Action: [steer (-1 to 1), throttle/brake (-1 to 1), handbrake (0 to 1)]
        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0, 0.0], dtype=np.float32),
            high=np.array([1.0, 1.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )

        # Observation dimension: 35
        # 6 (vehicle) + 5 (combo) + 6 (waypoints) + 2 (heading) + 16 (LIDAR)
        obs_dim = 6 + 5 + 6 + 2 + 16
        self.observation_space = spaces.Box(
            low=-10.0,
            high=10.0,
            shape=(obs_dim,),
            dtype=np.float32,
        )

        self.state: VehicleState = VehicleState()
        self.step_count: int = 0
        self.total_reward: float = 0.0
        self.last_waypoint_idx: int = 0

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)

        self.track.reset_clipping_zones()
        self.scorer.reset()
        self.step_count = 0
        self.total_reward = 0.0

        # Spawn vehicle along track
        spawn_idx = 0
        if options and "spawn_idx" in options:
            spawn_idx = options["spawn_idx"] % len(self.track.centerline)
        elif options and options.get("random_spawn", False):
            spawn_idx = self.np_random.integers(0, len(self.track.centerline))

        spawn_pos = self.track.centerline[spawn_idx].copy()
        next_idx = (spawn_idx + 2) % len(self.track.centerline)
        d_vec = self.track.centerline[next_idx] - spawn_pos
        spawn_yaw = float(np.arctan2(d_vec[1], d_vec[0]))

        # Add small random noise to spawn state
        noise_yaw = float(self.np_random.uniform(-0.1, 0.1))
        noise_speed = float(self.np_random.uniform(-1.0, 1.0))
        init_speed = max(4.0, self.track.spawn_speed + noise_speed)

        self.state = VehicleState(
            x=float(spawn_pos[0]),
            y=float(spawn_pos[1]),
            yaw=spawn_yaw + noise_yaw,
            vx=init_speed,
            vy=0.0,
            yaw_rate=0.0,
            steer=0.0,
        )
        self.last_waypoint_idx = spawn_idx

        obs = self._get_obs()
        info = self._get_info()
        return obs, info

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        self.step_count += 1

        steer_cmd = float(np.clip(action[0], -1.0, 1.0))
        throttle_cmd = float(np.clip(action[1], -1.0, 1.0))
        handbrake_cmd = float(np.clip(action[2], 0.0, 1.0))

        # Integrate dynamics
        self.state = self.dyn.step_rk4(
            self.state,
            steer=steer_cmd,
            throttle_brake=throttle_cmd,
            handbrake=handbrake_cmd,
            dt=self.dt,
        )

        corners = self.dyn.get_car_corners(self.state)
        collision = self.track.check_collision(corners)

        # Update scoring and tricks
        drift_reward, tricks, spinout = self.scorer.update(
            self.state, self.track, corners, dt=self.dt
        )

        # Progression along centerline
        curr_pos = np.array([self.state.x, self.state.y])
        curr_idx = self.track.find_nearest_waypoint_index(curr_pos)
        n_pts = len(self.track.centerline)
        waypoint_diff = (curr_idx - self.last_waypoint_idx) % n_pts
        if waypoint_diff > n_pts // 2:
            waypoint_diff -= n_pts  # Going backward
        self.last_waypoint_idx = curr_idx

        # Forward progression along track direction
        next_wp = self.track.centerline[(curr_idx + 1) % n_pts]
        curr_wp = self.track.centerline[curr_idx]
        track_dir = next_wp - curr_wp
        track_dir_norm = track_dir / max(np.linalg.norm(track_dir), 1e-6)

        car_vel_world = np.array([
            self.state.vx * np.cos(self.state.yaw) - self.state.vy * np.sin(self.state.yaw),
            self.state.vx * np.sin(self.state.yaw) + self.state.vy * np.cos(self.state.yaw),
        ])
        v_forward = float(np.dot(car_vel_world, track_dir_norm))

        # Reward formulation
        progression_reward = np.clip(v_forward * 0.08, -1.0, 2.0)
        trick_reward = sum(t.points * 0.003 * t.multiplier_bonus for t in tricks)

        step_reward = drift_reward + progression_reward + trick_reward

        # Penalties
        terminated = False
        truncated = False

        if collision:
            self.scorer.handle_collision()
            step_reward -= 60.0
            terminated = True
        elif spinout:
            step_reward -= 25.0
            # Terminate on spinout so agent quickly resets to learn recovery
            terminated = True

        if self.step_count >= self.max_steps:
            truncated = True

        self.total_reward += step_reward

        obs = self._get_obs()
        info = self._get_info()
        info["tricks"] = [t.name for t in tricks]
        info["collision"] = collision
        info["spinout"] = spinout

        return obs, step_reward, terminated, truncated, info

    def _get_obs(self) -> np.ndarray:
        s = self.state
        score_st = self.scorer.get_state()

        # 1. Vehicle dynamics (6 values)
        obs_veh = np.array([
            s.vx / 25.0,
            s.vy / 20.0,
            s.speed / 25.0,
            s.slip_angle / (0.5 * np.pi),
            s.yaw_rate / 4.0,
            s.steer / self.params.max_steer_angle,
        ], dtype=np.float32)

        # 2. Combo & Trick state (5 values)
        obs_combo = np.array([
            1.0 if score_st.is_drifting else 0.0,
            score_st.multiplier / 10.0,
            score_st.grace_timer / DriftScorer.GRACE_PERIOD,
            min(score_st.active_combo_score / 15000.0, 2.0),
            min(score_st.manji_chains / 5.0, 2.0),
        ], dtype=np.float32)

        # 3. Next 3 Waypoints relative to vehicle frame (6 values)
        car_pos = np.array([s.x, s.y])
        target_wps = self.track.get_target_waypoints(car_pos, count=3, step_stride=6)
        cos_y = np.cos(s.yaw)
        sin_y = np.sin(s.yaw)
        rot_to_body = np.array([[cos_y, sin_y], [-sin_y, cos_y]])

        wp_rel = []
        for wp in target_wps:
            delta = wp - car_pos
            delta_body = rot_to_body @ delta
            wp_rel.extend([delta_body[0] / 30.0, delta_body[1] / 30.0])
        obs_wps = np.array(wp_rel, dtype=np.float32)

        # 4. Heading error relative to track (2 values: sin, cos)
        curr_idx = self.track.find_nearest_waypoint_index(car_pos)
        next_idx = (curr_idx + 2) % len(self.track.centerline)
        track_tangent = self.track.centerline[next_idx] - self.track.centerline[curr_idx]
        track_yaw = np.arctan2(track_tangent[1], track_tangent[0])
        yaw_err = (track_yaw - s.yaw + np.pi) % (2 * np.pi) - np.pi
        obs_heading = np.array([np.sin(yaw_err), np.cos(yaw_err)], dtype=np.float32)

        # 5. LIDAR 16 rays around vehicle (16 values normalized [0, 1])
        lidar_dists = self.track.raycast_lidar(car_pos, s.yaw, num_rays=16, max_range=35.0)
        obs_lidar = (lidar_dists / 35.0).astype(np.float32)

        obs = np.concatenate([obs_veh, obs_combo, obs_wps, obs_heading, obs_lidar])
        return obs

    def _get_info(self) -> Dict[str, Any]:
        score_st = self.scorer.get_state()
        return {
            "score": score_st.total_score,
            "active_combo": score_st.active_combo_score,
            "multiplier": score_st.multiplier,
            "slip_angle_deg": self.state.slip_angle_deg,
            "speed": self.state.speed,
            "is_drifting": score_st.is_drifting,
            "manji_chains": score_st.manji_chains,
        }
