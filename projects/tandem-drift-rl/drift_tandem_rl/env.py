"""
Multi-Agent Tandem Drift Gymnasium & PettingZoo Environment.
Supports:
1. Multi-Agent simultaneous step: both Leader and Chaser receive rewards and observations.
2. Single-Agent Gym wrapper: train Chaser against pretrained or scripted Leader.
3. Realistic wake turbulence cone, dirty air downforce loss, tire smoke grip reduction,
   and SAT collision impulses.
"""

from typing import Optional, Dict, Any, Tuple, List
import gymnasium as gym
from gymnasium import spaces
import numpy as np

from drift_tandem_rl.dynamics import VehicleParams, VehicleState, TandemDynamics
from drift_tandem_rl.tracks import Track, get_track
from drift_tandem_rl.scoring import TandemJudge, TandemBattleResult


class TandemMultiAgentEnv(gym.Env):
    """
    PettingZoo-style Multi-Agent Parallel Environment for Tandem Drifting.
    Agents: 'leader', 'chaser'.
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50}

    def __init__(
        self,
        track_name: str = "touge",
        params: Optional[VehicleParams] = None,
        max_steps: int = 1000,
        render_mode: Optional[str] = None,
        leader_policy: Optional[Any] = None,
    ):
        super().__init__()
        self.track_name = track_name
        self.track = get_track(track_name)
        self.params = params or VehicleParams()
        self.dyn = TandemDynamics(self.params)
        self.judge = TandemJudge()
        self.max_steps = max_steps
        self.render_mode = render_mode
        self.dt = 0.02  # 50 Hz physics
        self.leader_policy = leader_policy

        self.agents = ["leader", "chaser"]
        self.possible_agents = ["leader", "chaser"]

        # Action: [steer in [-1, 1], throttle_brake in [-1, 1], handbrake in [0, 1]]
        single_action_space = spaces.Box(
            low=np.array([-1.0, -1.0, 0.0], dtype=np.float32),
            high=np.array([1.0, 1.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )
        self.action_spaces = {agent: single_action_space for agent in self.agents}
        self.action_space = single_action_space  # Default for single-agent wrappers

        # Observation dimension: 45
        obs_dim = 6 + 4 + 6 + 2 + 16 + 8 + 3
        single_obs_space = spaces.Box(
            low=-15.0,
            high=15.0,
            shape=(obs_dim,),
            dtype=np.float32,
        )
        self.observation_spaces = {agent: single_obs_space for agent in self.agents}
        self.observation_space = single_obs_space

        self.leader_state: VehicleState = VehicleState()
        self.chaser_state: VehicleState = VehicleState()
        self.smoke_clouds: List[Tuple[float, float, float, float]] = [] # [(x, y, radius, density)]
        self.step_count: int = 0
        self.last_lead_waypoint_idx: int = 0
        self.last_chase_waypoint_idx: int = 0

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        super().reset(seed=seed)

        self.track.reset_clipping_zones()
        self.judge.reset()
        self.smoke_clouds.clear()
        self.step_count = 0

        # Determine spawn points along track
        n_pts = len(self.track.centerline)
        spawn_idx = 0
        if options and "spawn_idx" in options:
            spawn_idx = options["spawn_idx"] % n_pts
        elif options and options.get("random_spawn", False):
            spawn_idx = int(self.np_random.integers(0, n_pts))

        # Leader spawns ahead
        lead_pos = self.track.centerline[spawn_idx].copy()
        next_idx = (spawn_idx + 2) % n_pts
        d_vec = self.track.centerline[next_idx] - lead_pos
        lead_yaw = float(np.arctan2(d_vec[1], d_vec[0]))
        lead_speed = max(6.0, self.track.spawn_speed + float(self.np_random.uniform(-1.0, 1.0)))

        self.leader_state = VehicleState(
            x=float(lead_pos[0]),
            y=float(lead_pos[1]),
            yaw=lead_yaw + float(self.np_random.uniform(-0.05, 0.05)),
            vx=lead_speed,
            vy=0.0,
            yaw_rate=0.0,
            steer=0.0,
        )

        # Chaser spawns slightly behind the leader (bumper-to-bumper 2.5 - 4.5m gap)
        chase_spawn_dist = float(self.np_random.uniform(7.0, 9.0))
        chase_back_dir = -np.array([np.cos(lead_yaw), np.sin(lead_yaw)])
        chase_lat_offset = float(self.np_random.uniform(-0.6, 0.6))
        chase_norm_dir = np.array([-chase_back_dir[1], chase_back_dir[0]])

        chase_pos = lead_pos + chase_back_dir * chase_spawn_dist + chase_norm_dir * chase_lat_offset
        chase_speed = lead_speed * float(self.np_random.uniform(0.95, 1.02))

        self.chaser_state = VehicleState(
            x=float(chase_pos[0]),
            y=float(chase_pos[1]),
            yaw=lead_yaw + float(self.np_random.uniform(-0.05, 0.05)),
            vx=chase_speed,
            vy=0.0,
            yaw_rate=0.0,
            steer=0.0,
        )

        self.last_lead_waypoint_idx = spawn_idx
        self.last_chase_waypoint_idx = self.track.find_nearest_waypoint_index(chase_pos)

        observations = {
            "leader": self._get_agent_obs("leader"),
            "chaser": self._get_agent_obs("chaser"),
        }
        infos = {
            "leader": {},
            "chaser": {},
        }
        return observations, infos

    def step(
        self,
        actions: Dict[str, np.ndarray],
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, float], Dict[str, bool], Dict[str, bool], Dict[str, Any]]:
        self.step_count += 1

        # Leader action
        lead_act = actions.get("leader", np.zeros(3, dtype=np.float32))
        chase_act = actions.get("chaser", np.zeros(3, dtype=np.float32))

        # 1. Aerodynamic wake interaction
        wake_intensity, buffet_fy, buffet_torque = self.dyn.compute_wake_interaction(
            leader=self.leader_state,
            follower=self.chaser_state,
        )

        # 2. Tire smoke evolution and friction loss
        self._update_smoke_plumes()
        lead_friction = self.dyn.compute_smoke_friction_loss(self.leader_state, self.smoke_clouds)
        chase_friction = self.dyn.compute_smoke_friction_loss(self.chaser_state, self.smoke_clouds)

        # 3. Physics integration
        # Leader is ahead, so zero wake buffeting
        self.leader_state = self.dyn.step_rk4(
            self.leader_state,
            steer=float(np.clip(lead_act[0], -1.0, 1.0)),
            throttle_brake=float(np.clip(lead_act[1], -1.0, 1.0)),
            handbrake=float(np.clip(lead_act[2], 0.0, 1.0)),
            wake_intensity=0.0,
            buffet_fy=0.0,
            buffet_torque=0.0,
            friction_scale=lead_friction,
            dt=self.dt,
        )

        # Chaser suffers wake turbulence and dirty air downforce loss
        self.chaser_state = self.dyn.step_rk4(
            self.chaser_state,
            steer=float(np.clip(chase_act[0], -1.0, 1.0)),
            throttle_brake=float(np.clip(chase_act[1], -1.0, 1.0)),
            handbrake=float(np.clip(chase_act[2], 0.0, 1.0)),
            wake_intensity=wake_intensity,
            buffet_fy=buffet_fy,
            buffet_torque=buffet_torque,
            friction_scale=chase_friction,
            dt=self.dt,
        )

        # Emit smoke if slipping hard
        self._emit_tire_smoke(self.leader_state)
        self._emit_tire_smoke(self.chaser_state)

        # 4. Collision checking
        lead_corners = self.dyn.get_car_corners(self.leader_state)
        chase_corners = self.dyn.get_car_corners(self.chaser_state)

        # Mutual car-to-car collision
        car_colliding, penetration, col_normal = self.dyn.check_car_collision_sat(lead_corners, chase_corners)
        if car_colliding:
            self.leader_state, self.chaser_state = self.dyn.resolve_elastic_collision(
                self.leader_state, self.chaser_state, col_normal, penetration
            )

        # Wall collisions
        lead_wall_crash = self.track.check_collision(lead_corners)
        chase_wall_crash = self.track.check_collision(chase_corners)

        # 5. Judging and reward computation
        r_lead, r_chase, judge_info = self.judge.evaluate_step(
            leader=self.leader_state,
            chaser=self.chaser_state,
            track=self.track,
            lead_corners=lead_corners,
            chase_corners=chase_corners,
            is_collision=car_colliding,
            collision_penetration=penetration,
            dt=self.dt,
        )

        # Track progression reward
        lead_prog = self._compute_progression(self.leader_state, is_leader=True)
        chase_prog = self._compute_progression(self.chaser_state, is_leader=False)
        r_lead += lead_prog
        r_chase += chase_prog

        # Termination conditions
        lead_term = False
        chase_term = False

        if lead_wall_crash:
            r_lead -= 50.0
            lead_term = True

        if chase_wall_crash:
            r_chase -= 50.0
            chase_term = True

        if car_colliding and penetration > 0.35:
            # Severe at-fault collision terminates the run
            lead_term = True
            chase_term = True

        # Spinout check (stopped or backwards)
        if self.leader_state.speed < 1.0 and abs(self.leader_state.vx) < 0.5 and self.step_count > 40:
            lead_term = True
            r_lead -= 20.0
        if self.chaser_state.speed < 1.0 and abs(self.chaser_state.vx) < 0.5 and self.step_count > 40:
            chase_term = True
            r_chase -= 20.0

        truncated = self.step_count >= self.max_steps
        terminated_any = lead_term or chase_term

        rewards = {"leader": float(r_lead), "chaser": float(r_chase)}
        terminations = {"leader": bool(terminated_any), "chaser": bool(terminated_any)}
        truncations = {"leader": bool(truncated), "chaser": bool(truncated)}

        observations = {
            "leader": self._get_agent_obs("leader"),
            "chaser": self._get_agent_obs("chaser"),
        }

        infos = {
            "leader": {
                "score": self.judge.lead_score.total_score,
                "wall_crash": lead_wall_crash,
                **judge_info,
            },
            "chaser": {
                "score": self.judge.chase_score.total_score,
                "wall_crash": chase_wall_crash,
                **judge_info,
            },
        }

        return observations, rewards, terminations, truncations, infos

    def _get_agent_obs(self, role: str) -> np.ndarray:
        ego = self.leader_state if role == "leader" else self.chaser_state
        opp = self.chaser_state if role == "leader" else self.leader_state
        role_sign = 1.0 if role == "leader" else -1.0

        # 1. Ego vehicle dynamics (6)
        obs_veh = [
            ego.vx / 25.0,
            ego.vy / 20.0,
            ego.speed / 25.0,
            ego.slip_angle / (0.5 * np.pi),
            ego.yaw_rate / 4.0,
            ego.steer / self.params.max_steer_angle,
        ]

        # 2. Ego drift combo state (4)
        score_state = self.judge.lead_score if role == "leader" else self.judge.chase_score
        obs_score = [
            1.0 if score_state.is_drifting else 0.0,
            min(score_state.drift_duration / 5.0, 2.0),
            1.0 if score_state.in_proximity_zone else 0.0,
            role_sign,
        ]

        # 3. Target waypoints in body coordinates (6)
        ego_pos = np.array([ego.x, ego.y])
        targets_world = self.track.get_target_waypoints(ego_pos, count=3, step_stride=6)
        cos_y = np.cos(ego.yaw)
        sin_y = np.sin(ego.yaw)

        obs_wp = []
        for tp in targets_world:
            rel = tp - ego_pos
            bx = (rel[0] * cos_y + rel[1] * sin_y) / 30.0
            by = (-rel[0] * sin_y + rel[1] * cos_y) / 30.0
            obs_wp.extend([bx, by])

        # 4. Heading error relative to track direction (2)
        n_pts = len(self.track.centerline)
        curr_idx = self.track.find_nearest_waypoint_index(ego_pos)
        next_wp = self.track.centerline[(curr_idx + 1) % n_pts]
        track_tangent = next_wp - self.track.centerline[curr_idx]
        track_yaw = float(np.arctan2(track_tangent[1], track_tangent[0]))
        yaw_err = (ego.yaw - track_yaw + np.pi) % (2.0 * np.pi) - np.pi
        obs_track = [np.cos(yaw_err), np.sin(yaw_err)]

        # 5. LIDAR barrier rays (16)
        lidar_dists = self.track.raycast_lidar(ego_pos, ego.yaw, num_rays=16, max_range=35.0)
        obs_lidar = (lidar_dists / 35.0).tolist()

        # 6. Opponent relative radar perception in ego body frame (8)
        rel_world = np.array([opp.x - ego.x, opp.y - ego.y])
        rel_bx = (rel_world[0] * cos_y + rel_world[1] * sin_y) / 30.0
        rel_by = (-rel_world[0] * sin_y + rel_world[1] * cos_y) / 30.0
        rel_dist = float(np.hypot(rel_world[0], rel_world[1])) / 35.0

        v_opp_w = opp.velocity_world
        v_ego_w = ego.velocity_world
        v_rel_w = v_opp_w - v_ego_w
        rel_bvx = (v_rel_w[0] * cos_y + v_rel_w[1] * sin_y) / 25.0
        rel_bvy = (-v_rel_w[0] * sin_y + v_rel_w[1] * cos_y) / 25.0

        delta_yaw = (opp.yaw - ego.yaw + np.pi) % (2.0 * np.pi) - np.pi
        delta_slip = (opp.slip_angle - ego.slip_angle) / (0.5 * np.pi)

        obs_opp = [
            rel_bx,
            rel_by,
            rel_dist,
            rel_bvx,
            rel_bvy,
            np.cos(delta_yaw),
            np.sin(delta_yaw),
            delta_slip,
        ]

        # 7. Aerodynamic wake & smoke sensor (3)
        obs_wake = [
            float(ego.in_wake),
            float(ego.smoke_exposure),
            1.0 if ego.in_wake > 0.05 else 0.0,
        ]

        full_obs = obs_veh + obs_score + obs_wp + obs_track + obs_lidar + obs_opp + obs_wake
        return np.array(full_obs, dtype=np.float32)

    def _compute_progression(self, car: VehicleState, is_leader: bool) -> float:
        curr_pos = np.array([car.x, car.y])
        curr_idx = self.track.find_nearest_waypoint_index(curr_pos)
        n_pts = len(self.track.centerline)

        last_idx = self.last_lead_waypoint_idx if is_leader else self.last_chase_waypoint_idx
        diff = (curr_idx - last_idx) % n_pts
        if diff > n_pts // 2:
            diff -= n_pts

        if is_leader:
            self.last_lead_waypoint_idx = curr_idx
        else:
            self.last_chase_waypoint_idx = curr_idx

        # Forward velocity along track
        next_wp = self.track.centerline[(curr_idx + 1) % n_pts]
        track_dir = next_wp - self.track.centerline[curr_idx]
        norm = np.linalg.norm(track_dir)
        if norm > 1e-5:
            track_dir = track_dir / norm
            v_w = car.velocity_world
            v_forward = float(np.dot(v_w, track_dir))
            return float(np.clip(v_forward * 0.06, -0.5, 1.5))
        return 0.0

    def _emit_tire_smoke(self, car: VehicleState):
        if car.speed > 5.0 and abs(car.slip_angle_deg) > 14.0:
            # Emit smoke puff near rear axle
            cos_y = np.cos(car.yaw)
            sin_y = np.sin(car.yaw)
            rear_x = car.x - self.params.lr * cos_y
            rear_y = car.y - self.params.lr * sin_y
            self.smoke_clouds.append((rear_x, rear_y, 4.0, 0.9))

    def _update_smoke_plumes(self):
        updated = []
        for x, y, rad, dens in self.smoke_clouds:
            dens_new = dens - 0.015  # Dissipation
            rad_new = rad + 0.05     # Expansion
            if dens_new > 0.05:
                updated.append((x, y, rad_new, dens_new))
        self.smoke_clouds = updated[-40:] # Keep recent 40 smoke clouds for performance


class TandemChaserGymEnv(gym.Env):
    """
    Gymnasium single-agent environment for training a CHASER drift agent
    to stalk and match a LEADER car under wake turbulence and dirty air.
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50}

    def __init__(
        self,
        track_name: str = "touge",
        leader_policy: Optional[Any] = None,
        max_steps: int = 1000,
        render_mode: Optional[str] = None,
    ):
        super().__init__()
        self.ma_env = TandemMultiAgentEnv(
            track_name=track_name,
            max_steps=max_steps,
            render_mode=render_mode,
        )
        self.leader_policy = leader_policy
        self.action_space = self.ma_env.action_space
        self.observation_space = self.ma_env.observation_space
        self.render_mode = render_mode

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None):
        super().reset(seed=seed)
        obs_dict, info_dict = self.ma_env.reset(seed=seed, options=options)
        self.latest_lead_obs = obs_dict["leader"]
        return obs_dict["chaser"], info_dict["chaser"]

    def step(self, chase_action: np.ndarray):
        # Determine leader action
        if self.leader_policy is not None:
            # Policy provided (e.g. pretrained drift PPO)
            lead_act, _ = self.leader_policy.predict(self.latest_lead_obs, deterministic=False)
        else:
            # Robust expert driver / autopilot line
            lead_act = self._get_expert_lead_action(self.ma_env.leader_state)

        actions = {
            "leader": np.array(lead_act, dtype=np.float32),
            "chaser": np.array(chase_action, dtype=np.float32),
        }

        obs_dict, rew_dict, term_dict, trunc_dict, info_dict = self.ma_env.step(actions)
        self.latest_lead_obs = obs_dict["leader"]

        obs = obs_dict["chaser"]
        reward = rew_dict["chaser"]
        terminated = term_dict["chaser"]
        truncated = trunc_dict["chaser"]
        info = info_dict["chaser"]

        return obs, reward, terminated, truncated, info

    def _get_expert_lead_action(self, state: VehicleState) -> np.ndarray:
        """
        Pure pursuit / sliding-mode controller that drives an aggressive drift line.
        """
        track = self.ma_env.track
        curr_pos = np.array([state.x, state.y])
        targets = track.get_target_waypoints(curr_pos, count=2, step_stride=7)
        target = targets[0]

        rel = target - curr_pos
        cos_y = np.cos(state.yaw)
        sin_y = np.sin(state.yaw)
        bx = rel[0] * cos_y + rel[1] * sin_y
        by = -rel[0] * sin_y + rel[1] * cos_y

        steer_cmd = np.clip(by * 0.12, -1.0, 1.0)
        target_speed = 13.5
        throttle_cmd = 0.85 if state.speed < target_speed else 0.25

        # Handbrake flick on sharp entries
        handbrake_cmd = 0.0
        if abs(by) > 4.5 and state.speed > 10.0 and abs(state.slip_angle_deg) < 15.0:
            handbrake_cmd = 0.85
            throttle_cmd = 1.0

        return np.array([steer_cmd, throttle_cmd, handbrake_cmd], dtype=np.float32)
