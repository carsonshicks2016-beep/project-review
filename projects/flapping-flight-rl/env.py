"""
Insect Flapping Flight Gymnasium Environment.

Operates at a 250 Hz policy control rate with 2500 Hz internal sub-stepped physics.
The agent modulates wing-tip pitch, stroke amplitude biases, and stroke-plane elevations
to achieve stable, precision hovering against gravity and rotational disturbances.
"""
from dataclasses import dataclass
import numpy as np

try:
    from .config import InsectSpec, SimConfig
    from .kinematics import KinematicEngine
    from .aerodynamics import AerodynamicSolver
    from .body import InsectBody
except (ImportError, ValueError):
    from config import InsectSpec, SimConfig
    from kinematics import KinematicEngine
    from aerodynamics import AerodynamicSolver
    from body import InsectBody

class InsectFlightEnv:
    def __init__(self, target_pos: np.ndarray = None):
        self.spec = InsectSpec()
        self.cfg = SimConfig()
        
        self.kinematics = KinematicEngine(self.spec)
        self.aero = AerodynamicSolver(self.spec)
        self.body = InsectBody(self.spec)

        self.target_pos = np.array(target_pos if target_pos is not None else [0.0, 0.0, 0.5], dtype=float)
        
        # Action space: 6 continuous outputs in [-1, 1]
        # [pitch_offset_L, pitch_offset_R, stroke_bias_L, stroke_bias_R, elev_offset_L, elev_offset_R]
        self.action_dim = 6
        # Observation space: 16 dims
        # [pos_err (3), vel (3), quat (4), omega (3), wing_phase (2), clap_active (1)]
        self.obs_dim = 16

        self.sim_time = 0.0
        self.step_count = 0
        self.max_steps = 1000 # 4.0 seconds of flight at 250 Hz

    def reset(self, seed: int = None) -> tuple[np.ndarray, dict]:
        if seed is not None:
            np.random.seed(seed)

        self.sim_time = 0.0
        self.step_count = 0
        
        # Initial slight state perturbation around target
        init_pos = self.target_pos + np.random.uniform(-0.02, 0.02, size=3)
        init_vel = np.random.uniform(-0.05, 0.05, size=3)
        init_omega = np.random.uniform(-0.1, 0.1, size=3)

        self.body.reset(pos=init_pos, vel=init_vel, omega=init_omega)
        self.kinematics.fling_boost_timer = 0.0
        self.kinematics.is_clapping = False

        obs = self._get_obs()
        info = {"target_pos": self.target_pos}
        return obs, info

    def _get_obs(self) -> np.ndarray:
        pos_err = self.body.pos - self.target_pos
        phase = 2.0 * np.pi * self.spec.nominal_frequency * self.sim_time
        phase_enc = np.array([np.cos(phase), np.sin(phase)])
        clap_enc = np.array([1.0 if self.kinematics.fling_boost_timer > 0 else 0.0])

        obs = np.concatenate([
            pos_err,
            self.body.vel,
            self.body.quat,
            self.body.omega,
            phase_enc,
            clap_enc
        ], dtype=np.float32)
        return obs

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        """
        Executes one 250 Hz control step (10 internal 2500 Hz physics sub-steps).
        """
        action = np.clip(action, -1.0, 1.0)
        self.step_count += 1

        # Scale action to physical units:
        # pitch offset +/- 25 degrees
        pitch_L = action[0] * np.radians(25.0)
        pitch_R = action[1] * np.radians(25.0)
        # stroke sweep bias +/- 15 degrees
        stroke_L = action[2] * np.radians(15.0)
        stroke_R = action[3] * np.radians(15.0)
        # elevation offset +/- 10 degrees
        elev_L = action[4] * np.radians(10.0)
        elev_R = action[5] * np.radians(10.0)

        dt = self.cfg.sim_dt
        cycle_lift = 0.0
        cycle_thrust = 0.0

        for _ in range(self.cfg.sub_steps):
            self.sim_time += dt

            # 1. Update wing kinematics
            left_wing, right_wing = self.kinematics.compute_wing_angles(
                t=self.sim_time,
                bias_phi_L=stroke_L, bias_phi_R=stroke_R,
                pitch_offset_L=pitch_L, pitch_offset_R=pitch_R,
                elevation_offset_L=elev_L, elevation_offset_R=elev_R
            )

            # 2. Clap-and-fling boost tracking
            if self.kinematics.fling_boost_timer > 0:
                fling_active = True
                self.kinematics.fling_boost_timer -= dt
            else:
                fling_active = False

            # 3. Aerodynamic forces on both wings
            F_L, M_L, tel_L = self.aero.compute_wing_wrench(left_wing, is_left=True, fling_active=fling_active)
            F_R, M_R, tel_R = self.aero.compute_wing_wrench(right_wing, is_left=False, fling_active=fling_active)

            cycle_lift += F_L[2] + F_R[2]
            cycle_thrust += F_L[0] + F_R[0]

            # 4. Integrate 6-DOF body dynamics
            self.body.step(F_L, M_L, F_R, M_R, dt)

        # Reward formulation for stable hover
        pos_err = np.linalg.norm(self.body.pos - self.target_pos)
        vel_err = np.linalg.norm(self.body.vel)
        rot_err = np.linalg.norm(self.body.omega)
        
        # Upright alignment (z axis of body aligned with world z)
        R = self.body.rotation_matrix()
        upright = R[2, 2] # dot product of body z with world z

        reward = (
            1.0                      # survival bonus
            - 20.0 * (pos_err ** 2)  # target position accuracy
            - 0.5 * (vel_err ** 2)   # velocity damping
            - 0.1 * (rot_err ** 2)   # angular rate stability
            + 2.0 * upright          # keep upright
            - 0.02 * np.sum(action ** 2) # control energy penalty
        )

        # Terminations: crashing into floor or drifting too far
        terminated = False
        if self.body.pos[2] <= 0.01 or pos_err > 0.8:
            terminated = True
            reward -= 10.0 # crash penalty

        truncated = (self.step_count >= self.max_steps)
        obs = self._get_obs()

        info = {
            "pos_error_m": pos_err,
            "mean_lift_mn": (cycle_lift / self.cfg.sub_steps) * 1e3,
            "upright_factor": upright,
            "sim_time_s": self.sim_time
        }

        return obs, reward, terminated, truncated, info
