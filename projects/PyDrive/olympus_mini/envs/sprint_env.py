import os
import numpy as np
from olympus_mini.envs.base_athlete_env import BaseAthleteEnv

class SprintEnv(BaseAthleteEnv):
    """
    Upgraded 100m Sprint Environment.
    Addresses all major deep RL bipedal locomotion pathologies:
    1. Gated forward velocity eliminates the 'face-plant dive' exploitation.
    2. Knee flexion reward and hyperextension penalty eliminate the 'peg-leg' trap.
    3. Strict pitch/roll/yaw constraints eliminate terminal falling and drifting.
    4. Flight phase (airtime) bonus and touchdown reward enable true sprinting flight.
    """
    def __init__(self, frame_skip=10, max_steps=600, target_speed=6.0):
        xml_path = os.path.join(os.path.dirname(__file__), "..", "models", "athlete.xml")
        super().__init__(xml_path=xml_path, frame_skip=frame_skip, max_steps=max_steps, task_id=0)
        self.target_speed = target_speed
        self.was_airborne = False
        self.airborne_steps = 0

    def reset(self, seed=None, options=None):
        self.was_airborne = False
        self.airborne_steps = 0
        return super().reset(seed=seed, options=options)

    def _compute_reward_and_termination(self, action):
        qpos = self.data.qpos
        qvel = self.data.qvel
        
        pelvis_x = qpos[0]
        pelvis_y = qpos[1]
        pelvis_z = qpos[2]
        
        # Base quaternion [qw, qx, qy, qz]
        qw, qx, qy, qz = qpos[3], qpos[4], qpos[5], qpos[6]
        sin_pitch = 2.0 * (qw * qy - qz * qx)
        pitch = float(np.arcsin(np.clip(sin_pitch, -1.0, 1.0)))
        roll = float(np.arctan2(2.0 * (qw * qx + qy * qz), 1.0 - 2.0 * (qx**2 + qy**2)))
        
        vx = float(qvel[0])
        vy = float(qvel[1])
        vz = float(qvel[2])
        wz = float(qvel[5])  # yaw rate
        
        contacts = self._get_foot_contacts()
        total_contacts = int(np.sum(contacts))
        is_airborne = (total_contacts == 0)
        
        # -------------------------------------------------------------
        # 1. Orientation & Posture Gating (Fixes Face-Plant Dive)
        # -------------------------------------------------------------
        # Forward speed is ONLY rewarded when the torso is upright and above minimum height
        posture_factor = float(np.clip((pelvis_z - 0.65) / 0.28, 0.0, 1.0)) * max(0.0, np.cos(pitch)) * max(0.0, np.cos(roll))
        r_speed = 3.0 * vx * posture_factor
        
        # Height bonus centered at 0.95m
        r_height = 1.2 * np.exp(-5.0 * (pelvis_z - 0.95)**2)
        
        # Heading alignment reward (keeping pelvis facing down-track [1, 0, 0])
        heading_x = 1.0 - 2.0 * (qy**2 + qz**2)
        r_heading = 1.5 * heading_x
        
        # Lateral drift and yaw rate penalties
        r_lane = -2.5 * (pelvis_y ** 2) - 1.5 * abs(vy) - 0.5 * (wz ** 2)
        
        # -------------------------------------------------------------
        # 2. Knee Mechanics (Fixes Peg-Leg / Stiff-Knee Trap)
        # -------------------------------------------------------------
        lknee = float(qpos[10])  # left knee pitch [-2.44, 0.0]
        rknee = float(qpos[16])  # right knee pitch [-2.44, 0.0]
        
        # Penalize hyperextension / locked knees (closer than 8 degrees to straight)
        r_stiff = -0.8 * (max(0.0, lknee + 0.15) + max(0.0, rknee + 0.15))
        
        # Reward knee flexion on the active swing leg (based on phase clock)
        phase_sin = np.sin(2 * np.pi * self.phase)
        r_flexion = 0.0
        if phase_sin > 0.1:  # Left leg swing
            if lknee < -0.35:
                r_flexion += 0.4
        elif phase_sin < -0.1:  # Right leg swing
            if rknee < -0.35:
                r_flexion += 0.4
                
        # -------------------------------------------------------------
        # 3. Flight Phase & Touchdown (Fixes Flight Credit Assignment)
        # -------------------------------------------------------------
        r_flight = 0.0
        if is_airborne:
            self.airborne_steps += 1
            if vx > 1.8 and posture_factor > 0.6:
                r_flight += 1.2  # Reward dynamic flight phase
        else:
            # Touchdown transition bonus: reward landing cleanly after flight
            if self.was_airborne and self.airborne_steps >= 2 and posture_factor > 0.6:
                r_flight += 1.5
            self.airborne_steps = 0
            
        self.was_airborne = is_airborne
        
        # -------------------------------------------------------------
        # 4. Action Smoothness & Torque Regularization
        # -------------------------------------------------------------
        action_diff = action - self.prev_action
        r_smooth = -0.015 * float(np.sum(action_diff ** 2))
        
        # -------------------------------------------------------------
        # 5. Strict Termination Boundaries
        # -------------------------------------------------------------
        terminated = False
        r_fall = 0.0
        
        # Terminal triggers:
        # a) Pelvis drops too low (<0.55m)
        # b) Pitch forward > 34 deg or backward > 24 deg
        # c) Roll sideways > 28 deg
        # d) Lateral deviation out of lane (|y| > 1.6m)
        if pelvis_z < 0.55:
            terminated = True
            r_fall = -25.0
        elif pitch > np.radians(34) or pitch < np.radians(-24):
            terminated = True
            r_fall = -25.0
        elif abs(roll) > np.radians(28):
            terminated = True
            r_fall = -25.0
        elif abs(pelvis_y) > 1.6:
            terminated = True
            r_fall = -20.0
            
        # Sprint finish line (100m)
        if pelvis_x >= 100.0:
            terminated = True
            r_speed += 60.0
            
        reward = r_speed + r_height + r_heading + r_lane + r_stiff + r_flexion + r_flight + r_smooth + r_fall
        
        info = {
            "vx": vx,
            "dist": pelvis_x,
            "height": pelvis_z,
            "pitch": pitch,
            "roll": roll,
            "is_airborne": is_airborne,
            "posture": posture_factor,
            "r_speed": r_speed,
            "r_fall": r_fall,
            "step": self.step_count
        }
        return reward, terminated, False, info
