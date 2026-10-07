import os
import numpy as np
import mujoco
from olympus_mini.envs.base_athlete_env import BaseAthleteEnv

class VaultEnv(BaseAthleteEnv):
    """
    Pole Vault Environment.
    Covers Runway approach -> Plant -> Elastic Pole Compression -> Inversion & Recoil -> Bar Clearance.
    """
    def __init__(self, frame_skip=10, max_steps=600):
        xml_path = os.path.join(os.path.dirname(__file__), "..", "models", "vault_track.xml")
        super().__init__(xml_path=xml_path, frame_skip=frame_skip, max_steps=max_steps, task_id=2)
        
        self.bar_cross_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "crossbar")
        self.landing_mat_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "mat_core")
        self.plant_target_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "plant_target")
        self.crossbar_cleared = False
        self.peak_z = 0.95

    def reset(self, seed=None, options=None):
        self.crossbar_cleared = False
        self.peak_z = 0.95
        return super().reset(seed=seed, options=options)

    def _compute_reward_and_termination(self, action):
        qpos = self.data.qpos
        qvel = self.data.qvel
        
        pelvis_x = qpos[0]
        pelvis_y = qpos[1]
        pelvis_z = qpos[2]
        vx = qvel[0]
        
        if pelvis_z > self.peak_z:
            self.peak_z = pelvis_z
            
        # Phase 1: Approach Runway (x < 17.0m)
        if pelvis_x < 17.0:
            r_phase = 2.5 * vx + 1.0 * np.exp(-3.0 * (pelvis_z - 0.95)**2)
        # Phase 2: Plant & Takeoff (17.0 <= x <= 19.0m)
        elif pelvis_x <= 19.0:
            # Reward vertical kinetic energy and height lift
            r_phase = 1.5 * vx + 3.0 * qvel[2] + 4.0 * (pelvis_z - 0.95)
        # Phase 3: Bar Clearance & Mat Landing (x > 19.0m)
        else:
            # Height apex clearance bonus
            r_phase = 5.0 * (pelvis_z - 1.0)
            if pelvis_z > 2.2 and not self.crossbar_cleared:
                self.crossbar_cleared = True
                r_phase += 25.0
                
        r_lane = -1.5 * (pelvis_y ** 2)
        action_diff = action - self.prev_action
        r_smooth = -0.015 * np.sum(action_diff ** 2)
        
        terminated = False
        # Termination conditions:
        # Before plant box: fall if pelvis drops below 0.45m
        if pelvis_x < 17.0 and (pelvis_z < 0.45 or abs(pelvis_y) > 1.8):
            terminated = True
            r_fall = -15.0
        # In landing zone (x > 21m), allow landing on the foam mat
        elif pelvis_x >= 21.0:
            terminated = True
            r_fall = 10.0 if self.crossbar_cleared else 2.0
        else:
            r_fall = 0.0
            
        reward = r_phase + r_lane + r_smooth + r_fall
        
        info = {
            "vx": vx,
            "dist": pelvis_x,
            "height": pelvis_z,
            "peak_z": self.peak_z,
            "crossbar_cleared": self.crossbar_cleared,
            "step": self.step_count
        }
        return reward, terminated, False, info
