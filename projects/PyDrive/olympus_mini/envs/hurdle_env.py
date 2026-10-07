import os
import numpy as np
import mujoco
from olympus_mini.envs.base_athlete_env import BaseAthleteEnv

class HurdleEnv(BaseAthleteEnv):
    """
    Hurdle Traversal Environment.
    Reward high-speed clearance of hurdles while penalizing excessive CoM vertical height.
    """
    def __init__(self, frame_skip=10, max_steps=800):
        xml_path = os.path.join(os.path.dirname(__file__), "..", "models", "hurdle_track.xml")
        super().__init__(xml_path=xml_path, frame_skip=frame_skip, max_steps=max_steps, task_id=1)
        
        self.hurdle_x = np.arange(12, 90, 10, dtype=np.float32)
        self.cleared_hurdles = np.zeros(len(self.hurdle_x), dtype=bool)
        
        # Cache hurdle crossbar geom IDs for collision detection
        self.bar_geom_ids = []
        for i in range(len(self.hurdle_x)):
            gid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, f"hurdle_{i}_bar")
            if gid != -1:
                self.bar_geom_ids.append(gid)

    def reset(self, seed=None, options=None):
        self.cleared_hurdles.fill(False)
        return super().reset(seed=seed, options=options)

    def _check_hurdle_collision(self):
        """Check if athlete bodies collided with any hurdle crossbars."""
        for i in range(self.data.ncon):
            con = self.data.contact[i]
            g1, g2 = con.geom1, con.geom2
            if g1 in self.bar_geom_ids or g2 in self.bar_geom_ids:
                return True
        return False

    def _compute_reward_and_termination(self, action):
        qpos = self.data.qpos
        qvel = self.data.qvel
        
        pelvis_x = qpos[0]
        pelvis_y = qpos[1]
        pelvis_z = qpos[2]
        vx = qvel[0]
        
        # 1. Forward progression
        r_speed = 2.0 * vx
        
        # 2. Hurdle clearance bonus
        r_clear = 0.0
        for idx, hx in enumerate(self.hurdle_x):
            if not self.cleared_hurdles[idx] and pelvis_x > hx:
                self.cleared_hurdles[idx] = True
                r_clear += 12.0
                # Parabolic CoM flatness penalty: penalize jumping unnecessarily high over hurdle
                # Pro hurdlers graze hurdles; they do not launch into orbit!
                if pelvis_z > 1.30:
                    r_clear -= 4.0 * (pelvis_z - 1.30)
                    
        # 3. Collision with hurdle bar penalty
        r_collision = 0.0
        if self._check_hurdle_collision():
            r_collision = -4.0
            
        # 4. Upright & lane centering
        r_lane = -1.5 * (pelvis_y ** 2)
        r_upright = 1.0 * np.exp(-3.0 * (pelvis_z - 0.95)**2) if pelvis_z < 1.1 else 0.5
        
        # 5. Smoothness
        action_diff = action - self.prev_action
        r_smooth = -0.015 * np.sum(action_diff ** 2)
        
        # Fall check
        terminated = False
        if pelvis_z < 0.48 or abs(pelvis_y) > 1.8:
            terminated = True
            r_fall = -15.0
        else:
            r_fall = 0.0
            
        reward = r_speed + r_clear + r_collision + r_lane + r_upright + r_smooth + r_fall
        
        info = {
            "vx": vx,
            "dist": pelvis_x,
            "hurdles_cleared": int(np.sum(self.cleared_hurdles)),
            "r_speed": r_speed,
            "r_clear": r_clear,
            "step": self.step_count
        }
        return reward, terminated, False, info
