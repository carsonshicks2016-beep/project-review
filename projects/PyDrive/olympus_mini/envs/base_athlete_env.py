import os
import numpy as np
import mujoco
import gymnasium as gym
from gymnasium import spaces

class BaseAthleteEnv(gym.Env):
    """
    Base Gymnasium Environment for Olympus Mini 16-DOF Humanoid.
    Simulation decimation: 10 substeps of 0.002s = 50Hz control frequency (0.02s).
    """
    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(self, xml_path, frame_skip=10, max_steps=500, task_id=0):
        super().__init__()
        self.xml_path = xml_path
        self.frame_skip = frame_skip
        self.max_steps = max_steps
        self.task_id = task_id  # 0: Sprint, 1: Hurdles, 2: Vault
        
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        
        self.n_act = self.model.nu  # 16
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(self.n_act,), dtype=np.float32)
        
        # Foot contact geom IDs
        self.contact_geom_names = [
            "left_contact_toe_in", "left_contact_toe_out", "left_contact_heel_in", "left_contact_heel_out",
            "right_contact_toe_in", "right_contact_toe_out", "right_contact_heel_in", "right_contact_heel_out"
        ]
        self.contact_geom_ids = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, name) for name in self.contact_geom_names]
        
        # Pelvis & joint references
        self.pelvis_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
        self.init_qpos = self.model.qpos0.copy()
        
        # Observation dimension: 74 dims
        self.obs_dim = 74
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.obs_dim,), dtype=np.float32)
        
        self.step_count = 0
        self.phase = 0.0
        self.prev_action = np.zeros(self.n_act, dtype=np.float32)
        self.prev_prev_action = np.zeros(self.n_act, dtype=np.float32)
        self.renderer = None

    def _get_foot_contacts(self):
        """Query active collision contacts for the 8 foot contact points."""
        contacts = np.zeros(8, dtype=np.float32)
        for i in range(self.data.ncon):
            con = self.data.contact[i]
            g1, g2 = con.geom1, con.geom2
            for idx, cid in enumerate(self.contact_geom_ids):
                if g1 == cid or g2 == cid:
                    contacts[idx] = 1.0
        return contacts

    def _get_obs(self):
        qpos = self.data.qpos
        qvel = self.data.qvel
        
        pelvis_z = np.array([qpos[2]], dtype=np.float32)
        pelvis_quat = qpos[3:7].astype(np.float32)
        joint_pos = qpos[7:23].astype(np.float32)
        
        pelvis_linvel = qvel[0:3].astype(np.float32)
        pelvis_angvel = qvel[3:6].astype(np.float32)
        joint_vel = qvel[6:22].astype(np.float32)
        
        foot_contacts = self._get_foot_contacts()
        prev_act = self.prev_action.copy()
        
        phase_clock = np.array([np.sin(2 * np.pi * self.phase), np.cos(2 * np.pi * self.phase)], dtype=np.float32)
        task_onehot = np.zeros(3, dtype=np.float32)
        task_onehot[self.task_id] = 1.0
        
        pos_progress = np.array([qpos[0], qpos[1]], dtype=np.float32)
        
        obs = np.concatenate([
            pelvis_z, pelvis_quat, joint_pos,
            pelvis_linvel, pelvis_angvel, joint_vel,
            foot_contacts, prev_act, phase_clock,
            task_onehot, pos_progress
        ])
        return obs

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.step_count = 0
        self.phase = 0.0
        self.prev_action.fill(0.0)
        self.prev_prev_action.fill(0.0)
        
        mujoco.mj_resetData(self.model, self.data)
        
        qpos = self.init_qpos.copy()
        qpos[2] = 0.98  # pelvis z
        # Slight knee bend: left knee index 10, right knee index 16
        qpos[10] = -0.35
        qpos[16] = -0.35
        # Slight hip flexion: left hip pitch idx 9, right hip pitch idx 15
        qpos[9] = 0.25
        qpos[15] = 0.25
        
        # Small exploration jitter
        qpos[7:23] += np.random.uniform(-0.02, 0.02, size=16)
        
        self.data.qpos[:] = qpos
        self.data.qvel[:] = 0.0
        
        mujoco.mj_forward(self.model, self.data)
        return self._get_obs(), {}

    def step(self, action):
        action = np.clip(action, -1.0, 1.0)
        self.data.ctrl[:self.n_act] = action
        
        for _ in range(self.frame_skip):
            mujoco.mj_step(self.model, self.data)
            
        self.step_count += 1
        self.phase = (self.phase + 0.02 * 1.5) % 1.0
        
        obs = self._get_obs()
        reward, terminated, truncated, info = self._compute_reward_and_termination(action)
        
        self.prev_prev_action[:] = self.prev_action
        self.prev_action[:] = action
        
        if self.step_count >= self.max_steps:
            truncated = True
            
        return obs, reward, terminated, truncated, info

    def _compute_reward_and_termination(self, action):
        raise NotImplementedError

    def render(self, mode="rgb_array", width=640, height=480):
        if self.renderer is None:
            self.renderer = mujoco.Renderer(self.model, height=height, width=width)
        cam = mujoco.MjvCamera()
        cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        cam.trackbodyid = self.pelvis_body_id
        cam.distance = 3.5
        cam.elevation = -12.0
        cam.azimuth = 90.0
        
        self.renderer.update_scene(self.data, camera=cam)
        return self.renderer.render()

    def close(self):
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None
