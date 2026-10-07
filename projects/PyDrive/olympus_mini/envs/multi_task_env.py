import gymnasium as gym
import numpy as np
from olympus_mini.envs.sprint_env import SprintEnv
from olympus_mini.envs.hurdle_env import HurdleEnv
from olympus_mini.envs.vault_env import VaultEnv

class MultiTaskOlympusEnv(gym.Env):
    """
    Unified Multi-Discipline Olympus Athlete Environment.
    Dynamically switches between Sprint (0), Hurdles (1), and Vault (2).
    """
    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(self, task_mode="random", frame_skip=10, max_steps=600):
        super().__init__()
        self.task_mode = task_mode
        self.envs = {
            "sprint": SprintEnv(frame_skip=frame_skip, max_steps=max_steps),
            "hurdle": HurdleEnv(frame_skip=frame_skip, max_steps=max_steps),
            "vault": VaultEnv(frame_skip=frame_skip, max_steps=max_steps)
        }
        self.task_names = ["sprint", "hurdle", "vault"]
        self.current_task_name = "sprint"
        self.active_env = self.envs["sprint"]
        
        self.action_space = self.active_env.action_space
        self.observation_space = self.active_env.observation_space

    def reset(self, seed=None, options=None):
        if self.task_mode == "random":
            self.current_task_name = np.random.choice(self.task_names)
        elif self.task_mode in self.task_names:
            self.current_task_name = self.task_mode
        else:
            self.current_task_name = "sprint"
            
        self.active_env = self.envs[self.current_task_name]
        obs, info = self.active_env.reset(seed=seed, options=options)
        info["task_name"] = self.current_task_name
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.active_env.step(action)
        info["task_name"] = self.current_task_name
        return obs, reward, terminated, truncated, info

    def render(self, mode="rgb_array", width=640, height=480):
        return self.active_env.render(mode=mode, width=width, height=height)

    def close(self):
        for e in self.envs.values():
            e.close()
