"""
StableBaselines3 VecEnv Wrapper for Parameter-Shared MARL.

Wraps the PettingZoo ParallelEnv (NormanTrafficEnv) into a SB3 VecEnv.
Since there are 7 agents, this wrapper presents them to SB3 as 7 parallel
environments. This naturally implements Parameter Sharing (IPPO) where
a single PPO policy collects transitions from all 7 agents simultaneously
and learns a unified strategy.
"""

import numpy as np
from stable_baselines3.common.vec_env import VecEnv
from soonergrid.gym.env import NormanTrafficEnv


class MARLToVecEnv(VecEnv):
    """
    Converts a PettingZoo ParallelEnv into a SB3 VecEnv for parameter sharing.
    
    Each agent in the environment is treated as a separate "environment" in the
    vectorized batch. A single step of this VecEnv calls step() on the underlying
    multi-agent env and distributes the results.
    """

    def __init__(self, env: NormanTrafficEnv):
        self.env = env
        self.agents = env.possible_agents
        num_envs = len(self.agents)
        
        # Assume homogeneous spaces across all agents
        obs_space = env.observation_space(self.agents[0])
        act_space = env.action_space(self.agents[0])
        
        super().__init__(num_envs, obs_space, act_space)
        self._actions = {}

    def reset(self):
        obs_dict, infos = self.env.reset()
        # Stack observations in consistent agent order
        return np.stack([obs_dict.get(a, np.zeros(self.observation_space.shape)) for a in self.agents])

    def step_async(self, actions: np.ndarray):
        # Map batch of actions back to agent dict
        self._actions = {
            agent: actions[i] for i, agent in enumerate(self.agents)
        }

    def step_wait(self):
        obs_dict, rewards_dict, terms, truncs, infos = self.env.step(self._actions)
        
        # In this traffic env, episode ends for all agents simultaneously
        done = any(terms.values()) or any(truncs.values())
        
        obs = np.stack([obs_dict.get(a, np.zeros(self.observation_space.shape)) for a in self.agents])
        rewards = np.array([rewards_dict.get(a, 0.0) for a in self.agents], dtype=np.float32)
        dones = np.array([terms.get(a, done) or truncs.get(a, done) for a in self.agents], dtype=np.bool_)
        
        out_infos = []
        for i, a in enumerate(self.agents):
            info = infos.get(a, {}).copy()
            if dones[i]:
                # SB3 requirement: original terminal observation must be in info
                info["terminal_observation"] = obs[i]
            out_infos.append(info)
            
        if done:
            # Auto-reset environment when done
            obs_dict, _ = self.env.reset()
            obs = np.stack([obs_dict.get(a, np.zeros(self.observation_space.shape)) for a in self.agents])
            
        return obs, rewards, dones, out_infos

    def close(self):
        self.env.close()

    def get_attr(self, attr_name, indices=None):
        return [getattr(self.env, attr_name)] * self.num_envs

    def set_attr(self, attr_name, value, indices=None):
        pass

    def env_method(self, method_name, *method_args, indices=None, **method_kwargs):
        method = getattr(self.env, method_name)
        return [method(*method_args, **method_kwargs)] * self.num_envs

    def env_is_wrapped(self, wrapper_class, indices=None):
        return [False] * self.num_envs
