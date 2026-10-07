"""
Environment utilities and wrappers for continuous control robotics.
"""

import gymnasium as gym
import numpy as np


class RunningMeanStd:
    """Tracks running mean and variance across batches."""
    def __init__(self, epsilon=1e-4, shape=()):
        self.mean = np.zeros(shape, "float64")
        self.var = np.ones(shape, "float64")
        self.count = epsilon

    def update(self, x):
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + np.square(delta) * self.count * batch_count / tot_count
        new_var = M2 / tot_count

        self.mean = new_mean
        self.var = new_var
        self.count = tot_count


class NormalizeObservation(gym.Wrapper):
    """Normalizes observations using running empirical mean and standard deviation."""
    def __init__(self, env, epsilon=1e-8, clip_obs=10.0):
        super().__init__(env)
        self.obs_rms = RunningMeanStd(shape=self.observation_space.shape)
        self.epsilon = epsilon
        self.clip_obs = clip_obs

    def step(self, action):
        obs, rews, terminated, truncated, infos = self.env.step(action)
        self.obs_rms.update(np.expand_dims(obs, 0))
        norm_obs = np.clip((obs - self.obs_rms.mean) / np.sqrt(self.obs_rms.var + self.epsilon), -self.clip_obs, self.clip_obs)
        return norm_obs.astype(np.float32), rews, terminated, truncated, infos

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.obs_rms.update(np.expand_dims(obs, 0))
        norm_obs = np.clip((obs - self.obs_rms.mean) / np.sqrt(self.obs_rms.var + self.epsilon), -self.clip_obs, self.clip_obs)
        return norm_obs.astype(np.float32), info


def make_env(env_id, idx, capture_video, run_name, gamma=0.99):
    def thunk():
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array")
        else:
            env = gym.make(env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = gym.wrappers.ClipAction(env)
        return env
    return thunk
