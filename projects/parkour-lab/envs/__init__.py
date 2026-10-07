"""Environment factories. Custom terrain is introduced after the walking review."""
import gymnasium as gym
from stable_baselines3.common.monitor import Monitor
from envs.parkour_env import ParkourHumanoidEnv

def make_env(env_id):
    def create():
        return Monitor(gym.make(env_id))
    return create

def make_parkour_env(**kwargs):
    def create():
        return Monitor(ParkourHumanoidEnv(**kwargs))
    return create
