"""Small integration check of the failure-prone reward/normalization boundary."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from eval import evaluate

def main():
    torch.set_num_threads(1)
    e=gym.make('Hopper-v5')
    obs,_=e.reset(seed=42)
    assert obs.shape == (11,) and e.action_space.shape == (3,)
    action=np.array([.2,-.1,.3])
    _,reward,_,_,info=e.step(action)
    np.testing.assert_allclose(reward,info['reward_forward']+info['reward_survive']+info['reward_ctrl'])
    np.testing.assert_allclose(info['reward_ctrl'],-.001*np.square(action).sum())
    e.close()
    checkpoint=Path(sys.argv[1])
    norm=VecNormalize.load(checkpoint/'vecnormalize.pkl',DummyVecEnv([lambda:gym.make('Hopper-v5')]))
    norm.training=False;norm.norm_reward=False
    before=norm.obs_rms.count
    model=PPO.load(checkpoint/'policy.zip',device='cpu')
    a=evaluate(model,norm,'Hopper-v5',[23456])
    b=evaluate(model,norm,'Hopper-v5',[23456])
    assert a==b, 'Deterministic checkpoint reload evaluation differs'
    assert norm.obs_rms.count==before, 'Evaluation changed observation statistics'
    norm.close()
    print('PASS: observation/action shapes, reward decomposition, deterministic evaluation, frozen normalization')

if __name__=='__main__':main()
