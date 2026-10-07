"""Verify Humanoid observations, rewards, termination semantics and checkpoint replay."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv,VecNormalize
from eval import evaluate

def main():
    torch.set_num_threads(1)
    e=gym.make('Humanoid-v5');u=e.unwrapped
    obs,_=e.reset(seed=4)
    assert obs.shape==(348,) and e.action_space.shape==(17,)
    action=np.linspace(-.2,.2,17)
    _,reward,_,_,info=e.step(action)
    np.testing.assert_allclose(reward,sum(v for k,v in info.items() if k.startswith('reward_')))
    np.testing.assert_allclose(info['reward_ctrl'],-.1*np.square(action).sum())
    np.testing.assert_allclose(info['reward_forward'],1.25*info['x_velocity'])
    assert u.dt==.015
    before=u._get_obs().copy();qpos=u.data.qpos.copy();qvel=u.data.qvel.copy()
    qpos[0]+=5;qpos[1]-=3;u.set_state(qpos,qvel)
    # Root x/y are omitted from generalized-position observations.
    np.testing.assert_allclose(before[:22],u._get_obs()[:22])
    qpos[2]=.5;u.set_state(qpos,qvel)
    assert not u.is_healthy and u.healthy_reward==0
    e.close()
    if len(sys.argv)>1:
        checkpoint=Path(sys.argv[1])
        norm=VecNormalize.load(checkpoint/'vecnormalize.pkl',DummyVecEnv([lambda:gym.make('Humanoid-v5')]))
        norm.training=False;norm.norm_reward=False
        before=norm.obs_rms.count
        model=PPO.load(checkpoint/'policy.zip',device='cpu')
        a=evaluate(model,norm,'Humanoid-v5',[23456]);b=evaluate(model,norm,'Humanoid-v5',[23456])
        assert a==b and norm.obs_rms.count==before
        row=a['episodes'][0]
        np.testing.assert_allclose(row['reward'],sum(row['reward_terms'].values()))
        norm.close()
    print('PASS: Humanoid state, reward, health and available checkpoint checks')

if __name__=='__main__':main()
