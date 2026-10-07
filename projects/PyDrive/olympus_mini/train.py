import argparse
import os
import time
import numpy as np
import torch

from olympus_mini.envs import make_env
from olympus_mini.policy import PPO

def collect_rollouts(env, ppo, num_steps=1024):
    """Collect experience transitions using the current policy."""
    obs_list, act_list, logp_list, rew_list, done_list, val_list = [], [], [], [], [], []
    
    obs, _ = env.reset()
    total_ep_reward = 0.0
    ep_rewards = []
    ep_lengths = []
    current_ep_len = 0
    
    for _ in range(num_steps):
        action, log_prob, value = ppo.select_action(obs)
        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        obs_list.append(obs)
        act_list.append(action)
        logp_list.append(log_prob)
        rew_list.append(reward)
        done_list.append(done)
        val_list.append(value)
        
        total_ep_reward += reward
        current_ep_len += 1
        
        if done:
            ep_rewards.append(total_ep_reward)
            ep_lengths.append(current_ep_len)
            total_ep_reward = 0.0
            current_ep_len = 0
            obs, _ = env.reset()
        else:
            obs = next_obs
            
    # Compute last value for GAE bootstrap
    _, _, last_val = ppo.select_action(obs)
    
    # Generalized Advantage Estimation (GAE)
    advantages = np.zeros(num_steps, dtype=np.float32)
    returns = np.zeros(num_steps, dtype=np.float32)
    last_gae = 0.0
    
    for t in reversed(range(num_steps)):
        if t == num_steps - 1:
            next_non_terminal = 1.0 - float(done_list[t])
            next_value = last_val
        else:
            next_non_terminal = 1.0 - float(done_list[t])
            next_value = val_list[t + 1]
            
        delta = rew_list[t] + ppo.gamma * next_value * next_non_terminal - val_list[t]
        advantages[t] = last_gae = delta + ppo.gamma * ppo.gae_lambda * next_non_terminal * last_gae
        returns[t] = advantages[t] + val_list[t]
        
    rollout_data = {
        "obs": np.array(obs_list, dtype=np.float32),
        "actions": np.array(act_list, dtype=np.float32),
        "log_probs": np.array(logp_list, dtype=np.float32),
        "returns": returns,
        "advantages": advantages
    }
    
    stats = {
        "mean_ep_reward": np.mean(ep_rewards) if len(ep_rewards) > 0 else total_ep_reward,
        "mean_ep_len": np.mean(ep_lengths) if len(ep_lengths) > 0 else current_ep_len,
        "episodes": len(ep_rewards)
    }
    return rollout_data, stats

def main():
    parser = argparse.ArgumentParser(description="Olympus Mini Training Pipeline")
    parser.add_argument("--task", type=str, default="sprint", choices=["sprint", "hurdle", "vault", "multi"], help="Discipline task")
    parser.add_argument("--total-steps", type=int, default=10000, help="Total environment timesteps")
    parser.add_argument("--rollout-steps", type=int, default=1024, help="Steps per PPO rollout collection")
    parser.add_argument("--batch-size", type=int, default=64, help="PPO mini-batch size")
    parser.add_argument("--epochs", type=int, default=8, help="PPO update epochs per rollout")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--save-dir", type=str, default="checkpoints", help="Directory to store model checkpoints")
    args = parser.parse_args()
    
    os.makedirs(args.save_dir, exist_ok=True)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"=== Olympus Mini Trainer ===")
    print(f"Task: {args.task} | Total Steps: {args.total_steps} | Device: {device}")
    
    env = make_env(task=args.task)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    
    ppo = PPO(obs_dim=obs_dim, act_dim=act_dim, lr=args.lr, device=device)
    
    step_count = 0
    iteration = 0
    best_reward = -float("inf")
    start_time = time.time()
    
    while step_count < args.total_steps:
        iteration += 1
        rollouts, stats = collect_rollouts(env, ppo, num_steps=args.rollout_steps)
        step_count += args.rollout_steps
        
        update_info = ppo.update(rollouts, epochs=args.epochs, batch_size=args.batch_size)
        fps = int(step_count / (time.time() - start_time))
        
        mean_rew = stats["mean_ep_reward"]
        print(f"Iter {iteration:03d} | Steps: {step_count:06d}/{args.total_steps} | FPS: {fps} | "
              f"EpReward: {mean_rew:.2f} | EpLen: {stats['mean_ep_len']:.1f} | "
              f"PiLoss: {update_info['policy_loss']:.4f} | ValLoss: {update_info['value_loss']:.4f}")
              
        if mean_rew > best_reward:
            best_reward = mean_rew
            ckpt_path = os.path.join(args.save_dir, f"champion_{args.task}.pt")
            ppo.save_checkpoint(ckpt_path)
            print(f"  --> New Champion saved: {ckpt_path} (Reward: {best_reward:.2f})")
            
    print(f"Training completed in {time.time() - start_time:.1f}s. Best reward: {best_reward:.2f}")

if __name__ == "__main__":
    main()
