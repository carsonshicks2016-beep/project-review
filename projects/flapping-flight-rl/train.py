"""
PPO (Proximal Policy Optimization) Training Pipeline for Flapping Flight RL.

Trains a neural policy to output 250 Hz wing kinematics adjustments
(pitch, sweep bias, elevation) for stable precision hovering.
"""
import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal

try:
    from .env import InsectFlightEnv
except (ImportError, ValueError):
    from env import InsectFlightEnv

class ActorCritic(nn.Module):
    def __init__(self, obs_dim: int = 16, act_dim: int = 6, hidden_dim: int = 128):
        super().__init__()
        # Actor network
        self.actor_fc = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, act_dim)
        )
        self.log_std = nn.Parameter(torch.zeros(act_dim) - 0.5)

        # Critic network
        self.critic_fc = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        return self.actor_fc(x), self.critic_fc(x)

    def get_action(self, obs, deterministic=False):
        obs_t = torch.as_tensor(obs, dtype=torch.float32)
        mu = self.actor_fc(obs_t)
        std = torch.exp(self.log_std)
        dist = Normal(mu, std)

        if deterministic:
            action = mu
            log_prob = dist.log_prob(action).sum(dim=-1)
        else:
            action = dist.sample()
            log_prob = dist.log_prob(action).sum(dim=-1)

        val = self.critic_fc(obs_t)
        return action.detach().numpy(), log_prob.detach().numpy(), val.detach().numpy().squeeze()

    def evaluate_actions(self, obs_batch, act_batch):
        mu = self.actor_fc(obs_batch)
        std = torch.exp(self.log_std)
        dist = Normal(mu, std)
        log_probs = dist.log_prob(act_batch).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        values = self.critic_fc(obs_batch).squeeze(-1)
        return log_probs, values, entropy

def compute_gae(rewards, values, dones, gamma=0.99, lam=0.95):
    """Computes Generalized Advantage Estimators."""
    advantages = np.zeros_like(rewards)
    last_gae = 0.0
    for t in reversed(range(len(rewards))):
        next_val = values[t + 1] if t + 1 < len(values) else 0.0
        non_terminal = 1.0 - dones[t]
        delta = rewards[t] + gamma * next_val * non_terminal - values[t]
        advantages[t] = last_gae = delta + gamma * lam * non_terminal * last_gae
    returns = advantages + values
    return advantages, returns

def train(max_iterations: int = 100, steps_per_iter: int = 2048):
    os.makedirs("checkpoints", exist_ok=True)
    env = InsectFlightEnv()

    obs_dim = env.obs_dim
    act_dim = env.action_dim
    policy = ActorCritic(obs_dim, act_dim, hidden_dim=128)
    optimizer = optim.Adam(policy.parameters(), lr=3e-4)

    best_reward = -float("inf")
    print("=" * 65)
    print("STARTING PPO FLAPPING FLIGHT RL TRAINING (250 Hz Control Loop)")
    print(f"Observation space: {obs_dim} dims | Action space: {act_dim} dims")
    print(f"Steps per iteration: {steps_per_iter} | Max iterations: {max_iterations}")
    print("=" * 65)

    obs, _ = env.reset()

    for iteration in range(1, max_iterations + 1):
        t0 = time.time()
        obs_buf = []
        act_buf = []
        rew_buf = []
        val_buf = []
        logp_buf = []
        done_buf = []

        episode_rewards = []
        curr_ep_reward = 0.0

        for step in range(steps_per_iter):
            action, logp, val = policy.get_action(obs)
            next_obs, reward, terminated, truncated, info = env.step(action)

            obs_buf.append(obs)
            act_buf.append(action)
            rew_buf.append(reward)
            val_buf.append(val)
            logp_buf.append(logp)
            done_buf.append(float(terminated or truncated))

            curr_ep_reward += reward
            obs = next_obs

            if terminated or truncated:
                episode_rewards.append(curr_ep_reward)
                curr_ep_reward = 0.0
                obs, _ = env.reset()

        if len(episode_rewards) == 0:
            episode_rewards.append(curr_ep_reward)

        # GAE calculation
        obs_arr = np.array(obs_buf, dtype=np.float32)
        act_arr = np.array(act_buf, dtype=np.float32)
        rew_arr = np.array(rew_buf, dtype=np.float32)
        val_arr = np.array(val_buf, dtype=np.float32)
        logp_arr = np.array(logp_buf, dtype=np.float32)
        done_arr = np.array(done_buf, dtype=np.float32)

        adv_arr, ret_arr = compute_gae(rew_arr, val_arr, done_arr)
        adv_arr = (adv_arr - adv_arr.mean()) / (adv_arr.std() + 1e-8)

        # PyTorch tensors
        obs_tensor = torch.from_numpy(obs_arr)
        act_tensor = torch.from_numpy(act_arr)
        ret_tensor = torch.from_numpy(ret_arr)
        adv_tensor = torch.from_numpy(adv_arr)
        logp_old = torch.from_numpy(logp_arr)

        # PPO optimization epochs
        batch_size = 128
        dataset_size = len(obs_buf)
        indices = np.arange(dataset_size)

        for epoch in range(4):
            np.random.shuffle(indices)
            for start in range(0, dataset_size, batch_size):
                end = start + batch_size
                b_idx = indices[start:end]

                b_obs = obs_tensor[b_idx]
                b_act = act_tensor[b_idx]
                b_ret = ret_tensor[b_idx]
                b_adv = adv_tensor[b_idx]
                b_logp_old = logp_old[b_idx]

                new_logp, new_val, entropy = policy.evaluate_actions(b_obs, b_act)
                ratio = torch.exp(new_logp - b_logp_old)

                # Clipped surrogate objective
                surr1 = ratio * b_adv
                surr2 = torch.clamp(ratio, 0.8, 1.2) * b_adv
                policy_loss = -torch.min(surr1, surr2).mean()

                value_loss = 0.5 * ((new_val - b_ret) ** 2).mean()
                entropy_loss = -0.01 * entropy.mean()

                loss = policy_loss + value_loss + entropy_loss

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
                optimizer.step()

        elapsed = time.time() - t0
        mean_ep_rew = np.mean(episode_rewards)

        print(f"Iter {iteration:3d}/{max_iterations} | "
              f"Mean Ep Reward: {mean_ep_rew:7.1f} | "
              f"Episodes: {len(episode_rewards):2d} | "
              f"Loss: {loss.item():6.3f} | "
              f"Speed: {steps_per_iter/elapsed:5.1f} steps/s ({elapsed:.1f}s)")

        # Save latest checkpoint
        torch.save(policy.state_dict(), "checkpoints/hover_latest.pt")

        if mean_ep_rew > best_reward:
            best_reward = mean_ep_rew
            torch.save(policy.state_dict(), "checkpoints/hover_best.pt")
            print(f"  --> [NEW BEST] Checkpoint banked: {best_reward:.1f}")

    print("\n[COMPLETE] Training run finished. Best policy saved to checkpoints/hover_best.pt")

if __name__ == "__main__":
    train(max_iterations=10, steps_per_iter=1024)
