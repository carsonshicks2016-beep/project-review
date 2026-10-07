import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from olympus_mini.policy.network import ActorCritic

class PPO:
    """
    Proximal Policy Optimization with Symmetry Regularization and Action Smoothing.
    """
    def __init__(
        self,
        obs_dim=74,
        act_dim=16,
        lr=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_ratio=0.2,
        value_coef=0.5,
        entropy_coef=0.01,
        symmetry_coef=0.05,
        smoothness_coef=0.02,
        max_grad_norm=0.5,
        device="cpu"
    ):
        self.device = torch.device(device)
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_ratio = clip_ratio
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.symmetry_coef = symmetry_coef
        self.smoothness_coef = smoothness_coef
        self.max_grad_norm = max_grad_norm
        
        self.policy = ActorCritic(obs_dim=obs_dim, act_dim=act_dim).to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr, eps=1e-5)

    def select_action(self, obs, deterministic=False):
        with torch.no_grad():
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
            action, log_prob, value = self.policy.get_action(obs_t, deterministic=deterministic)
        return action.squeeze(0).cpu().numpy(), log_prob.item(), value.item()

    def update(self, rollouts, epochs=10, batch_size=64):
        obs = torch.as_tensor(rollouts["obs"], dtype=torch.float32, device=self.device)
        actions = torch.as_tensor(rollouts["actions"], dtype=torch.float32, device=self.device)
        old_log_probs = torch.as_tensor(rollouts["log_probs"], dtype=torch.float32, device=self.device)
        returns = torch.as_tensor(rollouts["returns"], dtype=torch.float32, device=self.device)
        advantages = torch.as_tensor(rollouts["advantages"], dtype=torch.float32, device=self.device)
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        dataset_size = obs.shape[0]
        indices = np.arange(dataset_size)
        
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        n_updates = 0
        
        for _ in range(epochs):
            np.random.shuffle(indices)
            for start_idx in range(0, dataset_size, batch_size):
                batch_idx = indices[start_idx : start_idx + batch_size]
                
                b_obs = obs[batch_idx]
                b_act = actions[batch_idx]
                b_old_logp = old_log_probs[batch_idx]
                b_ret = returns[batch_idx]
                b_adv = advantages[batch_idx]
                
                values, log_probs, entropy, mean_act = self.policy.evaluate_actions(b_obs, b_act)
                
                values = values.squeeze(-1)
                log_probs = log_probs.squeeze(-1)
                b_ret = b_ret.view(-1)
                b_adv = b_adv.view(-1)
                
                # 1. PPO Clipped Surrogate Loss
                ratio = torch.exp(log_probs - b_old_logp)
                surr1 = ratio * b_adv
                surr2 = torch.clamp(ratio, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio) * b_adv
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # 2. Value Loss with Smooth L1 (Huber) to stabilize against contact spikes
                value_loss = nn.functional.smooth_l1_loss(values, b_ret, beta=1.0)
                
                # 3. Action smoothness loss
                smooth_loss = torch.mean(torch.square(mean_act[:, 1:] - mean_act[:, :-1])) if mean_act.shape[1] > 1 else 0.0
                
                # 4. Mirror Symmetry Regularization (prevents asymmetric limping)
                b_obs_mirrored = self.policy.mirror_obs(b_obs)
                _, _, _, mean_act_mirrored = self.policy.evaluate_actions(b_obs_mirrored, b_act)
                target_mirrored_act = self.policy.mirror_action(mean_act)
                sym_loss = nn.functional.mse_loss(mean_act_mirrored, target_mirrored_act)
                
                # Total loss
                loss = (
                    policy_loss
                    + self.value_coef * value_loss
                    - self.entropy_coef * entropy.mean()
                    + self.symmetry_coef * sym_loss
                    + self.smoothness_coef * smooth_loss
                )
                
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.optimizer.step()
                
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.mean().item()
                n_updates += 1
                
        return {
            "policy_loss": total_policy_loss / max(1, n_updates),
            "value_loss": total_value_loss / max(1, n_updates),
            "entropy": total_entropy / max(1, n_updates)
        }

    def save_checkpoint(self, path):
        torch.save({
            "model_state_dict": self.policy.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict()
        }, path)

    def load_checkpoint(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.policy.load_state_dict(checkpoint["model_state_dict"], strict=False)
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
