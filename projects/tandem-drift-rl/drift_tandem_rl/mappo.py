"""
Multi-Agent Proximal Policy Optimization (MAPPO) with Centralized Training & Decentralized Execution (CTDE).

Architecture:
- Decentralized Actors: Pi_lead(a_L | o_L) and Pi_chase(a_C | o_C) condition solely on local ego observations.
- Centralized Critic: V_joint(s_global) conditions on joint observations [o_L, o_C, rel_features]
  to provide stable GAE baseline under non-stationary multi-agent interaction.
- Support for role alternation / self-play curriculum.
"""

from typing import Tuple, List, Dict, Optional
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.normal import Normal


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Actor(nn.Module):
    """Decentralized Actor network for continuous control."""
    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            layer_init(nn.Linear(obs_dim, hidden_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_dim, hidden_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_dim, action_dim), std=0.01),
        )
        self.log_std = nn.Parameter(torch.zeros(1, action_dim) - 0.5)

    def forward(self, obs: torch.Tensor) -> Normal:
        mean = self.net(obs)
        # Bounded steer [-1, 1], throttle [-1, 1], handbrake [0, 1]
        mean = torch.tanh(mean)
        std = self.log_std.exp().expand_as(mean)
        return Normal(mean, std)

    def get_action(self, obs: torch.Tensor, deterministic: bool = False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        dist = self.forward(obs)
        if deterministic:
            action = dist.mean
        else:
            action = dist.sample()
        log_prob = dist.log_prob(action).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        return action, log_prob, entropy


class CentralizedCritic(nn.Module):
    """
    Centralized Critic network V(s_global) estimating state values
    from the joint observations of both vehicles.
    """
    def __init__(self, global_state_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            layer_init(nn.Linear(global_state_dim, hidden_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_dim, hidden_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_dim, 1), std=1.0),
        )

    def forward(self, global_state: torch.Tensor) -> torch.Tensor:
        return self.net(global_state)


class MAPPO:
    """
    Multi-Agent PPO Trainer implementing CTDE for Leader and Chaser vehicles.
    """
    def __init__(
        self,
        obs_dim: int = 45,
        action_dim: int = 3,
        lr_actor: float = 3e-4,
        lr_critic: float = 1e-3,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_coef: float = 0.2,
        ent_coef: float = 0.01,
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        device: str = "cpu",
    ):
        self.device = torch.device(device)
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_coef = clip_coef
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm

        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.global_state_dim = obs_dim * 2

        # Lead and Chase Actors
        self.actor_lead = Actor(obs_dim, action_dim).to(self.device)
        self.actor_chase = Actor(obs_dim, action_dim).to(self.device)

        # Centralized Critics for Lead and Chase value functions
        self.critic_lead = CentralizedCritic(self.global_state_dim).to(self.device)
        self.critic_chase = CentralizedCritic(self.global_state_dim).to(self.device)

        # Optimizers
        self.opt_actor_lead = optim.Adam(self.actor_lead.parameters(), lr=lr_actor, eps=1e-5)
        self.opt_actor_chase = optim.Adam(self.actor_chase.parameters(), lr=lr_actor, eps=1e-5)
        self.opt_critic_lead = optim.Adam(self.critic_lead.parameters(), lr=lr_critic, eps=1e-5)
        self.opt_critic_chase = optim.Adam(self.critic_chase.parameters(), lr=lr_critic, eps=1e-5)

    def get_global_state(self, obs_lead: np.ndarray, obs_chase: np.ndarray) -> np.ndarray:
        """Concatenates lead and chase observations into joint global state."""
        return np.concatenate([obs_lead, obs_chase], axis=-1)

    def select_action(
        self,
        obs_lead: np.ndarray,
        obs_chase: np.ndarray,
        deterministic: bool = False,
    ) -> Dict[str, np.ndarray]:
        """Inference / interaction step."""
        with torch.no_grad():
            t_lead = torch.as_tensor(obs_lead, dtype=torch.float32, device=self.device)
            t_chase = torch.as_tensor(obs_chase, dtype=torch.float32, device=self.device)

            if t_lead.dim() == 1:
                t_lead = t_lead.unsqueeze(0)
                t_chase = t_chase.unsqueeze(0)

            act_lead, logp_lead, _ = self.actor_lead.get_action(t_lead, deterministic)
            act_chase, logp_chase, _ = self.actor_chase.get_action(t_chase, deterministic)

            global_s = torch.cat([t_lead, t_chase], dim=-1)
            v_lead = self.critic_lead(global_s)
            v_chase = self.critic_chase(global_s)

        return {
            "act_lead": act_lead.squeeze(0).cpu().numpy(),
            "act_chase": act_chase.squeeze(0).cpu().numpy(),
            "logp_lead": logp_lead.squeeze(0).cpu().numpy(),
            "logp_chase": logp_chase.squeeze(0).cpu().numpy(),
            "val_lead": v_lead.squeeze().cpu().numpy(),
            "val_chase": v_chase.squeeze().cpu().numpy(),
        }

    def compute_gae(
        self,
        rewards: np.ndarray,
        values: np.ndarray,
        dones: np.ndarray,
        next_value: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Computes Generalized Advantage Estimation (GAE-lambda) and returns."""
        steps = len(rewards)
        advantages = np.zeros(steps, dtype=np.float32)
        last_gae = 0.0

        for t in reversed(range(steps)):
            if t == steps - 1:
                next_non_terminal = 1.0 - float(dones[t])
                next_v = next_value
            else:
                next_non_terminal = 1.0 - float(dones[t])
                next_v = values[t + 1]

            delta = rewards[t] + self.gamma * next_v * next_non_terminal - values[t]
            advantages[t] = last_gae = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae

        returns = advantages + values
        return advantages, returns

    def train_step(
        self,
        buffer_lead: Dict[str, np.ndarray],
        buffer_chase: Dict[str, np.ndarray],
        buffer_global: Dict[str, np.ndarray],
        epochs: int = 4,
        batch_size: int = 64,
    ) -> Dict[str, float]:
        """Updates Lead and Chase Actor and Centralized Critic networks using PPO."""
        obs_lead = torch.as_tensor(buffer_lead["obs"], dtype=torch.float32, device=self.device)
        act_lead = torch.as_tensor(buffer_lead["actions"], dtype=torch.float32, device=self.device)
        old_logp_lead = torch.as_tensor(buffer_lead["log_probs"], dtype=torch.float32, device=self.device)
        adv_lead = torch.as_tensor(buffer_lead["advantages"], dtype=torch.float32, device=self.device)
        ret_lead = torch.as_tensor(buffer_lead["returns"], dtype=torch.float32, device=self.device)

        obs_chase = torch.as_tensor(buffer_chase["obs"], dtype=torch.float32, device=self.device)
        act_chase = torch.as_tensor(buffer_chase["actions"], dtype=torch.float32, device=self.device)
        old_logp_chase = torch.as_tensor(buffer_chase["log_probs"], dtype=torch.float32, device=self.device)
        adv_chase = torch.as_tensor(buffer_chase["advantages"], dtype=torch.float32, device=self.device)
        ret_chase = torch.as_tensor(buffer_chase["returns"], dtype=torch.float32, device=self.device)

        global_states = torch.as_tensor(buffer_global["states"], dtype=torch.float32, device=self.device)

        # Normalize advantages
        adv_lead = (adv_lead - adv_lead.mean()) / (adv_lead.std() + 1e-8)
        adv_chase = (adv_chase - adv_chase.mean()) / (adv_chase.std() + 1e-8)

        total_samples = len(obs_lead)
        metrics = {"loss_actor_lead": 0.0, "loss_actor_chase": 0.0, "loss_critic_lead": 0.0, "loss_critic_chase": 0.0}
        n_updates = 0

        for _ in range(epochs):
            indices = np.random.permutation(total_samples)
            for start in range(0, total_samples, batch_size):
                idx = indices[start:start + batch_size]
                n_updates += 1

                # ---------------- Leader Update ----------------
                dist_l = self.actor_lead(obs_lead[idx])
                new_logp_l = dist_l.log_prob(act_lead[idx]).sum(dim=-1)
                entropy_l = dist_l.entropy().sum(dim=-1).mean()

                ratio_l = torch.exp(new_logp_l - old_logp_lead[idx])
                surr1_l = ratio_l * adv_lead[idx]
                surr2_l = torch.clamp(ratio_l, 1.0 - self.clip_coef, 1.0 + self.clip_coef) * adv_lead[idx]
                actor_loss_l = -torch.min(surr1_l, surr2_l).mean() - self.ent_coef * entropy_l

                val_pred_l = self.critic_lead(global_states[idx]).squeeze(-1)
                critic_loss_l = 0.5 * ((val_pred_l - ret_lead[idx]) ** 2).mean()

                self.opt_actor_lead.zero_grad()
                actor_loss_l.backward()
                nn.utils.clip_grad_norm_(self.actor_lead.parameters(), self.max_grad_norm)
                self.opt_actor_lead.step()

                self.opt_critic_lead.zero_grad()
                critic_loss_l.backward()
                nn.utils.clip_grad_norm_(self.critic_lead.parameters(), self.max_grad_norm)
                self.opt_critic_lead.step()

                # ---------------- Chaser Update ----------------
                dist_c = self.actor_chase(obs_chase[idx])
                new_logp_c = dist_c.log_prob(act_chase[idx]).sum(dim=-1)
                entropy_c = dist_c.entropy().sum(dim=-1).mean()

                ratio_c = torch.exp(new_logp_c - old_logp_chase[idx])
                surr1_c = ratio_c * adv_chase[idx]
                surr2_c = torch.clamp(ratio_c, 1.0 - self.clip_coef, 1.0 + self.clip_coef) * adv_chase[idx]
                actor_loss_c = -torch.min(surr1_c, surr2_c).mean() - self.ent_coef * entropy_c

                val_pred_c = self.critic_chase(global_states[idx]).squeeze(-1)
                critic_loss_c = 0.5 * ((val_pred_c - ret_chase[idx]) ** 2).mean()

                self.opt_actor_chase.zero_grad()
                actor_loss_c.backward()
                nn.utils.clip_grad_norm_(self.actor_chase.parameters(), self.max_grad_norm)
                self.opt_actor_chase.step()

                self.opt_critic_chase.zero_grad()
                critic_loss_c.backward()
                nn.utils.clip_grad_norm_(self.critic_chase.parameters(), self.max_grad_norm)
                self.opt_critic_chase.step()

                metrics["loss_actor_lead"] += actor_loss_l.item()
                metrics["loss_actor_chase"] += actor_loss_c.item()
                metrics["loss_critic_lead"] += critic_loss_l.item()
                metrics["loss_critic_chase"] += critic_loss_c.item()

        for k in metrics:
            metrics[k] /= max(n_updates, 1)

        return metrics

    def save(self, checkpoint_path: str):
        """Save model checkpoints."""
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
        torch.save({
            "actor_lead": self.actor_lead.state_dict(),
            "actor_chase": self.actor_chase.state_dict(),
            "critic_lead": self.critic_lead.state_dict(),
            "critic_chase": self.critic_chase.state_dict(),
        }, checkpoint_path)
        print(f"Saved MAPPO checkpoint to {checkpoint_path}")

    def load(self, checkpoint_path: str):
        """Load model checkpoints."""
        data = torch.load(checkpoint_path, map_location=self.device)
        self.actor_lead.load_state_dict(data["actor_lead"])
        self.actor_chase.load_state_dict(data["actor_chase"])
        self.critic_lead.load_state_dict(data["critic_lead"])
        self.critic_chase.load_state_dict(data["critic_chase"])
        print(f"Loaded MAPPO checkpoint from {checkpoint_path}")
