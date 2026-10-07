import torch
import torch.nn as nn
from torch.distributions import Normal
import numpy as np

class ActorCritic(nn.Module):
    """
    Task-Conditioned Actor-Critic for Olympus Mini 16-DOF Humanoid.
    Includes explicit Left/Right mirror transformation for Gait Symmetry Regularization.
    """
    def __init__(self, obs_dim=74, act_dim=16, hidden_dim=256):
        super().__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        
        # Shared Biomechanical Feature Extractor
        self.backbone = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Tanh()
        )
        
        # Actor Head (Mean of continuous control distribution)
        self.actor_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, act_dim)
        )
        # Learnable log standard deviation
        self.log_std = nn.Parameter(torch.zeros(act_dim) - 0.5)
        
        # Critic Head (State Value Function)
        self.critic_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        # Build symmetry index mapping tensors
        self._init_symmetry_mappings()

    def _init_symmetry_mappings(self):
        """Construct index permutations and sign flips for left-right symmetry."""
        # Action layout (16):
        # 0: torso_pitch (+1)
        # 1: torso_yaw (-1)
        # 2: left_arm_pitch <-> 3: right_arm_pitch (+1)
        # 4..9: Left leg (yaw, roll, pitch, knee, ankle_pitch, ankle_roll)
        # 10..15: Right leg (yaw, roll, pitch, knee, ankle_pitch, ankle_roll)
        act_perm = [0, 1, 3, 2, 10, 11, 12, 13, 14, 15, 4, 5, 6, 7, 8, 9]
        act_signs = [1.0, -1.0, 1.0, 1.0, -1.0, -1.0, 1.0, 1.0, 1.0, -1.0, -1.0, -1.0, 1.0, 1.0, 1.0, -1.0]
        
        self.register_buffer("act_perm", torch.tensor(act_perm, dtype=torch.long))
        self.register_buffer("act_signs", torch.tensor(act_signs, dtype=torch.float32))

        # Observation layout (74):
        obs_perm = list(range(74))
        obs_signs = [1.0] * 74
        # Quat signs [qw, qx, qy, qz] -> roll/yaw flip: [1, -1, 1, -1]
        obs_signs[2] = -1.0
        obs_signs[4] = -1.0

        def map_joints(offset):
            obs_signs[offset + 1] = -1.0
            obs_perm[offset + 2] = offset + 3
            obs_perm[offset + 3] = offset + 2
            for i in range(6):
                obs_perm[offset + 4 + i] = offset + 10 + i
                obs_perm[offset + 10 + i] = offset + 4 + i
            for l_idx in [offset + 4, offset + 5, offset + 9, offset + 10, offset + 11, offset + 15]:
                obs_signs[l_idx] = -1.0

        map_joints(5)   # joint_pos (16)
        map_joints(27)  # joint_vel (16)
        map_joints(51)  # prev_action (16)

        # linvel: vy flips
        obs_signs[22] = -1.0
        # angvel: wx and wz flip
        obs_signs[24] = -1.0
        obs_signs[26] = -1.0

        # foot contacts (8): left 43..46 <-> right 47..50
        for i in range(4):
            obs_perm[43 + i] = 47 + i
            obs_perm[47 + i] = 43 + i

        # phase clock: sin/cos flip half-cycle
        obs_signs[67] = -1.0
        obs_signs[68] = -1.0
        # progress y flips
        obs_signs[73] = -1.0

        self.register_buffer("obs_perm", torch.tensor(obs_perm, dtype=torch.long))
        self.register_buffer("obs_signs", torch.tensor(obs_signs, dtype=torch.float32))

    def mirror_action(self, action):
        """Apply mirror transform to action tensor (swaps left/right and flips lateral signs)."""
        return action[:, self.act_perm] * self.act_signs

    def mirror_obs(self, obs):
        """Apply mirror transform to observation tensor."""
        return obs[:, self.obs_perm] * self.obs_signs

    def forward(self, obs):
        feat = self.backbone(obs)
        mean = self.actor_head(feat)
        value = self.critic_head(feat)
        return mean, value

    def get_action(self, obs, deterministic=False):
        feat = self.backbone(obs)
        mean = self.actor_head(feat)
        value = self.critic_head(feat)
        
        # Clamp log_std to [-1.6, 0.0] so exploration std stays in [0.20, 1.0] (prevents entropy collapse)
        std = torch.exp(self.log_std.clamp(-1.6, 0.0))
        dist = Normal(mean, std)
        
        if deterministic:
            action = mean
        else:
            action = dist.rsample()
            
        log_prob = dist.log_prob(action).sum(dim=-1, keepdim=True)
        return action, log_prob, value

    def evaluate_actions(self, obs, actions):
        feat = self.backbone(obs)
        mean = self.actor_head(feat)
        value = self.critic_head(feat)
        
        std = torch.exp(self.log_std.clamp(-2.0, 0.5))
        dist = Normal(mean, std)
        
        log_prob = dist.log_prob(actions).sum(dim=-1, keepdim=True)
        entropy = dist.entropy().sum(dim=-1, keepdim=True)
        return value, log_prob, entropy, mean
