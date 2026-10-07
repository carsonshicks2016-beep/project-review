"""
CleanRL-style Vectorized PPO for Continuous Control Robotics in MuJoCo.
Designed for readability, transparency, and robotics/physics experimentation.
"""

import os
import sys
import time
import yaml
import argparse
from dataclasses import dataclass
from typing import Optional

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.normal import Normal
from torch.utils.tensorboard import SummaryWriter
import imageio

# Ensure repository root is in python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from humanoid_parkour.envs.wrappers import make_env, NormalizeObservation


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    """Orthogonal initialization for neural network layers."""
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    """
    Actor-Critic architecture for continuous robotics control.
    - Critic: estimates state-value V(s).
    - Actor: outputs Gaussian mean actions mu(s), with separate learnable log-stds.
    """
    def __init__(self, envs):
        super().__init__()
        obs_dim = np.array(envs.single_observation_space.shape).prod()
        act_dim = np.prod(envs.single_action_space.shape)

        # Critic network: 2-layer MLP with Tanh activations
        self.critic = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 1), std=1.0),
        )

        # Actor network: outputs mean action vectors
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, act_dim), std=0.01),
        )

        # State-independent learnable action standard deviation
        self.actor_logstd = nn.Parameter(torch.zeros(1, act_dim) - 0.5)

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None, deterministic=False):
        action_mean = self.actor_mean(x)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)

        if deterministic:
            action = action_mean
        elif action is None:
            action = probs.sample()

        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.critic(x)


def evaluate_policy(env_id, agent, device, num_episodes=3, record_video=True, video_path=None):
    """Runs deterministic evaluation episodes and optionally records video."""
    eval_env = gym.make(env_id, render_mode="rgb_array")
    eval_env = gym.wrappers.ClipAction(eval_env)

    returns = []
    lengths = []
    frames = []

    for ep in range(num_episodes):
        obs, info = eval_env.reset(seed=1000 + ep)
        done = False
        ep_return = 0.0
        ep_len = 0

        while not done:
            if record_video and ep == 0:
                frame = eval_env.render()
                frames.append(frame)

            with torch.no_grad():
                obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
                action, _, _, _ = agent.get_action_and_value(obs_t, deterministic=True)
                action = action.cpu().numpy()[0]

            obs, reward, terminated, truncated, info = eval_env.step(action)
            ep_return += reward
            ep_len += 1
            done = terminated or truncated

        returns.append(ep_return)
        lengths.append(ep_len)

    eval_env.close()

    # Save video if frames were captured
    if record_video and len(frames) > 0 and video_path:
        os.makedirs(os.path.dirname(video_path), exist_ok=True)
        imageio.mimsave(video_path, frames, fps=30)

    return float(np.mean(returns)), float(np.mean(lengths))


def train(config_path: str, override_timesteps: Optional[int] = None):
    # Load configuration
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    if override_timesteps:
        cfg["total_timesteps"] = override_timesteps

    run_name = f"{cfg['exp_name']}_{int(time.time())}"
    print(f"=== Starting PPO Training: {run_name} ===")
    print(f"Environment:       {cfg['env_id']}")
    print(f"Total Timesteps:   {cfg['total_timesteps']:,}")
    print(f"Parallel Envs:     {cfg['num_envs']}")
    print(f"Rollout Length:    {cfg['num_steps']}")
    print(f"Device:            {'cuda' if cfg['cuda'] and torch.cuda.is_available() else 'cpu'}")

    # Seed
    seed = cfg["seed"]
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = cfg["torch_deterministic"]

    device = torch.device("cuda" if cfg["cuda"] and torch.cuda.is_available() else "cpu")

    # Logging setup
    writer = SummaryWriter(os.path.join(cfg["run_dir"], run_name))
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in cfg.items()])),
    )

    os.makedirs(cfg["model_dir"], exist_ok=True)
    os.makedirs(cfg["video_dir"], exist_ok=True)

    # Vector environment setup
    envs = gym.vector.AsyncVectorEnv(
        [make_env(cfg["env_id"], i, False, run_name, cfg["gamma"]) for i in range(cfg["num_envs"])]
    )

    agent = Agent(envs).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=float(cfg["learning_rate"]), eps=1e-5)

    # Storage setup
    batch_size = int(cfg["num_envs"] * cfg["num_steps"])
    minibatch_size = int(batch_size // cfg["num_minibatches"])
    num_updates = cfg["total_timesteps"] // batch_size

    obs = torch.zeros((cfg["num_steps"], cfg["num_envs"]) + envs.single_observation_space.shape).to(device)
    actions = torch.zeros((cfg["num_steps"], cfg["num_envs"]) + envs.single_action_space.shape).to(device)
    logprobs = torch.zeros((cfg["num_steps"], cfg["num_envs"])).to(device)
    rewards = torch.zeros((cfg["num_steps"], cfg["num_envs"])).to(device)
    dones = torch.zeros((cfg["num_steps"], cfg["num_envs"])).to(device)
    values = torch.zeros((cfg["num_steps"], cfg["num_envs"])).to(device)

    # Global tracking
    global_step = 0
    start_time = time.time()
    next_obs, _ = envs.reset(seed=seed)
    next_obs = torch.as_tensor(next_obs, dtype=torch.float32).to(device)
    next_done = torch.zeros(cfg["num_envs"]).to(device)

    best_eval_return = -float("inf")

    for update in range(1, num_updates + 1):
        # Annealing learning rate
        if cfg["anneal_lr"]:
            frac = 1.0 - (update - 1.0) / num_updates
            lrnow = frac * float(cfg["learning_rate"])
            optimizer.param_groups[0]["lr"] = lrnow

        # Rollout collection
        for step in range(0, cfg["num_steps"]):
            global_step += cfg["num_envs"]
            obs[step] = next_obs
            dones[step] = next_done

            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs)
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            # Step environment
            next_obs_np, reward, terminations, truncations, infos = envs.step(action.cpu().numpy())
            next_done_np = np.logical_or(terminations, truncations)
            rewards[step] = torch.as_tensor(reward).to(device).view(-1)

            next_obs = torch.as_tensor(next_obs_np, dtype=torch.float32).to(device)
            next_done = torch.as_tensor(next_done_np, dtype=torch.float32).to(device)

            if "final_info" in infos:
                for item in infos["final_info"]:
                    if item and "episode" in item:
                        writer.add_scalar("charts/episodic_return", item["episode"]["r"], global_step)
                        writer.add_scalar("charts/episodic_length", item["episode"]["l"], global_step)

        # Bootstrap value if not done with Generalized Advantage Estimation (GAE)
        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(cfg["num_steps"])):
                if t == cfg["num_steps"] - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = rewards[t] + cfg["gamma"] * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = delta + cfg["gamma"] * cfg["gae_lambda"] * nextnonterminal * lastgaelam
            returns = advantages + values

        # Flatten batch
        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # Optimizing the policy and value network
        b_inds = np.arange(batch_size)
        clipfracs = []
        for epoch in range(cfg["update_epochs"]):
            np.random.shuffle(b_inds)
            for start in range(0, batch_size, minibatch_size):
                end = start + minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs[mb_inds], b_actions[mb_inds])
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > cfg["clip_coef"]).float().mean().item()]

                mb_advantages = b_advantages[mb_inds]
                if cfg["norm_adv"]:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - cfg["clip_coef"], 1 + cfg["clip_coef"])
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                if cfg["clip_vloss"]:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -cfg["clip_coef"],
                        cfg["clip_coef"],
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - cfg["ent_coef"] * entropy_loss + v_loss * cfg["vf_coef"]

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), cfg["max_grad_norm"])
                optimizer.step()

            if cfg["target_kl"] is not None and approx_kl > cfg["target_kl"]:
                break

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        sps = int(global_step / (time.time() - start_time))
        print(f"Update {update:3d}/{num_updates} | Step: {global_step:7,d} | Loss: {loss.item():.3f} | KL: {approx_kl:.4f} | EV: {explained_var:.3f} | SPS: {sps}")

        # TensorBoard logging
        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)
        writer.add_scalar("charts/SPS", sps, global_step)

        # Periodic Evaluation and Video Recording
        if update % cfg["eval_freq_iterations"] == 0 or update == num_updates:
            video_filename = os.path.join(cfg["video_dir"], f"{cfg['exp_name']}_step_{global_step:07d}.mp4")
            eval_ret, eval_len = evaluate_policy(
                cfg["env_id"],
                agent,
                device,
                num_episodes=cfg["eval_episodes"],
                record_video=True,
                video_path=video_filename
            )
            print(f"--> [Evaluation @ Step {global_step:,}] Return: {eval_ret:.2f} | Length: {eval_len:.1f} | Video: {video_filename}")
            writer.add_scalar("eval/mean_return", eval_ret, global_step)
            writer.add_scalar("eval/mean_length", eval_len, global_step)

            if eval_ret > best_eval_return:
                best_eval_return = eval_ret
                best_model_path = os.path.join(cfg["model_dir"], f"{cfg['exp_name']}_best.pt")
                torch.save(agent.state_dict(), best_model_path)
                print(f"--> [Checkpointed New Best Model] {best_model_path} (Return: {eval_ret:.2f})")

    envs.close()
    writer.close()
    print(f"Training Complete! Best Evaluation Return: {best_eval_return:.2f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CleanRL Continuous PPO for Robotics")
    parser.add_argument("--config", type=str, default="humanoid_parkour/configs/stage1_walker.yaml", help="Path to YAML config")
    parser.add_argument("--timesteps", type=int, default=None, help="Override total timesteps")
    args = parser.parse_args()

    train(args.config, args.timesteps)
