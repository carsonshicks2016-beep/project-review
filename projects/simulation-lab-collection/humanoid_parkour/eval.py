"""
Evaluation and video recording script for trained locomotion policies.
"""

import os
import sys
import argparse
import yaml
import gymnasium as gym
import numpy as np
import torch
import imageio

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from humanoid_parkour.train import Agent


def evaluate(
    checkpoint_path: str,
    env_id: str = "Walker2d-v5",
    num_episodes: int = 5,
    video_path: str = "humanoid_parkour/videos/eval_playback.mp4",
    deterministic: bool = True
):
    print("=" * 60)
    print(f"EVALUATING CHECKPOINT: {checkpoint_path}")
    print(f"Environment:          {env_id}")
    print(f"Episodes:             {num_episodes}")
    print(f"Video Output:         {video_path}")
    print("=" * 60)

    # Dummy env to build agent architecture
    dummy_env = gym.vector.SyncVectorEnv([lambda: gym.make(env_id)])
    agent = Agent(dummy_env)
    dummy_env.close()

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    state_dict = torch.load(checkpoint_path, map_location="cpu")
    agent.load_state_dict(state_dict)
    agent.eval()

    eval_env = gym.make(env_id, render_mode="rgb_array")
    eval_env = gym.wrappers.ClipAction(eval_env)

    returns = []
    lengths = []
    velocities = []
    frames = []

    for ep in range(num_episodes):
        obs, info = eval_env.reset(seed=2026 + ep)
        done = False
        ep_ret = 0.0
        ep_len = 0
        forward_vels = []

        while not done:
            if ep == 0:  # Record video for episode 0
                frames.append(eval_env.render())

            with torch.no_grad():
                obs_t = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
                action, _, _, _ = agent.get_action_and_value(obs_t, deterministic=deterministic)
                action = action.cpu().numpy()[0]

            obs, reward, terminated, truncated, info = eval_env.step(action)
            ep_ret += reward
            ep_len += 1
            if "x_velocity" in info:
                forward_vels.append(info["x_velocity"])

            done = terminated or truncated

        returns.append(ep_ret)
        lengths.append(ep_len)
        if forward_vels:
            velocities.append(np.mean(forward_vels))

        print(f"Episode {ep + 1:2d} | Return: {ep_ret:8.2f} | Length: {ep_len:4d} | Mean Fwd Vel: {np.mean(forward_vels) if forward_vels else 0.0:.2f} m/s")

    eval_env.close()

    if frames and video_path:
        os.makedirs(os.path.dirname(video_path), exist_ok=True)
        imageio.mimsave(video_path, frames, fps=30)
        print(f"\n[Saved Evaluation Video] {video_path} ({len(frames)} frames)")

    mean_ret = np.mean(returns)
    std_ret = np.std(returns)
    mean_len = np.mean(lengths)
    mean_vel = np.mean(velocities) if velocities else 0.0

    print("-" * 60)
    print(f"Mean Return:         {mean_ret:.2f} +/- {std_ret:.2f}")
    print(f"Mean Episode Length: {mean_len:.1f} steps")
    print(f"Mean Forward Speed:  {mean_vel:.2f} m/s")
    print("=" * 60)

    return mean_ret


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Trained Policy")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to .pt checkpoint")
    parser.add_argument("--env-id", type=str, default="Walker2d-v5", help="Gymnasium environment ID")
    parser.add_argument("--episodes", type=int, default=5, help="Number of evaluation episodes")
    parser.add_argument("--video", type=str, default="humanoid_parkour/videos/eval_playback.mp4", help="Output MP4 path")
    parser.add_argument("--stochastic", action="store_true", help="Sample actions stochastically instead of mean")

    args = parser.parse_args()
    evaluate(
        checkpoint_path=args.checkpoint,
        env_id=args.env_id,
        num_episodes=args.episodes,
        video_path=args.video,
        deterministic=not args.stochastic
    )
