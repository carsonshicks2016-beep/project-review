"""
Sanity check script to verify MuJoCo, Gymnasium, PyTorch, and video rendering.
"""

import sys
import os
import platform
import gymnasium as gym
import torch
import mujoco
import imageio

def run_checks():
    print("=" * 60)
    print("HUMANOID PARKOUR // ENVIRONMENT & HARDWARE AUDIT")
    print("=" * 60)
    print(f"Platform:        {platform.platform()}")
    print(f"Architecture:    {platform.machine()}")
    print(f"Logical CPUs:    {os.cpu_count()}")
    print(f"PyTorch Version: {torch.__version__}")
    print(f"MPS Available:   {torch.backends.mps.is_available()}")
    print(f"Gymnasium Ver:   {gym.__version__}")
    print(f"MuJoCo Version:  {mujoco.__version__}")
    print("-" * 60)

    # Test environments
    test_envs = ["Walker2d-v5", "Hopper-v5", "Humanoid-v5"]
    for env_id in test_envs:
        try:
            env = gym.make(env_id, render_mode="rgb_array")
            obs, info = env.reset(seed=42)
            action = env.action_space.sample()
            next_obs, reward, terminated, truncated, info = env.step(action)
            frame = env.render()
            print(f"[{env_id}] OK - Obs shape: {obs.shape}, Action shape: {action.shape}, Render frame: {frame.shape}")
            env.close()
        except Exception as e:
            print(f"[{env_id}] FAILED: {e}")
            sys.exit(1)

    # Test video encoder
    test_video_path = "/tmp/test_render.mp4"
    try:
        writer = imageio.get_writer(test_video_path, fps=30)
        import numpy as np
        dummy_frame = np.zeros((240, 240, 3), dtype=np.uint8)
        for _ in range(15):
            writer.append_data(dummy_frame)
        writer.close()
        if os.path.exists(test_video_path):
            os.remove(test_video_path)
        print("[Video Encoding] OK - imageio_ffmpeg mp4 writer operational")
    except Exception as e:
        print(f"[Video Encoding] FAILED: {e}")
        sys.exit(1)

    print("=" * 60)
    print("ALL DEPENDENCIES AND SIMULATION CHECKS PASSED!")
    print("=" * 60)

if __name__ == "__main__":
    run_checks()
