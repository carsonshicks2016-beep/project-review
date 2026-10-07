import argparse
import os
import time
import numpy as np
import torch

from olympus_mini.envs import make_env
from olympus_mini.policy import PPO
from olympus_mini.viewer import EpisodeRecorder, LiveViewer

def run_demo():
    parser = argparse.ArgumentParser(description="Olympus Mini Athletic Simulation Demo")
    parser.add_argument("--task", type=str, default="sprint", choices=["sprint", "hurdle", "vault", "multi"], help="Discipline")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to checkpoint .pt")
    parser.add_argument("--render-gif", type=str, default=None, help="Save rollout to GIF path (e.g. demo.gif)")
    parser.add_argument("--render-mp4", type=str, default=None, help="Save rollout to MP4 path (e.g. demo.mp4)")
    parser.add_argument("--max-steps", type=int, default=200, help="Max steps to simulate")
    parser.add_argument("--live", action="store_true", help="Launch live Pygame window if display available")
    args = parser.parse_args()
    
    print(f"=== Olympus Mini Demo ===")
    print(f"Task: {args.task} | Max steps: {args.max_steps}")
    
    env = make_env(task=args.task, max_steps=args.max_steps)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    
    ppo = PPO(obs_dim=obs_dim, act_dim=act_dim, device="cpu")
    if args.checkpoint and os.path.exists(args.checkpoint):
        print(f"Loading policy from {args.checkpoint}...")
        ppo.load_checkpoint(args.checkpoint)
    else:
        print("Running nominal athletic exploration controller...")
        
    recorder = EpisodeRecorder(fps=30) if (args.render_gif or args.render_mp4) else None
    viewer = LiveViewer(title=f"Olympus Mini - {args.task.capitalize()}") if args.live else None
    if viewer:
        viewer.init_display()
        
    obs, info = env.reset()
    done = False
    step = 0
    total_reward = 0.0
    
    while not done and step < args.max_steps:
        # Heuristic / athletic stepping base if no trained checkpoint
        if args.checkpoint is None:
            # Generate rhythmic athletic swing based on phase clock in obs
            phase_sin = obs[62]
            action = np.zeros(act_dim, dtype=np.float32)
            # Torso forward lean
            action[0] = 0.25
            # Alternating hip pitch and arm swing
            action[2] = 0.5 * phase_sin   # left arm
            action[3] = -0.5 * phase_sin  # right arm
            action[6] = 0.4 * phase_sin   # left hip pitch
            action[12] = -0.4 * phase_sin # right hip pitch
            action[7] = -0.3 + 0.3 * max(0, -phase_sin) # left knee
            action[13] = -0.3 + 0.3 * max(0, phase_sin)  # right knee
        else:
            action, _, _ = ppo.select_action(obs, deterministic=True)
            
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        step += 1
        done = terminated or truncated
        
        # Offscreen render
        if recorder or viewer:
            rgb_frame = env.render(mode="rgb_array", width=640, height=480)
            telemetry = {
                "task": info.get("task_name", args.task),
                "vx": info.get("vx", 0.0),
                "dist": info.get("dist", 0.0),
                "height": info.get("height", 0.0),
                "step": step
            }
            if recorder:
                recorder.add_frame(rgb_frame, telemetry)
            if viewer:
                viewer.render_frame(rgb_frame)
                
    print(f"Rollout finished: Steps={step}, Total Reward={total_reward:.2f}")
    if "vx" in info:
        print(f"Final Speed: {info['vx']:.2f} m/s | Final Distance: {info['dist']:.2f} m")
        
    if recorder:
        if args.render_gif:
            recorder.save_gif(args.render_gif)
        if args.render_mp4:
            recorder.save_video(args.render_mp4)
            
    if viewer:
        viewer.close()
    env.close()

if __name__ == "__main__":
    run_demo()
