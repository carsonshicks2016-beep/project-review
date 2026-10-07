"""
Training pipeline for Multi-Agent Tandem Drift Battles using MAPPO.
Performs Centralized Training with Decentralized Execution (CTDE)
with real-time telemetry, role alternation, and checkpointing.
"""

import argparse
import os
import sys
import time

# Ensure repository root is on sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from drift_tandem_rl.env import TandemMultiAgentEnv
from drift_tandem_rl.mappo import MAPPO


def train_mappo(
    total_timesteps: int = 50000,
    rollout_steps: int = 512,
    epochs: int = 5,
    batch_size: int = 64,
    lr_actor: float = 3e-4,
    lr_critic: float = 1e-3,
    track_name: str = "touge",
    save_dir: str = "models",
    eval_freq: int = 5,
):
    os.makedirs(save_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"--- Training MAPPO Tandem Drift on {device.upper()} ---")
    print(f"Track: {track_name} | Total Steps: {total_timesteps:,} | Rollout: {rollout_steps}")

    env = TandemMultiAgentEnv(track_name=track_name, max_steps=800)
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    mappo = MAPPO(
        obs_dim=obs_dim,
        action_dim=action_dim,
        lr_actor=lr_actor,
        lr_critic=lr_critic,
        device=device,
    )

    best_chase_reward = -float("inf")
    current_step = 0
    iteration = 0

    obs_dict, _ = env.reset()

    while current_step < total_timesteps:
        iteration += 1
        t_start = time.time()

        # Buffers for current rollout
        buf_obs_lead = []
        buf_act_lead = []
        buf_logp_lead = []
        buf_val_lead = []
        buf_rew_lead = []
        buf_done_lead = []

        buf_obs_chase = []
        buf_act_chase = []
        buf_logp_chase = []
        buf_val_chase = []
        buf_rew_chase = []
        buf_done_chase = []

        buf_global_state = []

        episode_proximity = []
        episode_angle_sync = []
        episode_wake_count = 0
        episodes_completed = 0

        for _ in range(rollout_steps):
            current_step += 1
            o_lead = obs_dict["leader"]
            o_chase = obs_dict["chaser"]
            g_state = mappo.get_global_state(o_lead, o_chase)

            step_data = mappo.select_action(o_lead, o_chase)
            act_lead = step_data["act_lead"]
            act_chase = step_data["act_chase"]

            # If leader is untrained at the beginning, blend with autopilot to accelerate chaser curriculum
            if iteration < 8 and np.random.random() < 0.60:
                s = env.leader_state
                track = env.track
                curr_pos = np.array([s.x, s.y])
                targets = track.get_target_waypoints(curr_pos, count=2, step_stride=7)
                rel = targets[0] - curr_pos
                cos_y = np.cos(s.yaw)
                sin_y = np.sin(s.yaw)
                by = -rel[0] * sin_y + rel[1] * cos_y
                act_lead = np.array([np.clip(by * 0.11, -1.0, 1.0), 0.85 if s.speed < 13.0 else 0.35, 0.0], dtype=np.float32)

            actions = {"leader": act_lead, "chaser": act_chase}
            next_obs_dict, rew_dict, term_dict, trunc_dict, info_dict = env.step(actions)

            done_lead = term_dict["leader"] or trunc_dict["leader"]
            done_chase = term_dict["chaser"] or trunc_dict["chaser"]

            # Store step
            buf_obs_lead.append(o_lead)
            buf_act_lead.append(act_lead)
            buf_logp_lead.append(step_data["logp_lead"])
            buf_val_lead.append(step_data["val_lead"])
            buf_rew_lead.append(rew_dict["leader"])
            buf_done_lead.append(done_lead)

            buf_obs_chase.append(o_chase)
            buf_act_chase.append(act_chase)
            buf_logp_chase.append(step_data["logp_chase"])
            buf_val_chase.append(step_data["val_chase"])
            buf_rew_chase.append(rew_dict["chaser"])
            buf_done_chase.append(done_chase)

            buf_global_state.append(g_state)

            c_info = info_dict["chaser"]
            if "distance" in c_info:
                episode_proximity.append(c_info["distance"])
            if "angle_diff_deg" in c_info:
                episode_angle_sync.append(c_info["angle_diff_deg"])
            if c_info.get("in_wake", 0.0) > 0.15:
                episode_wake_count += 1

            if done_lead or done_chase:
                episodes_completed += 1
                obs_dict, _ = env.reset()
            else:
                obs_dict = next_obs_dict

        # Bootstrap value for GAE
        final_o_lead = obs_dict["leader"]
        final_o_chase = obs_dict["chaser"]
        final_data = mappo.select_action(final_o_lead, final_o_chase)

        adv_lead, ret_lead = mappo.compute_gae(
            rewards=np.array(buf_rew_lead, dtype=np.float32),
            values=np.array(buf_val_lead, dtype=np.float32).squeeze(),
            dones=np.array(buf_done_lead, dtype=np.float32),
            next_value=float(np.squeeze(final_data["val_lead"])),
        )

        adv_chase, ret_chase = mappo.compute_gae(
            rewards=np.array(buf_rew_chase, dtype=np.float32),
            values=np.array(buf_val_chase, dtype=np.float32).squeeze(),
            dones=np.array(buf_done_chase, dtype=np.float32),
            next_value=float(np.squeeze(final_data["val_chase"])),
        )

        # MAPPO CTDE Update
        metrics = mappo.train_step(
            buffer_lead={
                "obs": np.array(buf_obs_lead, dtype=np.float32),
                "actions": np.array(buf_act_lead, dtype=np.float32),
                "log_probs": np.array(buf_logp_lead, dtype=np.float32),
                "advantages": adv_lead,
                "returns": ret_lead,
            },
            buffer_chase={
                "obs": np.array(buf_obs_chase, dtype=np.float32),
                "actions": np.array(buf_act_chase, dtype=np.float32),
                "log_probs": np.array(buf_logp_chase, dtype=np.float32),
                "advantages": adv_chase,
                "returns": ret_chase,
            },
            buffer_global={"states": np.array(buf_global_state, dtype=np.float32)},
            epochs=epochs,
            batch_size=batch_size,
        )

        dt_iter = time.time() - t_start
        fps = int(rollout_steps / dt_iter)
        mean_rew_chase = float(np.mean(buf_rew_chase))
        mean_rew_lead = float(np.mean(buf_rew_lead))
        mean_prox = float(np.mean(episode_proximity)) if episode_proximity else 0.0
        mean_angle_err = float(np.mean(episode_angle_sync)) if episode_angle_sync else 0.0

        print(
            f"Iter {iteration:03d} | Step {current_step:06d}/{total_timesteps:06d} | "
            f"Chase Rew: {mean_rew_chase:+.2f} | Lead Rew: {mean_rew_lead:+.2f} | "
            f"Door Gap: {mean_prox:.2f}m | Angle Diff: {mean_angle_err:.1f}° | "
            f"In-Wake Steps: {episode_wake_count} | FPS: {fps}"
        )

        # Checkpoint best model
        if mean_rew_chase > best_chase_reward and iteration >= 3:
            best_chase_reward = mean_rew_chase
            best_path = os.path.join(save_dir, "mappo_tandem_best.pt")
            mappo.save(best_path)

        # Periodic checkpoint
        if iteration % eval_freq == 0:
            ckpt_path = os.path.join(save_dir, f"mappo_tandem_iter_{iteration}.pt")
            mappo.save(ckpt_path)

    final_path = os.path.join(save_dir, "mappo_tandem_final.pt")
    mappo.save(final_path)
    print("--- MAPPO Training Complete! ---")
    return mappo


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train MAPPO Tandem Drift Battles")
    parser.add_argument("--steps", type=int, default=15000, help="Total training steps")
    parser.add_argument("--rollout", type=int, default=512, help="Rollout steps per iter")
    parser.add_argument("--track", type=str, default="touge", choices=["touge", "gymkhana"])
    parser.add_argument("--save-dir", type=str, default="models", help="Model save dir")

    args = parser.parse_args()
    train_mappo(
        total_timesteps=args.steps,
        rollout_steps=args.rollout,
        track_name=args.track,
        save_dir=args.save_dir,
    )
