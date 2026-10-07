"""
Evaluation and telemetry benchmarking for trained drift models.
"""

import argparse
import json
import os
from typing import Dict, Any, List
import numpy as np
from stable_baselines3 import PPO, SAC

from drift_rl.env import DriftGymkhanaEnv


def evaluate_agent(
    model_path: Optional[str] = None,
    algo: str = "ppo",
    track: str = "touge",
    episodes: int = 5,
    save_trajectory: Optional[str] = None,
) -> Dict[str, Any]:
    """Run evaluation episodes and record detailed drift telemetry."""
    env = DriftGymkhanaEnv(track_name=track)

    model = None
    if model_path and os.path.exists(model_path):
        print(f"Loading trained {algo.upper()} model from {model_path}...")
        if algo.lower() == "sac":
            model = SAC.load(model_path)
        else:
            model = PPO.load(model_path)
    else:
        print("No model path provided or file missing. Running evaluation with random policy...")

    all_scores: List[float] = []
    all_multipliers: List[float] = []
    all_drift_angles: List[float] = []
    all_drift_speeds: List[float] = []
    trick_counts: Dict[str, int] = {}
    collisions: int = 0
    spinouts: int = 0

    full_trajectories: List[List[Dict[str, Any]]] = []

    for ep in range(episodes):
        obs, info = env.reset(seed=100 + ep)
        ep_score = 0.0
        max_mult = 1.0
        done = False
        step = 0
        ep_traj = []

        while not done:
            if model is not None:
                action, _ = model.predict(obs, deterministic=True)
            else:
                action = env.action_space.sample()

            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            # Track metrics
            slip_deg = abs(info["slip_angle_deg"])
            speed = info["speed"]
            mult = info["multiplier"]
            ep_score = info["score"]

            if info["is_drifting"]:
                all_drift_angles.append(slip_deg)
                all_drift_speeds.append(speed)

            if mult > max_mult:
                max_mult = mult

            for trick_name in info.get("tricks", []):
                trick_counts[trick_name] = trick_counts.get(trick_name, 0) + 1

            if info.get("collision", False):
                collisions += 1
            if info.get("spinout", False):
                spinouts += 1

            if save_trajectory:
                ep_traj.append({
                    "step": step,
                    "x": float(env.state.x),
                    "y": float(env.state.y),
                    "yaw": float(env.state.yaw),
                    "vx": float(env.state.vx),
                    "vy": float(env.state.vy),
                    "steer": float(env.state.steer),
                    "slip_deg": float(slip_deg),
                    "speed": float(speed),
                    "mult": float(mult),
                    "score": float(ep_score),
                    "tricks": info.get("tricks", []),
                })

            step += 1

        all_scores.append(ep_score)
        all_multipliers.append(max_mult)
        if save_trajectory:
            full_trajectories.append(ep_traj)

        print(f" Episode {ep + 1:02d}/{episodes:02d} | Score: {ep_score:,.1f} | Peak Multiplier: x{max_mult:.1f} | Steps: {step}")

    env.close()

    metrics = {
        "episodes": episodes,
        "mean_score": float(np.mean(all_scores)) if all_scores else 0.0,
        "max_score": float(np.max(all_scores)) if all_scores else 0.0,
        "mean_peak_multiplier": float(np.mean(all_multipliers)) if all_multipliers else 1.0,
        "max_multiplier": float(np.max(all_multipliers)) if all_multipliers else 1.0,
        "mean_drift_angle_deg": float(np.mean(all_drift_angles)) if all_drift_angles else 0.0,
        "max_drift_angle_deg": float(np.max(all_drift_angles)) if all_drift_angles else 0.0,
        "mean_drift_speed_mps": float(np.mean(all_drift_speeds)) if all_drift_speeds else 0.0,
        "mean_drift_speed_kmh": float(np.mean(all_drift_speeds) * 3.6) if all_drift_speeds else 0.0,
        "trick_counts": trick_counts,
        "collision_rate": float(collisions / episodes),
        "spinout_rate": float(spinouts / episodes),
    }

    print("\n============================================================")
    print(" DRIFT TELEMETRY SUMMARY")
    print("============================================================")
    print(f" Mean Drift Score:      {metrics['mean_score']:,.1f} pts")
    print(f" High Score:            {metrics['max_score']:,.1f} pts")
    print(f" Max Combo Multiplier:  x{metrics['max_multiplier']:.1f}")
    print(f" Mean Drift Angle:      {metrics['mean_drift_angle_deg']:.1f}°")
    print(f" Peak Drift Angle:      {metrics['max_drift_angle_deg']:.1f}°")
    print(f" Mean Drift Speed:      {metrics['mean_drift_speed_kmh']:.1f} km/h ({metrics['mean_drift_speed_mps']:.1f} m/s)")
    print(f" Tricks Executed:       {sum(trick_counts.values())} total")
    for trick, count in sorted(trick_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"   - {trick}: {count}x")
    print(f" Crash Rate:            {metrics['collision_rate'] * 100:.1f}%")
    print(f" Spinout Rate:          {metrics['spinout_rate'] * 100:.1f}%")
    print("============================================================")

    if save_trajectory:
        with open(save_trajectory, "w") as f:
            json.dump({"metrics": metrics, "trajectories": full_trajectories}, f, indent=2)
        print(f"Saved trajectory telemetry to {save_trajectory}")

    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Drift Model")
    parser.add_argument("--model", type=str, default=None, help="Path to trained model .zip")
    parser.add_argument("--algo", type=str, default="ppo", choices=["ppo", "sac"], help="Algorithm type")
    parser.add_argument("--track", type=str, default="touge", choices=["touge", "gymkhana", "stadium"], help="Track")
    parser.add_argument("--episodes", type=int, default=5, help="Number of evaluation episodes")
    parser.add_argument("--save-trajectory", type=str, default=None, help="Output JSON trajectory file")
    args = parser.parse_args()

    evaluate_agent(
        model_path=args.model,
        algo=args.algo,
        track=args.track,
        episodes=args.episodes,
        save_trajectory=args.save_trajectory,
    )
