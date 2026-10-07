"""
Evaluation and trajectory exporter for trained Satellite RVD agent.
Generates telemetry logs for the 3D WebGL visualizer and static trajectory analysis plots.
"""

import os
import sys
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stable_baselines3 import PPO
from satellite_rl.env import SatelliteDockingEnv


def evaluate(
    model_path: str = "models/ppo_satellite_docking.zip",
    num_episodes: int = 20,
    stage: str = "curriculum",
    curriculum_level: float = 0.5,
    output_json: str = "visualizer/trajectories.json",
    output_plot: str = "satellite_rl/trajectory_analysis.png",
    deterministic: bool = True
):
    print(f"--- Evaluating Satellite Docking Agent ---")
    print(f"Model path: {model_path}")
    print(f"Test episodes: {num_episodes}")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")

    model = PPO.load(model_path)
    env = SatelliteDockingEnv(stage=stage, curriculum_level=curriculum_level)

    episodes_data = []
    success_count = 0
    collision_count = 0
    out_of_bounds_count = 0
    delta_vs = []
    final_speeds = []
    durations = []

    best_episode = None
    best_reward = -float("inf")

    for ep in range(num_episodes):
        obs, reset_info = env.reset()
        init_distance = reset_info["initial_distance"]
        done = False
        total_reward = 0.0

        states = []
        actions = []
        speeds = []
        distances = []
        corridor_flags = []
        fuel_levels = []

        while not done:
            action, _ = model.predict(obs, deterministic=deterministic)

            # Record telemetry before step
            state = env.state.copy()
            states.append({
                "x": round(float(state[0]), 3),
                "y": round(float(state[1]), 3),
                "z": round(float(state[2]), 3),
                "vx": round(float(state[3]), 4),
                "vy": round(float(state[4]), 4),
                "vz": round(float(state[5]), 4)
            })
            actions.append({
                "ux": round(float(action[0]), 3),
                "uy": round(float(action[1]), 3),
                "uz": round(float(action[2]), 3)
            })
            dist = float(np.linalg.norm(state[:3]))
            speed = float(np.linalg.norm(state[3:6]))
            distances.append(round(dist, 3))
            speeds.append(round(speed, 4))
            fuel_levels.append(round(float(env.propellant), 3))
            corridor_flags.append(bool((state[1] > 0) and (np.sqrt(state[0]**2 + state[2]**2) <= state[1] * env.corridor_tan + 0.1)))

            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            done = terminated or truncated

        # Add final state
        state = env.state.copy()
        states.append({
            "x": round(float(state[0]), 3),
            "y": round(float(state[1]), 3),
            "z": round(float(state[2]), 3),
            "vx": round(float(state[3]), 4),
            "vy": round(float(state[4]), 4),
            "vz": round(float(state[5]), 4)
        })
        actions.append({"ux": 0.0, "uy": 0.0, "uz": 0.0})
        distances.append(round(float(np.linalg.norm(state[:3])), 3))
        speeds.append(round(float(np.linalg.norm(state[3:6])), 4))
        fuel_levels.append(round(float(env.propellant), 3))
        corridor_flags.append(True)

        if info["success"]:
            success_count += 1
        elif info["collision"]:
            collision_count += 1
        elif info["out_of_bounds"]:
            out_of_bounds_count += 1

        delta_vs.append(env.total_delta_v)
        final_speeds.append(float(np.linalg.norm(env.state[3:6])))
        durations.append(len(states))

        ep_summary = {
            "episode": ep + 1,
            "success": bool(info["success"]),
            "collision": bool(info["collision"]),
            "initial_distance": round(float(init_distance), 2),
            "final_distance": round(float(info["distance"]), 3),
            "final_speed": round(float(info["speed"]), 4),
            "delta_v": round(float(env.total_delta_v), 3),
            "propellant_remaining": round(float(env.propellant), 3),
            "steps": len(states),
            "total_reward": round(total_reward, 2),
            "states": states,
            "actions": actions,
            "distances": distances,
            "speeds": speeds,
            "corridor_flags": corridor_flags,
            "fuel_levels": fuel_levels
        }
        episodes_data.append(ep_summary)

        if total_reward > best_reward:
            best_reward = total_reward
            best_episode = ep_summary

    success_rate = (success_count / num_episodes) * 100.0
    mean_dv = float(np.mean(delta_vs))
    mean_duration = float(np.mean(durations))
    mean_final_speed = float(np.mean(final_speeds))

    print(f"\n===== Performance Results ({num_episodes} Runs) =====")
    print(f"Success Rate:             {success_rate:.1f}% ({success_count}/{num_episodes})")
    print(f"Collisions:               {collision_count}/{num_episodes}")
    print(f"Out of Bounds:            {out_of_bounds_count}/{num_episodes}")
    print(f"Mean Delta-V Expended:    {mean_dv:.3f} m/s")
    print(f"Mean Final Speed at Dock: {mean_final_speed * 100:.2f} cm/s")
    print(f"Mean Episode Duration:    {mean_duration:.1f} steps")

    # Export to JSON for 3D Visualizer
    os.makedirs(os.path.dirname(output_json), exist_ok=True)
    export_payload = {
        "summary": {
            "num_episodes": num_episodes,
            "success_rate": round(success_rate, 1),
            "mean_delta_v": round(mean_dv, 3),
            "mean_final_speed": round(mean_final_speed, 4),
            "corridor_angle_deg": 25.0
        },
        "episodes": episodes_data
    }
    with open(output_json, "w") as f:
        json.dump(export_payload, f, indent=2)
    print(f"Telemetry exported to visualizer: {output_json}")

    # Generate static trajectory plot using best episode
    if best_episode is not None:
        plot_trajectory(best_episode, output_plot)
        print(f"Trajectory analysis plot saved to: {output_plot}")

    return success_rate


def plot_trajectory(ep: dict, save_path: str):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"Orbital Rendezvous & Docking Trajectory Analysis (Episode #{ep['episode']})", fontsize=14, fontweight='bold')

    states = ep["states"]
    actions = ep["actions"]
    xs = [s["x"] for s in states]
    ys = [s["y"] for s in states]
    zs = [s["z"] for s in states]
    steps = np.arange(len(states))

    # 1. In-Plane Trajectory (V-bar vs R-bar: Y vs X)
    ax1 = axes[0, 0]
    ax1.plot(ys, xs, color="#00ffff", lw=2, label="Chaser Trajectory")
    ax1.scatter([ys[0]], [xs[0]], color="#ff9900", s=60, zorder=5, label="Initial Spawn")
    ax1.scatter([ys[-1]], [xs[-1]], color="#00ff00", s=60, zorder=5, label="Final Dock")
    # Plot V-bar corridor cone lines
    max_y = max(ys) * 1.05
    y_cone = np.linspace(0, max_y, 100)
    tan_c = np.tan(np.radians(25.0))
    ax1.plot(y_cone, y_cone * tan_c, 'r--', alpha=0.5, label="Approach Cone")
    ax1.plot(y_cone, -y_cone * tan_c, 'r--', alpha=0.5)
    ax1.scatter([0], [0], color="#ffffff", s=100, marker='*', label="Target Port")
    ax1.set_xlabel("Along-track / V-bar Y [m]")
    ax1.set_ylabel("Radial / R-bar X [m]")
    ax1.set_title("Relative Motion in LVLH Orbit Plane")
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="upper right", fontsize=8)

    # 2. Closing Distance Profile
    ax2 = axes[0, 1]
    ax2.plot(steps, ep["distances"], color="#3399ff", lw=2)
    ax2.set_xlabel("Time Steps (dt = 1s)")
    ax2.set_ylabel("Distance to Target [m]")
    ax2.set_title("Range to Docking Port vs Time")
    ax2.grid(True, alpha=0.3)

    # 3. Relative Speed vs Glideslope Limit
    ax3 = axes[1, 0]
    ax3.plot(steps, ep["speeds"], color="#ff3366", lw=2, label="Actual Relative Speed")
    # Glideslope soft target
    nom_speeds = [0.05 + 0.06 * np.sqrt(max(0.0, d)) for d in ep["distances"]]
    ax3.plot(steps, nom_speeds, 'g--', alpha=0.7, label="Nominal Glideslope Target")
    ax3.axhline(y=0.06, color='yellow', linestyle=':', label="Max Docking Capture Speed (6 cm/s)")
    ax3.set_xlabel("Time Steps (dt = 1s)")
    ax3.set_ylabel("Relative Speed [m/s]")
    ax3.set_title("Velocity Braking Profile")
    ax3.grid(True, alpha=0.3)
    ax3.legend(loc="upper right", fontsize=8)

    # 4. RCS Thruster Control Inputs
    ax4 = axes[1, 1]
    ux = [a["ux"] for a in actions]
    uy = [a["uy"] for a in actions]
    uz = [a["uz"] for a in actions]
    ax4.plot(steps, ux, label="Ux (Radial)", alpha=0.8)
    ax4.plot(steps, uy, label="Uy (In-track / Braking)", alpha=0.8)
    ax4.plot(steps, uz, label="Uz (Cross-track)", alpha=0.8)
    ax4.set_xlabel("Time Steps (dt = 1s)")
    ax4.set_ylabel("Normalized Thrust [-1.0, 1.0]")
    ax4.set_title("RCS Thruster Actuation Profile")
    ax4.grid(True, alpha=0.3)
    ax4.legend(loc="upper right", fontsize=8)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Satellite RVD Agent")
    parser.add_argument("--model", type=str, default="models/ppo_satellite_docking.zip")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--stage", type=str, default="curriculum")
    parser.add_argument("--level", type=float, default=0.5)
    parser.add_argument("--json", type=str, default="visualizer/trajectories.json")
    parser.add_argument("--plot", type=str, default="satellite_rl/trajectory_analysis.png")

    args = parser.parse_args()
    evaluate(
        model_path=args.model,
        num_episodes=args.episodes,
        stage=args.stage,
        curriculum_level=args.level,
        output_json=args.json,
        output_plot=args.plot
    )
