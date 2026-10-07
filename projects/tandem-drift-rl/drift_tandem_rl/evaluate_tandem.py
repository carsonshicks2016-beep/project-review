"""
Evaluation and Tournament Benchmark for Tandem Drift Battles.
Simulates multi-heat Formula Drift battles, computes judging statistics,
and generates telemetry analysis graphs.
"""

import argparse
import os
import sys

# Ensure repository root is on sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
import torch

from drift_tandem_rl.env import TandemMultiAgentEnv
from drift_tandem_rl.mappo import MAPPO


def evaluate_battles(
    model_path: Optional[str] = None,
    track_name: str = "touge",
    num_heats: int = 5,
    max_steps_per_heat: int = 500,
    save_chart: str = "tandem_eval_telemetry.png",
):
    print(f"\n=======================================================")
    print(f"  FORMULA DRIFT TANDEM BATTLE TOURNAMENT EVALUATION")
    print(f"=======================================================")
    print(f"Track: {track_name.upper()} | Heats: {num_heats} | Steps/Heat: {max_steps_per_heat}\n")

    env = TandemMultiAgentEnv(track_name=track_name, max_steps=max_steps_per_heat)
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    mappo = MAPPO(obs_dim=obs_dim, action_dim=action_dim, device=device)

    if model_path and os.path.exists(model_path):
        mappo.load(model_path)
    else:
        print("Note: Running evaluation with initialized policy / autopilot baseline.")

    heat_results = []
    best_telemetry = None
    best_chase_score = -float("inf")

    for heat_idx in range(1, num_heats + 1):
        obs_dict, _ = env.reset(seed=100 + heat_idx)
        heat_steps = 0

        # Telemetry logs for plotting
        t_steps = []
        log_clearance = []
        log_lead_slip = []
        log_chase_slip = []
        log_wake = []
        log_lead_score = []
        log_chase_score = []

        while heat_steps < max_steps_per_heat:
            heat_steps += 1
            o_lead = obs_dict["leader"]
            o_chase = obs_dict["chaser"]

            actions_data = mappo.select_action(o_lead, o_chase, deterministic=True)
            act_lead = actions_data["act_lead"]
            act_chase = actions_data["act_chase"]

            # Autopilot fallback if lead actor is untrained
            if np.abs(act_lead).sum() < 0.1:
                s = env.leader_state
                track = env.track
                curr_pos = np.array([s.x, s.y])
                targets = track.get_target_waypoints(curr_pos, count=2, step_stride=7)
                rel = targets[0] - curr_pos
                cos_y = np.cos(s.yaw)
                sin_y = np.sin(s.yaw)
                by = -rel[0] * sin_y + rel[1] * cos_y
                act_lead = np.array([np.clip(by * 0.11, -1.0, 1.0), 0.85 if s.speed < 13.5 else 0.35, 0.0], dtype=np.float32)

            actions = {"leader": act_lead, "chaser": act_chase}
            next_obs, rewards, terms, truncs, infos = env.step(actions)

            c_info = infos["chaser"]
            t_steps.append(heat_steps * 0.02)
            log_clearance.append(c_info.get("distance", 0.0))
            log_lead_slip.append(float(env.leader_state.slip_angle_deg))
            log_chase_slip.append(float(env.chaser_state.slip_angle_deg))
            log_wake.append(float(c_info.get("in_wake", 0.0) * 100.0))
            log_lead_score.append(float(infos["leader"].get("lead_score", 0.0)))
            log_chase_score.append(float(infos["chaser"].get("chase_score", 0.0)))

            if terms["leader"] or terms["chaser"] or truncs["leader"]:
                break
            obs_dict = next_obs

        # Heat decision from Formula Drift Judge
        decision = env.judge.judge_heat_decision()
        avg_gap = np.mean(log_clearance) if log_clearance else 0.0
        sweet_spot_pct = (
            np.mean([(0.8 <= g <= 2.8) for g in log_clearance]) * 100.0
            if log_clearance else 0.0
        )
        wake_pct = np.mean([(w > 15.0) for w in log_wake]) * 100.0 if log_wake else 0.0

        final_lead_pts = env.judge.lead_score.total_score
        final_chase_pts = env.judge.chase_score.total_score

        heat_results.append({
            "heat": heat_idx,
            "steps": heat_steps,
            "lead_pts": final_lead_pts,
            "chase_pts": final_chase_pts,
            "winner": decision.winner,
            "avg_gap": avg_gap,
            "sweet_spot_pct": sweet_spot_pct,
            "wake_pct": wake_pct,
            "notes": decision.notes,
        })

        print(
            f"Heat {heat_idx:02d}: Winner={decision.winner:<7} | "
            f"Lead={final_lead_pts:5.1f} pts | Chase={final_chase_pts:5.1f} pts | "
            f"Avg Gap={avg_gap:.2f}m | SweetSpot={sweet_spot_pct:.0f}% | "
            f"Wake Time={wake_pct:.0f}% | Notes: {', '.join(decision.notes) if decision.notes else 'Clean tandem'}"
        )

        if final_chase_pts > best_chase_score:
            best_chase_score = final_chase_pts
            best_telemetry = {
                "t": t_steps,
                "gap": log_clearance,
                "lead_slip": log_lead_slip,
                "chase_slip": log_chase_slip,
                "wake": log_wake,
                "lead_score": log_lead_score,
                "chase_score": log_chase_score,
                "heat": heat_idx,
            }

    # Summary
    print("\n----------------- TOURNAMENT SUMMARY -----------------")
    chase_wins = sum(1 for h in heat_results if h["winner"] == "CHASER")
    lead_wins = sum(1 for h in heat_results if h["winner"] == "LEADER")
    ties = sum(1 for h in heat_results if h["winner"] == "TIE")
    overall_gap = np.mean([h["avg_gap"] for h in heat_results])
    overall_sweet = np.mean([h["sweet_spot_pct"] for h in heat_results])

    print(f"Record: Chaser Wins: {chase_wins} | Leader Wins: {lead_wins} | Ties (OMT): {ties}")
    print(f"Tournament Mean Door-to-Door Clearance: {overall_gap:.2f} meters")
    print(f"Time in Proximity Sweet Spot (0.8m - 2.8m): {overall_sweet:.1f}%")

    # Generate telemetry chart
    if best_telemetry and len(best_telemetry["t"]) > 1:
        _plot_telemetry(best_telemetry, save_chart)

    return heat_results


def _plot_telemetry(tdata: dict, filename: str):
    """Plots a 4-panel comprehensive tandem drift telemetry chart."""
    fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
    fig.patch.set_facecolor("#161a20")

    t = tdata["t"]

    # 1. Door-to-Door Proximity (Clearance)
    ax1 = axes[0]
    ax1.set_facecolor("#1e232b")
    ax1.plot(t, tdata["gap"], color="#ffc107", lw=2, label="Door-to-Door Clearance (m)")
    ax1.axhspan(0.8, 2.8, color="#28a745", alpha=0.25, label="Optimal Proximity Sweet Spot (0.8m - 2.8m)")
    ax1.set_ylabel("Clearance (m)", color="white")
    ax1.legend(loc="upper right", facecolor="#161a20", labelcolor="white")
    ax1.grid(True, color="#343a40", linestyle="--")
    ax1.tick_params(colors="white")

    # 2. Angle Synchronization (Lead vs Chase Slip Angles)
    ax2 = axes[1]
    ax2.set_facecolor("#1e232b")
    ax2.plot(t, tdata["lead_slip"], color="#00e5ff", lw=2, label="Leader Slip Angle β (°)")
    ax2.plot(t, tdata["chase_slip"], color="#ff7043", lw=2, linestyle="--", label="Chaser Slip Angle β (°)")
    ax2.set_ylabel("Slip Angle (°)", color="white")
    ax2.legend(loc="upper right", facecolor="#161a20", labelcolor="white")
    ax2.grid(True, color="#343a40", linestyle="--")
    ax2.tick_params(colors="white")

    # 3. Dirty Air Wake & Downforce Loss
    ax3 = axes[2]
    ax3.set_facecolor("#1e232b")
    ax3.fill_between(t, 0, tdata["wake"], color="#e91e63", alpha=0.4, label="Wake Dirty Air Intensity (%)")
    ax3.plot(t, tdata["wake"], color="#e91e63", lw=1.5)
    ax3.set_ylabel("Wake Deficit (%)", color="white")
    ax3.legend(loc="upper right", facecolor="#161a20", labelcolor="white")
    ax3.grid(True, color="#343a40", linestyle="--")
    ax3.tick_params(colors="white")

    # 4. Cumulative Formula Drift Battle Score
    ax4 = axes[3]
    ax4.set_facecolor("#1e232b")
    ax4.plot(t, tdata["lead_score"], color="#00e5ff", lw=2.2, label="Leader Drift Points")
    ax4.plot(t, tdata["chase_score"], color="#ffc107", lw=2.2, label="Chaser Drift Points")
    ax4.set_xlabel("Battle Time (seconds)", color="white")
    ax4.set_ylabel("Judging Score", color="white")
    ax4.legend(loc="upper left", facecolor="#161a20", labelcolor="white")
    ax4.grid(True, color="#343a40", linestyle="--")
    ax4.tick_params(colors="white")

    fig.suptitle(
        f"Formula Drift Tandem Battle Telemetry Analysis (Heat {tdata['heat']})",
        color="#00e5ff",
        fontsize=16,
        fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(filename, dpi=180, facecolor=fig.get_facecolor())
    plt.close()
    print(f"Generated telemetry analysis chart at {filename}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Tandem Drift Battles")
    parser.add_argument("--model", type=str, default=None, help="Path to MAPPO checkpoint (.pt)")
    parser.add_argument("--track", type=str, default="touge", choices=["touge", "gymkhana"])
    parser.add_argument("--heats", type=int, default=5, help="Number of tandem battle heats")
    parser.add_argument("--chart", type=str, default="tandem_eval_telemetry.png", help="Telemetry plot path")

    args = parser.parse_args()
    evaluate_battles(
        model_path=args.model,
        track_name=args.track,
        num_heats=args.heats,
        save_chart=args.chart,
    )
