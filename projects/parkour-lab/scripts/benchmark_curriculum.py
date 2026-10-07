"""Comprehensive curriculum benchmark: evaluates across all 5 difficulty tiers.
Produces quantitative metrics, evaluation videos, contact sheets, and comparison figures.
"""
import argparse
import json
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import imageio.v2 as imageio
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from envs.parkour_env import ParkourHumanoidEnv
from eval import evaluate

def make_contact_sheet(video_path: Path, out_png: Path, n_frames: int = 6):
    reader = imageio.get_reader(video_path)
    total_frames = reader.count_frames()
    meta = reader.get_meta_data()
    cols = 3
    rows = (n_frames + cols - 1) // cols
    w, h = 320, 240
    sheet = Image.new('RGB', (cols * w, rows * (h + 30)), 'white')
    draw = ImageDraw.Draw(sheet)
    for idx in range(n_frames):
        f_idx = round(idx * (total_frames - 1) / max(1, n_frames - 1))
        frame = Image.fromarray(reader.get_data(f_idx))
        frame.thumbnail((w, h))
        col = idx % cols
        row = idx // cols
        x = col * w
        y = row * (h + 30)
        sheet.paste(frame, (x, y))
        t_sec = f_idx / meta['fps']
        draw.text((x + 8, y + h + 6), f"t = {t_sec:.2f} s (frame {f_idx})", fill='black')
    reader.close()
    sheet.save(out_png)

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', type=Path, default=Path('runs/stage4-deep/best'))
    p.add_argument('--out-dir', type=Path, default=Path('runs/stage4-deep/benchmark'))
    p.add_argument('--episodes-per-tier', type=int, default=10)
    p.add_argument('--base-seed', type=int, default=30000)
    args = p.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)

    print(f"Loading checkpoint from: {args.checkpoint}")
    model = PPO.load(args.checkpoint / 'policy.zip', device='cpu')

    tier_results = {}
    tiers = [1.0, 2.0, 3.0, 4.0, 5.0]
    tier_names = {
        1.0: "Level 1: Novice",
        2.0: "Level 2: Intermediate",
        3.0: "Level 3: Advanced",
        4.0: "Level 4: Expert",
        5.0: "Level 5: Extreme Parkour",
    }

    for lvl in tiers:
        name = tier_names[lvl]
        print(f"\n==========================================")
        print(f"Evaluating {name} (difficulty={lvl})")
        print(f"==========================================")
        
        vec = DummyVecEnv([lambda: ParkourHumanoidEnv(difficulty=lvl)])
        norm = VecNormalize.load(args.checkpoint / 'vecnormalize.pkl', vec)
        norm.training = False
        norm.norm_reward = False

        video_path = args.out_dir / f"level_{int(lvl)}_eval.mp4"
        seeds = range(args.base_seed, args.base_seed + args.episodes_per_tier)
        res = evaluate(model, norm, 'ParkourHumanoid', seeds, video=video_path, difficulty=lvl)
        norm.close()

        contact_path = args.out_dir / f"level_{int(lvl)}_contact.png"
        if video_path.exists():
            make_contact_sheet(video_path, contact_path, n_frames=6)

        episodes = res['episodes']
        rewards = [e['reward'] for e in episodes]
        distances = [e['distance'] for e in episodes]
        speeds = [e['speed'] for e in episodes]
        clearances = [e['clearance_rate'] for e in episodes]
        completions = [e['completion_rate'] for e in episodes]
        uprights = [e['upright'] for e in episodes]
        facings = [e['facing'] for e in episodes]
        lengths = [e['length'] for e in episodes]
        success_count = sum(1 for e in episodes if e.get('completed', False) or e['clearance_rate'] >= 0.99)

        tier_summary = {
            "tier": lvl,
            "name": name,
            "mean_reward": float(np.mean(rewards)),
            "std_reward": float(np.std(rewards)),
            "mean_distance": float(np.mean(distances)),
            "std_distance": float(np.std(distances)),
            "mean_speed": float(np.mean(speeds)),
            "mean_clearance_rate": float(np.mean(clearances)),
            "std_clearance_rate": float(np.std(clearances)),
            "mean_completion_rate": float(np.mean(completions)),
            "success_rate": float(success_count / len(episodes)),
            "mean_upright": float(np.mean(uprights)),
            "mean_facing": float(np.mean(facings)),
            "mean_length": float(np.mean(lengths)),
            "episodes": episodes,
        }
        tier_results[f"level_{int(lvl)}"] = tier_summary
        print(f"  Return: {tier_summary['mean_reward']:.1f} +/- {tier_summary['std_reward']:.1f}")
        print(f"  Clearance Rate: {tier_summary['mean_clearance_rate']*100:.1f}%")
        print(f"  Success Rate: {tier_summary['success_rate']*100:.1f}%")
        print(f"  Distance: {tier_summary['mean_distance']:.2f} m")
        print(f"  Speed: {tier_summary['mean_speed']:.2f} m/s")
        print(f"  Facing: {tier_summary['mean_facing']:.3f}")
        print(f"  Upright: {tier_summary['mean_upright']:.3f}")

    benchmark_json = args.out_dir / 'curriculum_benchmark.json'
    benchmark_json.write_text(json.dumps(tier_results, indent=2))
    print(f"\nSaved benchmark metrics to {benchmark_json}")

    # Plot 1: 5-Tier Benchmark Comparison Figure
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    tier_labels = [f"L1\nNovice", f"L2\nInterm.", f"L3\nAdv.", f"L4\nExpert", f"L5\nExtreme"]
    colors = ['#2ca02c', '#1f77b4', '#9467bd', '#ff7f0e', '#d62728']

    # (a) Obstacle Clearance Rate & Success Rate
    ax = axes[0, 0]
    clr_means = [tier_results[f"level_{int(l)}"]["mean_clearance_rate"] * 100 for l in tiers]
    succ_rates = [tier_results[f"level_{int(l)}"]["success_rate"] * 100 for l in tiers]
    x_pos = np.arange(len(tiers))
    width = 0.35
    ax.bar(x_pos - width/2, clr_means, width, label='Obstacle Clearance Rate (%)', color='#1f77b4')
    ax.bar(x_pos + width/2, succ_rates, width, label='Full Course Cleared (%)', color='#2ca02c')
    ax.set_ylabel('Rate (%)')
    ax.set_title('Course Mastery by Curriculum Tier')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(tier_labels)
    ax.set_ylim(0, 110)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    # (b) Mean Raw Return
    ax = axes[0, 1]
    ret_means = [tier_results[f"level_{int(l)}"]["mean_reward"] for l in tiers]
    ret_stds = [tier_results[f"level_{int(l)}"]["std_reward"] for l in tiers]
    ax.bar(x_pos, ret_means, yerr=ret_stds, capsize=5, color=colors, alpha=0.85)
    ax.set_ylabel('Raw Episode Return')
    ax.set_title('Episode Return Across Difficulty Tiers')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(tier_labels)
    ax.grid(alpha=0.3)

    # (c) Forward Distance Traveled
    ax = axes[1, 0]
    dist_means = [tier_results[f"level_{int(l)}"]["mean_distance"] for l in tiers]
    dist_stds = [tier_results[f"level_{int(l)}"]["std_distance"] for l in tiers]
    ax.bar(x_pos, dist_means, yerr=dist_stds, capsize=5, color='#34495e', alpha=0.85)
    ax.set_ylabel('Distance (meters)')
    ax.set_title('Forward Obstacle Distance Cleared')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(tier_labels)
    ax.grid(alpha=0.3)

    # (d) Locomotion Quality: Speed & Facing
    ax = axes[1, 1]
    speed_means = [tier_results[f"level_{int(l)}"]["mean_speed"] for l in tiers]
    facing_means = [tier_results[f"level_{int(l)}"]["mean_facing"] for l in tiers]
    ax.plot(x_pos, speed_means, marker='o', linewidth=2.5, color='#e74c3c', label='Forward Speed (m/s)')
    ax.set_ylabel('Speed (m/s)', color='#e74c3c')
    ax.set_ylim(0.8, 1.8)
    ax_twin = ax.twinx()
    ax_twin.plot(x_pos, facing_means, marker='s', linewidth=2.5, linestyle='--', color='#2980b9', label='Heading Alignment ($f_x$)')
    ax_twin.set_ylabel('Heading Alignment', color='#2980b9')
    ax_twin.set_ylim(0.85, 1.0)
    ax.set_title('Locomotion Speed and Forward Facing Alignment')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(tier_labels)
    lines_a, labels_a = ax.get_legend_handles_labels()
    lines_b, labels_b = ax_twin.get_legend_handles_labels()
    ax.legend(lines_a + lines_b, labels_a + labels_b, loc='lower left', fontsize=9)
    ax.grid(alpha=0.3)

    fig.suptitle('Stage 4 Automated Curriculum Learning: Multi-Tier Generalization Benchmark', fontsize=14, fontweight='bold')
    fig.tight_layout()
    chart_path = args.out_dir / 'curriculum_tier_comparison.png'
    fig.savefig(chart_path, dpi=180)
    plt.close(fig)
    print(f"Saved comparison figure to {chart_path}")

    # Plot 2: Combined Training Curve (runs/stage4 + runs/stage4-deep)
    evals1_path = Path('runs/stage4/evaluations.json')
    evals2_path = Path('runs/stage4-deep/evaluations.json')
    all_evals = []
    if evals1_path.exists():
        all_evals.extend(json.loads(evals1_path.read_text()))
    if evals2_path.exists():
        evals2 = json.loads(evals2_path.read_text())
        last_step = all_evals[-1]['timesteps'] if all_evals else 0
        for e in evals2:
            e_copy = dict(e)
            e_copy['timesteps'] = last_step + e['timesteps']
            all_evals.append(e_copy)

    if all_evals:
        fig, axes = plt.subplots(4, 1, figsize=(11, 13), sharex=True)
        steps = [e['timesteps'] for e in all_evals]
        rewards = [e['mean_reward'] for e in all_evals]
        levels = [e.get('curriculum_level', 1.0) for e in all_evals]
        clearances = [e.get('clearance_rate', 0.0) * 100 for e in all_evals]
        lengths = [np.mean([ep['length'] for ep in e['episodes']]) for e in all_evals]
        speeds = [e.get('speed', 0.0) for e in all_evals]
        facings = [e.get('facing', 1.0) for e in all_evals]

        # Panel 0: Return
        axes[0].plot(steps, rewards, marker='o', color='#1f77b4', label='Evaluation Return (mean over 5 seeds)')
        axes[0].fill_between(steps,
                             [min(ep['reward'] for ep in e['episodes']) for e in all_evals],
                             [max(ep['reward'] for ep in e['episodes']) for e in all_evals],
                             alpha=0.15, color='#1f77b4')
        axes[0].set_ylabel('Raw Episode Return')
        axes[0].set_title('A. Return Trajectory During Automated Curriculum Learning')
        axes[0].legend(fontsize=9)
        axes[0].grid(alpha=0.3)

        # Panel 1: Curriculum Ladder Progression
        ax1_twin = axes[1].twinx()
        l1 = axes[1].step(steps, levels, where='post', color='#d62728', linewidth=2.5, label='Curriculum Level (1-5)')
        axes[1].set_ylabel('Curriculum Level', color='#d62728', fontweight='bold')
        axes[1].set_ylim(0.5, 5.5)
        axes[1].set_yticks([1, 2, 3, 4, 5])
        axes[1].set_yticklabels(['L1 Novice', 'L2 Interm.', 'L3 Adv.', 'L4 Expert', 'L5 Extreme'])

        l2 = ax1_twin.plot(steps, clearances, marker='s', markersize=4, linestyle='--', color='#2ca02c', label='Obstacle Clearance Rate (%)')
        ax1_twin.set_ylabel('Clearance Rate (%)', color='#2ca02c', fontweight='bold')
        ax1_twin.set_ylim(-5, 105)
        lines = l1 + l2
        labels = [l.get_label() for l in lines]
        axes[1].legend(lines, labels, loc='lower right', fontsize=9)
        axes[1].set_title('B. Automated Curriculum Progression (Promotion on Clearance >= 60%)')
        axes[1].grid(alpha=0.3)

        # Panel 2: Episode Survival / Length
        axes[2].plot(steps, lengths, marker='o', color='#8e44ad', label='Mean Episode Length (max=1000)')
        axes[2].axhline(1000, color='gray', linestyle=':', label='Max Horizon (1000 steps)')
        axes[2].set_ylabel('Episode Steps')
        axes[2].set_ylim(0, 1050)
        axes[2].set_title('C. Episode Duration and Course Survival')
        axes[2].legend(fontsize=9)
        axes[2].grid(alpha=0.3)

        # Panel 3: Locomotion Speed and Facing
        ax3_twin = axes[3].twinx()
        l3 = axes[3].plot(steps, speeds, marker='o', color='#e67e22', label='Forward Speed (m/s)')
        axes[3].set_ylabel('Speed (m/s)', color='#e67e22')
        axes[3].set_ylim(0.5, 1.8)

        l4 = ax3_twin.plot(steps, facings, marker='^', markersize=4, linestyle=':', color='#2980b9', label='Heading Alignment ($f_x$)')
        ax3_twin.set_ylabel('Heading Alignment', color='#2980b9')
        ax3_twin.set_ylim(0.85, 1.02)
        lines2 = l3 + l4
        labels2 = [l.get_label() for l in lines2]
        axes[3].legend(lines2, labels2, loc='lower right', fontsize=9)
        axes[3].set_title(r'D. Forward Velocity and Heading Stability ($f_x \approx 0.96$)')
        axes[3].set_xlabel('Environment Control Decisions (Timesteps)')
        axes[3].grid(alpha=0.3)

        fig.suptitle('Stage 4 Automated Curriculum Learning: Full Training Progression', fontsize=14, fontweight='bold')
        fig.tight_layout()
        prog_path = args.out_dir / 'curriculum_ladder_progression.png'
        fig.savefig(prog_path, dpi=180)
        plt.close(fig)
        print(f"Saved full progression figure to {prog_path}")

if __name__ == '__main__':
    main()
