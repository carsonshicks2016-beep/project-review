"""Plot raw returns, survival, curriculum level, and forward gait metrics."""
import argparse
import json
from pathlib import Path
import yaml
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

p = argparse.ArgumentParser()
p.add_argument('run', type=Path)
a = p.parse_args()
cfg = yaml.safe_load((a.run / 'config.yaml').read_text())
rows = json.loads((a.run / 'evaluations.json').read_text())

if cfg['stage'] == 4:
    fig, axes = plt.subplots(4, 1, figsize=(10, 12), sharex=True)
elif cfg['stage'] in (2, 3):
    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
else:
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

x = [r['timesteps'] for r in rows]

# Panel 0: Return
axes[0].plot(x, [r['mean_reward'] for r in rows], marker='o', label=f"Deterministic evaluation mean ({cfg['eval_episodes']} fixed seeds)")
axes[0].fill_between(x, [min(e['reward'] for e in r['episodes']) for r in rows],
                     [max(e['reward'] for e in r['episodes']) for r in rows], alpha=.15, label='Evaluation min–max')
for event in (a.run / 'tensorboard').rglob('events.out.tfevents.*'):
    acc = EventAccumulator(str(event)); acc.Reload()
    tag = 'rollout/ep_rew_mean'
    if tag in acc.Tags()['scalars']:
        values = acc.Scalars(tag)
        axes[0].plot([v.step for v in values], [v.value for v in values], alpha=.65, label='Training return (stochastic, rolling mean)')

best = max(rows, key=lambda r: r['mean_reward'])
axes[0].scatter([best['timesteps']], [best['mean_reward']], s=130, marker='*', color='darkgreen', zorder=5, label='Selected checkpoint')
axes[0].set_ylabel('Raw episode return')
axes[0].legend(fontsize=8)

# Panel 1: Episode Length
axes[1].plot(x, [sum(e['length'] for e in r['episodes']) / len(r['episodes']) for r in rows], marker='o', color='purple')
axes[1].set_ylabel('Evaluation episode length')
axes[1].set_ylim(0, 1050)

if cfg['stage'] == 4:
    # Panel 2: Curriculum Level & Obstacle Clearance
    ax2_twin = axes[2].twinx()
    l1 = axes[2].step(x, [r.get('curriculum_level', 1.0) for r in rows], where='post', color='crimson', linewidth=2, label='Curriculum Level (1-5)')
    axes[2].set_ylabel('Curriculum Level', color='crimson')
    axes[2].set_ylim(0.5, 5.5)
    axes[2].set_yticks([1, 2, 3, 4, 5])
    
    clr_rates = [r.get('clearance_rate', 0.0) * 100 for r in rows]
    l2 = ax2_twin.plot(x, clr_rates, marker='s', color='navy', linestyle='--', label='Obstacle Clearance Rate (%)')
    ax2_twin.set_ylabel('Clearance Rate (%)', color='navy')
    ax2_twin.set_ylim(-5, 105)
    lines = l1 + l2
    labels = [l.get_label() for l in lines]
    axes[2].legend(lines, labels, loc='lower right', fontsize=8)

    # Panel 3: Speed & Facing
    ax3_twin = axes[3].twinx()
    l3 = axes[3].plot(x, [r.get('speed', 0.0) for r in rows], marker='o', color='teal', label='Forward speed (m/s)')
    axes[3].set_ylabel('Speed (m/s)', color='teal')
    axes[3].axhline(0, color='gray', linewidth=.8)
    
    facing_vals = [r.get('facing', 1.0) for r in rows]
    l4 = ax3_twin.plot(x, facing_vals, marker='^', color='orange', linestyle=':', label='Torso Facing (+x)')
    ax3_twin.set_ylabel('Facing alignment', color='orange')
    ax3_twin.set_ylim(-0.2, 1.05)
    lines2 = l3 + l4
    labels2 = [l.get_label() for l in lines2]
    axes[3].legend(lines2, labels2, loc='lower right', fontsize=8)

elif cfg['stage'] in (2, 3):
    axes[2].plot(x, [r['speed'] for r in rows], marker='o', label='Mean forward COM speed')
    axes[2].axhline(0, color='gray', linewidth=.8)
    axes[2].set_ylabel('Forward speed (m/s)')
    axes[2].legend()

axes[-1].set_xlabel('Environment control decisions')
for ax in axes: ax.grid(alpha=.2)
fig.suptitle(f"Stage {cfg['stage']} • {cfg['env_id']} • Automated Curriculum PPO")
fig.tight_layout()
fig.savefig(a.run / 'reward-curve.png', dpi=160)

