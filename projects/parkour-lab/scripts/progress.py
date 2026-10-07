"""Compact live progress from TensorBoard plus the latest deterministic review."""
import argparse,json
from pathlib import Path
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
p=argparse.ArgumentParser();p.add_argument('run',type=Path);a=p.parse_args()
for f in (a.run/'tensorboard').rglob('events.out.tfevents.*'):
    e=EventAccumulator(str(f),size_guidance={'scalars':1});e.Reload()
    out={}
    for tag in ['time/fps','rollout/ep_rew_mean','rollout/ep_len_mean','train/approx_kl','train/clip_fraction','train/std']:
        if tag in e.Tags()['scalars']:
            v=e.Scalars(tag)[-1];out[tag]={'steps':v.step,'value':round(v.value,4)}
    print(json.dumps(out))
if (a.run/'status.json').exists():
    r=json.loads((a.run/'status.json').read_text())
    print(json.dumps({k:r[k] for k in ['timesteps','mean_reward','speed','distance','survival_fraction','upright']}))
