"""Prospective matched-seed odor-response acceptance, retaining wrong-way turns."""
import numpy as np
import math


def paired_statistics(values,sides):
    values=np.asarray(values,dtype=float);sides=np.asarray(sides)
    if values.ndim!=1 or not np.isfinite(values).all():raise ValueError('Finite effects required')
    groups=[np.flatnonzero(sides==s) for s in ('loom_left','loom_right')]
    rng=np.random.default_rng(8200)
    samples=np.concatenate([rng.choice(g,size=(10000,len(g)),replace=True) for g in groups],axis=1)
    # Compensated sums avoid a spurious positive interval when equal and
    # opposite effects cancel mathematically. Sampling/thresholds are unchanged.
    means=np.fromiter((math.fsum(values[j] for j in sample)/len(values) for sample in samples),dtype=float)
    interval=np.quantile(means,[.0125,.9875]).tolist()
    return {'mean':math.fsum(values)/len(values),'ci_97_5':interval,'repetitions':10000,
            'seed':8200,'stratified_by_cue':True,'positive':interval[0]>0,
            'summation':'Compensated math.fsum'}


def summarize(effects,replay_verified=False):
    if len(effects)!=20 or len({e['seed'] for e in effects})!=20:
        raise ValueError('Twenty independent matched seeds required')
    if any(e['cue'] not in ('odor_left','odor_right') for e in effects):raise ValueError('Registered cue required')
    sides=['loom_left' if e['cue']=='odor_left' else 'loom_right' for e in effects]
    if sides.count('loom_left')!=10 or sides.count('loom_right')!=10:
        raise ValueError('Ten seeds per registered cue required')
    keys=('motor_rms_vs_sensory_off','motor_rms_vs_steering_cut',
          'toward_heading_vs_sensory_off','toward_heading_vs_steering_cut')
    statistics={key:paired_statistics([e[key] for e in effects],sides) for key in keys}
    counts={'toward_at_least_0_05_rad':sum(e['toward_heading_vs_sensory_off']>=.05 for e in effects),
        'departure_at_least_0_1_mm':sum(e['final_xy_departure_mm']>=.1 for e in effects),
        'no_flip':sum(e['flipped_frames']==0 for e in effects)}
    gates={'causal_motor_effect':all(statistics[k]['positive'] for k in keys[:2]),
        'directional_orientation':all(statistics[k]['positive'] for k in keys[2:]),
        'useful_response':counts['toward_at_least_0_05_rad']>=16 and counts['departure_at_least_0_1_mm']>=16 and counts['no_flip']>=18,
        'exact_motor_replay':bool(replay_verified)}
    gates['progression_ready']=all(gates.values())
    return {'statistics':statistics,'counts':counts,'gates':gates}
