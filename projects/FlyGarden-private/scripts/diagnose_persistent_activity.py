"""Stage 2: full-network causal activity diagnostics, isolated from the arena.

No production edits or automatic parameter promotion. Trials and source hashes
are committed before running. Events are written in bounded 100ms chunks.
"""
import argparse
import json
import resource
import shutil
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.audit_brain_reference import sha
from flygarden.recording import atomic_json, space_check

SEEDS = (2101, 2102, 2199)
DT = .1
PROFILES = {
    'current': {'refractory': 'all_inputs_zero', 'jump_mV': 68.75},
    'selected_refractory': {'refractory': 'selected_inputs_zero', 'jump_mV': 68.75},
    'ordinary_refractory': {'refractory': 'all_2.2ms', 'jump_mV': 68.75},
    'jump_20pct': {'refractory': 'all_inputs_zero', 'jump_mV': 13.75},
    'jump_10pct': {'refractory': 'all_inputs_zero', 'jump_mV': 6.875},
    'recurrent_off_after_pulse': {'refractory': 'all_inputs_zero', 'jump_mV': 68.75,
                                  'disable_recurrence_at': .5},
}


def conditions():
    base = {'quiet': [0]*8, 'odor_a': [.5,0,0,0,0,0,0,0],
            'odor_b': [0,.5,0,0,0,0,0,0], 'walking': [0,0,.65,.65,0,0,0,0],
            'loom_left': [0,0,0,0,.5,0,0,0],
            'combined': [.4,.2,.65,.65,0,0,0,0], 'switch_ab': [.5,0,0,0,0,0,0,0]}
    rows = []
    for profile in PROFILES:
        names = list(base) if profile == 'current' else ['odor_a', 'odor_b', 'combined']
        if profile.startswith('jump_'): names = ['odor_a', 'combined']
        for name in names:
            rows.append({'profile': profile, 'cue': name, 'values': base[name], 'duration': 2.})
    for rate in (5,25,100):
        for cue in ('odor_a', 'odor_b'):
            values = [0]*8; values[0 if cue == 'odor_a' else 1] = rate/100
            rows.append({'profile': 'current', 'cue': f'{cue}_{rate}hz', 'values': values, 'duration': 2.})
    # A longer recovery checks whether a 1.5-second plateau was just a transient.
    for profile in ('current', 'selected_refractory', 'recurrent_off_after_pulse'):
        rows.append({'profile': profile, 'cue': 'odor_a_long', 'values': base['odor_a'], 'duration': 6.})
    return rows


def values_at(condition, tick):
    if 2 <= tick < 5: return condition['values']
    if condition['cue'] == 'switch_ab' and 5 <= tick < 8:
        return [0,.5,0,0,0,0,0,0]
    return [0]*8


class IncomingObserver:
    """One-way synaptic delivery counters, honoring Brian's refractory gate.

    Summed voltage increments are not membrane currents or biological E/I balance.
    This observer reads spikes/state and never writes to the modeled neurons.
    """
    def __init__(self, brain, selected):
        import brian2 as b
        self.group = brain.neurons
        self.selected = np.asarray(selected,dtype=np.int32)
        # Separate readout arrays avoid linked-variable indexing inside Synapses.
        # Core v/g/rfc and spike logic never read these observational variables.
        for name in ('diag_exc','diag_inh','diag_external'):
            self.group.variables.add_array(name, size=len(brain.neurons), dimensions=b.volt.dim)
        post = np.asarray(brain.synapses.j[:], dtype=np.int32)
        mask = np.isin(post, selected)
        edges = np.flatnonzero(mask)
        target = post[mask].copy()
        pre = np.asarray(brain.synapses.i[edges], dtype=np.int32)
        self.recurrent = b.Synapses(brain.neurons, self.group, 'weight : volt',
            on_pre='diag_exc_post += clip(weight, 0*mV, inf*mV) * int(not_refractory_post)\ndiag_inh_post += clip(-weight, 0*mV, inf*mV) * int(not_refractory_post)',
            delay=1.8*b.ms, clock=brain.clock)
        self.recurrent.connect(i=pre,j=target)
        self.recurrent.weight = brain.synapses.w[edges]
        self.external = b.Synapses(brain.input, self.group,
            on_pre='diag_external_post += diagnostic_jump * int(not_refractory_post)',
            namespace={'diagnostic_jump': 68.75*b.mV}, clock=brain.clock)
        pairs = [(i,int(n)) for i,n in enumerate(brain.input_indices) if n in selected]
        self.external.connect(i=np.array([i for i,j in pairs],dtype=np.int32), j=np.array([j for i,j in pairs],dtype=np.int32))
        self.objects = (self.recurrent, self.external)

    def counters(self):
        import brian2 as b
        return np.stack([np.asarray(getattr(self.group, name)[self.selected]/b.mV) for name in ('diag_exc','diag_inh','diag_external')])


def select(brain):
    import pandas as pd
    ann = pd.read_csv(ROOT/'data/annotations.tsv', sep='\t', low_memory=False).fillna('')
    ann = ann[ann.root_id.isin(brain.ids)].copy()
    pops = {k:v.copy() for k,v in brain.populations.items()}
    for cell in ('DM1_lPN','DM2_lPN','APL','DNa02'):
        pops[cell] = np.array([brain.index[int(i)] for i in ann.loc[ann.cell_type.eq(cell),'root_id']], dtype=np.int32)
    selected = []
    for name in ('odor_a','odor_b','DM1_lPN','DM2_lPN','APL','kcg','mbon_reward',
                 'mbon_aversive','pam_gamma5','ppl_gamma1','p9_left','p9_right','DNa02','loom_left','loom_right'):
        selected.extend(pops[name][:3].tolist())
    selected = np.array(sorted(set(selected)), dtype=np.int32)
    by_root = ann.set_index('root_id')
    metadata = [{'index': int(i), 'root_id': str(brain.ids[i]),
                 'cell_type': by_root.loc[int(brain.ids[i]),'cell_type'],
                 'side': by_root.loc[int(brain.ids[i]),'side']} for i in selected]
    return selected, pops, metadata


def instrumentation_check(brain, observer, state_monitor):
    """Same full network/state/input: adding counters must not change outcomes."""
    results = []
    for enabled in (False, True):
        brain.reset_transient(2100)
        for obj in observer.objects: obj.active = enabled
        state_monitor.active = enabled
        counts = np.zeros(brain.n, dtype=np.int64)
        events = []
        for step in range(2):
            brain.advance(DT, odor=(.5,0))
            counts += brain.last_delta
            events.append(tuple(a.copy() for a in brain.last_spikes))
        results.append((counts, events, np.asarray(brain.neurons.v[:]).copy(), np.asarray(brain.neurons.g[:]).copy()))
    checks = {'counts': np.array_equal(results[0][0], results[1][0]),
              'events': all(np.array_equal(a,b) for a_window,b_window in zip(results[0][1],results[1][1]) for a,b in zip(a_window,b_window)),
              'v': np.allclose(results[0][2],results[1][2],rtol=0,atol=1e-13),
              'g': np.allclose(results[0][3],results[1][3],rtol=0,atol=1e-13)}
    for obj in observer.objects: obj.active = True
    state_monitor.active = True
    if not all(checks.values()): raise RuntimeError(f'Instrumentation changes outcome: {checks}')
    return {k: bool(v) for k,v in checks.items()}


def summarize(folder, brain, condition, seed, records, counts, selected, cumulative_input):
    phases = np.array([row['phase'] for row in records])
    times = np.array([row['time'] for row in records])
    stimulation = (times > .2+1e-9) & (times <= .5+1e-9)
    tail = times > condition['duration']-.5+1e-9
    firing = counts/DT
    tail_rate = firing[tail].mean(axis=0)
    high = np.argsort(tail_rate)[-256:][::-1]
    high = high[tail_rate[high] > 0]
    cv = np.std(firing[tail][:,high],axis=0)/np.maximum(np.mean(firing[tail][:,high],axis=0),1e-12) if len(high) else np.array([])
    stim_total = float(counts[stimulation].sum()/sum(stimulation)/DT)
    tail_total = float(counts[tail].sum()/sum(tail)/DT)
    result = {'profile': condition['profile'], 'cue': condition['cue'], 'seed': seed,
              'duration': condition['duration'], 'values': condition['values'],
              'spikes': int(counts.sum()), 'external_events': cumulative_input,
              'stimulation_spikes_per_second': stim_total,
              'tail_spikes_per_second': tail_total,
              'tail_to_stim_ratio': tail_total/stim_total if stim_total else None,
              'tail_active_neurons': int((tail_rate>0).sum()),
              'top256_tail_median_cv': float(np.median(cv)) if len(cv) else None,
              'top256_never_silent_fraction': float(np.mean(np.all(firing[tail][:,high]>0,axis=0))) if len(high) else None,
              'top_neurons': [{'root_id': str(brain.ids[i]), 'rate_hz': float(tail_rate[i])} for i in high[:20]],
              'finite_state': all(row['finite'] for row in records),
              'plasticity_updates': brain.plasticity.updates,
              'stim_rates': {key: float(firing[stimulation][:,pop].mean()) if len(pop) else None for key,pop in brain.populations.items()},
              'tail_rates': {key: float(tail_rate[pop].mean()) if len(pop) else None for key,pop in brain.populations.items()},
              'folder': str(folder.name)}
    atomic_json(folder/'summary.json', result)
    return result


def run_trial(root, brain, observer, state_monitor, selected, pops, condition, seed):
    import brian2 as b
    profile = PROFILES[condition['profile']]
    folder = root/'trials'/f"{condition['profile']}-{condition['cue']}-{seed}"
    space_check(root, 32*1024**2)
    folder.mkdir(parents=True, exist_ok=False)
    brain.reset_transient(seed)
    brain.plasticity.enabled = False
    brain.synapses.active = True
    for obj in observer.objects: obj.active = True
    brain.neurons.rfc = 2.2*b.ms
    if profile['refractory'] == 'all_inputs_zero':
        brain.neurons.rfc[brain.input_indices] = 0*b.ms
    elif profile['refractory'] == 'selected_inputs_zero':
        used = np.array(condition['values']) > 0
        if condition['cue'] == 'switch_ab': used[1] = True
        brain.neurons.rfc[brain.input_indices[np.concatenate([np.full(len(brain.populations[k]), use) for k,use in zip(brain.input_keys,used)])]] = 0*b.ms
    brain.inputs.namespace['diagnostic_jump'] = profile['jump_mV']*b.mV
    observer.external.namespace['diagnostic_jump'] = profile['jump_mV']*b.mV
    before_input = np.zeros(len(brain.input_indices),dtype=np.int64)
    records, count_bins = [], []
    counters = observer.counters()
    started = time.perf_counter()
    updates = brain.plasticity.updates
    meta = {'status': 'running', 'condition': condition, 'seed': seed,
            'profile': profile, 'learning': False, 'chunks': [], 'frames': 0,
            'clock_seconds': .0001, 'bin_seconds': DT,
            'state_sampling_seconds': .001, 'scope': 'Isolated brain diagnostic; no body, route or game score.'}
    atomic_json(folder/'manifest.json', meta)
    for tick in range(round(condition['duration']/DT)):
        values = values_at(condition, tick)
        at = profile.get('disable_recurrence_at')
        if at is not None and tick*DT >= at-1e-9:
            brain.synapses.active = False
            observer.recurrent.active = False
        state_monitor.resize(0)
        brain.advance(DT, odor=values[:2], p9=values[2:4], loom=values[4:6])
        indices, times = brain.last_spikes
        delta = brain.last_delta.copy()
        if len(indices) != int(delta.sum()): raise RuntimeError('Raw spikes and monitor count diverge')
        input_counts = np.asarray(brain.diagnostic_input.count[:],dtype=np.int64)
        input_delta = input_counts-before_input; before_input=input_counts.copy()
        post_counters = observer.counters(); delivered = post_counters-counters; counters=post_counters
        stamps = np.asarray(state_monitor.t[:]/b.second).copy()
        voltage = np.asarray(state_monitor.v[:]/b.mV).copy()
        synaptic = np.asarray(state_monitor.g[:]/b.mV).copy()
        gate = np.asarray(state_monitor.not_refractory[:]).copy()
        phase = 'baseline' if tick<2 else 'stimulus_a' if tick<5 else 'stimulus_b' if condition['cue']=='switch_ab' and tick<8 else 'recovery'
        rfc = np.asarray(brain.neurons.rfc[:]/b.second)
        ordinary = rfc>0
        near_limit = ordinary & (delta/DT >= .8/np.maximum(rfc,.0001))
        finite = bool(np.isfinite(brain.neurons.v[:]).all() and np.isfinite(brain.neurons.g[:]).all())
        row = {'time': (tick+1)*DT, 'phase': phase, 'input_values': values,
               'external_events': int(input_delta.sum()), 'spikes': len(indices),
               'active_neurons': int((delta>0).sum()), 'near_refractory_limit_neurons': int(near_limit.sum()),
               'population_rates': {k: float(delta[v].mean()/DT) if len(v) else None for k,v in pops.items()},
               'selected_counts': delta[selected].tolist(), 'motor': brain.motor.tolist(),
               'delivered_exc_mV': delivered[0].tolist(), 'delivered_inh_mV': delivered[1].tolist(),
               'delivered_external_mV': delivered[2].tolist(),
               'selected_mean_v_mV': voltage.mean(axis=1).tolist(),
               'selected_mean_g_mV': synaptic.mean(axis=1).tolist(),
               'selected_refractory_fraction': (1-gate.mean(axis=1)).tolist(), 'finite': finite,
               'recurrence_enabled': brain.synapses.active}
        file = folder/f'window-{tick:04d}.npz'
        space_check(folder, indices.nbytes + times.nbytes + voltage.nbytes + synaptic.nbytes + delta.nbytes + 1024**2)
        temporary = file.with_suffix('.npz.tmp')
        with temporary.open('wb') as stream:
            np.savez_compressed(stream, i=indices, t=times, counts=delta,
                selected=selected, state_times=stamps, v_mV=voltage, g_mV=synaptic, not_refractory=gate,
                input_counts=input_delta, delivered_mV=delivered)
        temporary.replace(file)
        meta['chunks'].append({'file': file.name, 'sha256': sha(file), 'spikes': len(indices), 'end': (tick+1)*DT})
        meta['frames'] += 1
        records.append(row); count_bins.append(delta)
        atomic_json(folder/'bins.json', records)
        atomic_json(folder/'manifest.json', meta)
        if not finite: raise RuntimeError('Non-finite neural state')
    count_bins = np.stack(count_bins)
    if brain.plasticity.updates != updates: raise RuntimeError('Plasticity changed during frozen diagnostic')
    summary = summarize(folder,brain,condition,seed,records,count_bins,selected,int(before_input.sum()))
    summary['wall_seconds'] = time.perf_counter()-started
    summary['weights_unchanged'] = np.array_equal(brain.plasticity.weights,brain.plasticity.base)
    atomic_json(folder/'summary.json',summary)
    meta.update(status='complete',wall_seconds=summary['wall_seconds'],spikes=summary['spikes'])
    atomic_json(folder/'manifest.json', meta)
    return summary


def run(root):
    import brian2 as b
    from flygarden.brain import FullBrain
    if root.exists(): raise FileExistsError('Use a fresh diagnostic evidence folder')
    space_check(root.parent,64*1024**2)
    root.mkdir(parents=True)
    sources = {str(p.relative_to(ROOT)): sha(p) for p in
               [ROOT/'flygarden/brain.py', ROOT/'flygarden/engine.py', ROOT/'scripts/diagnose_persistent_activity.py',
                ROOT/'reports/provenance.json', ROOT/'reports/neuron-mapping.json']}
    protocol = {'schema_version':1,'created':time.time(),'seeds':list(SEEDS),
                'seed_policy':'2101/2102 diagnostic calibration, 2199 prospectively held out from them; no automatic promotion.',
                'profiles':PROFILES,'conditions':conditions(),'sources':sources,'learning':False,
                'normal_duration_seconds':2.,'baseline_seconds':.2,'pulse_seconds':.3,
                'recovery_seconds':1.5,'long_recovery_seconds':5.5,'neural_clock_seconds':.0001,
                'alterations':'Diagnostic worker objects only; all neurons and connections imported. Recurrence-off is an explicitly labeled lesion, never an interactive substitute.',
                'criteria':{'instrumentation':'Exact spike counts/events and matching final neural state with observer enabled vs disabled.',
                            'persistence':'Report last 500ms network firing vs stimulation, not a consciousness/physiology score.',
                            'sensory_discrimination':'Paired A/B population rate vectors, input/output response and readout saturation; no choice claim.',
                            'promotion':'None. Lower gain/rate or refractory variants are calibration probes, not validated biology.'}}
    atomic_json(root/'protocol.json',protocol)
    progress={'status':'building','trials':[],'planned_trials':len(conditions())*len(SEEDS)}
    atomic_json(root/'progress.json',progress)
    print(f'Building full network; {progress["planned_trials"]} predeclared trials',flush=True)
    started=time.perf_counter();brain=FullBrain(seed=2100,learning=False)
    selected,pops,metadata=select(brain)
    observer=IncomingObserver(brain,selected)
    state=b.StateMonitor(brain.neurons,['v','g','not_refractory'],record=selected,dt=.001*b.second,when='end')
    brain.diagnostic_input=b.SpikeMonitor(brain.input,record=False)
    brain.network.add(*observer.objects,state,brain.diagnostic_input)
    brain.inputs.pre.code='v_post += diagnostic_jump'
    brain.inputs.namespace['diagnostic_jump']=68.75*b.mV
    brain.mark_initial()
    atomic_json(root/'neuron-selection.json',{'selected':metadata,'populations':{k:[str(brain.ids[i]) for i in indices] for k,indices in pops.items()},
        'coverage':{'neurons':brain.n,'connections':brain.edges,'anatomical_synapses':brain.anatomical_synapses},
        'selection':'First three annotated cells per selected population for internal-state inspection; per-neuron spike counts cover the entire network.'})
    checks=instrumentation_check(brain,observer,state)
    atomic_json(root/'instrumentation-check.json',checks)
    print('Full-network instrumentation check passed',flush=True)
    progress['status']='running'
    for condition in conditions():
        for seed in SEEDS:
            progress['current']={**condition,'seed':seed};atomic_json(root/'progress.json',progress)
            trial=run_trial(root,brain,observer,state,selected,pops,condition,seed)
            progress['trials'].append(trial)
            progress['wall_seconds']=time.perf_counter()-started
            progress['peak_rss_bytes']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            atomic_json(root/'progress.json',progress)
            print(f"{len(progress['trials'])}/{progress['planned_trials']} {condition['profile']} {condition['cue']} seed {seed}: tail {trial['tail_spikes_per_second']:.0f} spikes/s; {trial['wall_seconds']:.1f}s wall",flush=True)
    unchanged={name:sha(ROOT/name)==digest for name,digest in sources.items()}
    if not all(unchanged.values()):raise RuntimeError('Diagnostic sources changed during run')
    progress.update(status='complete',sources_unchanged=unchanged)
    atomic_json(root/'progress.json',progress)
    print(f'Completed {root}',flush=True)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--folder',type=Path)
    args=parser.parse_args()
    root=args.folder or ROOT/'reports/brain-integration'/time.strftime('stage2-%Y%m%d-%H%M%S')
    try:run(root)
    except Exception as error:
        if root.exists():
            atomic_json(root/'failure.json',{'error':str(error),'time':time.time(),'status':'interrupted' if isinstance(error,OSError) else 'failed',
                                          'preservation':'Completed windows/trials retained; no automatic deletion.'})
        raise


if __name__=='__main__':main()
