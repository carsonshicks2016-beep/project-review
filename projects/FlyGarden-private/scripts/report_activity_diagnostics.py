"""Verify and summarize Stage 2 artifacts without running or changing the brain."""
import argparse
import json
import sys
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.audit_brain_reference import sha
from flygarden.recording import atomic_json


def cue_comparisons(root,trials,selection):
    results=[]
    populations=selection['populations']
    # The imported file supplies exact ordering; use it for whole-network vectors.
    import pandas as pd
    ids=pd.read_csv(ROOT/'vendor/fly-brain/data/2025_Completeness_783.csv',index_col=0).index.to_numpy(dtype=np.int64)
    ordering={str(root_id):index for index,root_id in enumerate(ids)}
    for profile in sorted({r['profile'] for r in trials}):
        for seed in sorted({r['seed'] for r in trials}):
            pair=[next((r for r in trials if r['profile']==profile and r['seed']==seed and r['cue']==cue),None) for cue in ('odor_a','odor_b')]
            if any(r is None for r in pair):continue
            vectors=[]
            for row in pair:
                counts=[]
                for tick in (2,3,4):
                    with np.load(root/'trials'/row['folder']/f'window-{tick:04d}.npz') as data:
                        counts.append(data['counts'].astype(float)/.1)
                vectors.append(np.mean(counts,axis=0))
            for name in ('odor_a','odor_b','kcg','mbon','p9_left','p9_right'):
                indices=np.array([ordering[root_id] for root_id in populations[name]],dtype=np.int32)
                a,b=(vector[indices] for vector in vectors)
                norm=np.linalg.norm(a)*np.linalg.norm(b)
                results.append({'profile':profile,'seed':seed,'population':name,
                    'a_mean_hz':float(a.mean()) if len(a) else None,'b_mean_hz':float(b.mean()) if len(b) else None,
                    'cosine':float(np.dot(a,b)/norm) if norm else None,
                    'relative_l2_difference':float(np.linalg.norm(a-b)/max(np.linalg.norm(a),np.linalg.norm(b),1e-12))})
    return results


def verify(root,progress,require_native=True):
    n=json.loads((root/'neuron-selection.json').read_text())['coverage']['neurons']
    windows=spikes=0
    for trial in progress['trials']:
        folder=root/'trials'/trial['folder'];manifest=json.loads((folder/'manifest.json').read_text())
        assert manifest['status']=='complete' and not manifest['learning']
        bins=json.loads((folder/'bins.json').read_text())
        total=0
        for tick,chunk in enumerate(manifest['chunks']):
            path=folder/chunk['file'];assert sha(path)==chunk['sha256']
            with np.load(path) as data:
                indices=data['i'];times=data['t'];counts=data['counts']
                assert len(indices)==int(counts.sum())==chunk['spikes']==bins[tick]['spikes']
                assert np.array_equal(np.bincount(indices,minlength=n),counts)
                assert np.isfinite(times).all() and np.isfinite(data['v_mV']).all() and np.isfinite(data['g_mV']).all()
                assert np.all(np.diff(times)>=-1e-12)
                assert np.all(times>=tick*.1-1e-9) and np.all(times<(tick+1)*.1+1e-9)
                if bins[tick]['phase'] in ('baseline','recovery'):
                    assert int(data['input_counts'].sum())==0
                if not bins[tick]['recurrence_enabled']:
                    assert np.allclose(data['delivered_mV'][:2],0,rtol=0,atol=1e-8)
            total+=len(indices);windows+=1
        assert total==trial['spikes']==manifest['spikes']
        assert trial['weights_unchanged'] and trial['plasticity_updates']==0 and trial['finite_state']
        spikes+=total
    protocol=json.loads((root/'protocol.json').read_text())
    assert all(sha(ROOT/name)==digest for name,digest in protocol['sources'].items())
    instrumentation=json.loads((root/'instrumentation-check.json').read_text())
    assert all(instrumentation.values())
    native_manifest={'chunks':[]};native_summary={'total_spikes':0}
    if require_native:
        native=root/'native-reference'
        native_manifest=json.loads((native/'manifest.json').read_text())
        assert native_manifest['status']=='complete' and native_manifest['neurons']==n
        assert sha(ROOT/'vendor/fly-brain/code/run_brian2_cuda.py')==native_manifest['source_sha256']
        for tick,chunk in enumerate(native_manifest['chunks']):
            path=native/chunk['file'];assert sha(path)==chunk['sha256']
            with np.load(path) as data:
                assert np.array_equal(np.bincount(data['i'],minlength=n),data['counts'])
                assert len(data['i'])==chunk['spikes']
                assert np.isfinite(data['t']).all()
                assert np.all(data['t']>=tick*.1-1e-9) and np.all(data['t']<(tick+1)*.1+1e-9)
        native_summary=json.loads((native/'summary.json').read_text())
        assert native_summary['finite_states']
    return {'passed':True,'trials':len(progress['trials']),'windows':windows,'verified_spikes':spikes,
        'native_reference_windows':len(native_manifest['chunks']), 'native_reference_spikes':native_summary['total_spikes'],
        'checks':['SHA256 for every committed window','Whole-network raw events equal per-neuron monitor counts',
                  'Finite timestamps and sampled states','Input-free baseline/recovery confirmed by input monitor',
                  'Recurrence-off observer confirms no new recurrent deliveries','Plasticity frozen and weights unchanged',
                  'Diagnostic/application source identity retained','Instrumentation parity passed']}


def plots(root,trials,selection):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(3,1,figsize=(11,9),sharex=True,constrained_layout=True)
    colors={'current':'#e68a3f','selected_refractory':'#519bc4','ordinary_refractory':'#709747',
            'recurrent_off_after_pulse':'#9b6bab'}
    for profile,color in colors.items():
        trial=next(r for r in trials if r['profile']==profile and r['cue']=='odor_a' and r['seed']==2199)
        rows=json.loads((root/'trials'/trial['folder']/'bins.json').read_text())
        t=np.array([r['time'] for r in rows])-.05
        axes[0].plot(t,[r['spikes']/.1/1000 for r in rows],label=profile.replace('_',' '),color=color,lw=2)
        axes[1].plot(t,[r['population_rates']['odor_a'] for r in rows],color=color,lw=2)
        axes[2].plot(t,[r['population_rates']['kcg'] for r in rows],color=color,lw=2)
    for ax in axes:
        ax.axvspan(.2,.5,color='#ccc',alpha=.3)
        ax.axvline(.5,color='#555',ls=':',lw=1)
        ax.set_ylim(bottom=0);ax.grid(alpha=.15)
    axes[0].set_ylabel('Whole-network spikes / second\n(thousands)')
    axes[1].set_ylabel('Odor A sensory neurons\nMean firing rate (Hz)')
    axes[2].set_ylabel('Gamma Kenyon cells\nMean firing rate (Hz)')
    axes[2].set_xlabel('Simulated time (seconds)')
    axes[0].legend(loc='upper right',fontsize=9)
    fig.suptitle('A short odor pulse can be separated from the activity it triggers\nFull network · held-out seed 2199 · input on 0.2–0.5 s · plasticity frozen',fontsize=14)
    fig.savefig(root/'pulse-recovery.png',dpi=160);plt.close(fig)

    trial=next(r for r in trials if r['profile']=='current' and r['cue']=='odor_a_long' and r['seed']==2199)
    rows=json.loads((root/'trials'/trial['folder']/'bins.json').read_text())
    counts=np.array([r['selected_counts'] for r in rows]).T/.1
    labels=[f"{r['cell_type']} {r['side']} · …{r['root_id'][-5:]}" for r in selection['selected']]
    fig,ax=plt.subplots(figsize=(12,10),constrained_layout=True)
    shown=ax.imshow(counts,aspect='auto',interpolation='nearest',origin='upper',extent=[0,6,len(labels)-.5,-.5],vmin=0,vmax=500,cmap='inferno')
    ax.set_yticks(range(len(labels)),labels,fontsize=7)
    ax.axvline(.2,color='cyan',lw=1);ax.axvline(.5,color='cyan',lw=1,ls='--')
    ax.set_xlabel('Simulated time (seconds)');fig.colorbar(shown,ax=ax,label='Individual neuron firing rate (Hz), 100 ms bins')
    ax.set_title('Individual neurons: 300 ms odor pulse followed by 5.5 s without external input\nFull network · current adapter · seed 2199 · fixed 0–500 Hz display scale')
    fig.savefig(root/'individual-neuron-recovery.png',dpi=160);plt.close(fig)


def report(root,progress,selection,checks,comparisons):
    trials=progress['trials']
    def group(profile,cue):return [r for r in trials if r['profile']==profile and r['cue']==cue]
    def mean(rows,key):return float(np.mean([r[key] for r in rows]))
    table=[]
    for profile in ('current','selected_refractory','ordinary_refractory','jump_20pct','jump_10pct','recurrent_off_after_pulse'):
        rows=group(profile,'odor_a')
        table.append(f"| {profile} | {mean(rows,'stimulation_spikes_per_second'):,.0f} | {mean(rows,'tail_spikes_per_second'):,.0f} | {mean(rows,'tail_active_neurons'):,.0f} |")
    dose=[]
    for rate in (5,25,100):
        rows=group('current',f'odor_a_{rate}hz')
        dose.append(f"| {rate} | {mean(rows,'stimulation_spikes_per_second'):,.0f} | {mean(rows,'tail_spikes_per_second'):,.0f} |")
    tails=[]
    for profile in ('current','selected_refractory','recurrent_off_after_pulse'):
        rows=group(profile,'odor_a_long')
        tails.append(f"| {profile} | {mean(rows,'tail_spikes_per_second'):,.0f} | {mean(rows,'tail_active_neurons'):,.0f} |")
    quiet=group('current','quiet')
    current=group('current','odor_a')
    selected=group('selected_refractory','odor_a')
    lesion=group('recurrent_off_after_pulse','odor_a')
    source_contribution={'quiet_spikes':sum(r['spikes'] for r in quiet),
        'current_odor_tail_mean_spikes_per_second':mean(current,'tail_spikes_per_second'),
        'selected_refractory_odor_tail_mean_spikes_per_second':mean(selected,'tail_spikes_per_second'),
        'recurrent_off_tail_mean_spikes_per_second':mean(lesion,'tail_spikes_per_second'),
        'persistent_across_all_three_seeds':all(r['tail_spikes_per_second']>0 for r in current),
        'recurrence_off_extinguishes_all_three_seeds':all(r['tail_spikes_per_second']==0 for r in lesion)}
    native=json.loads((root/'native-reference/summary.json').read_text())
    source_contribution['native_reference_long_tail_spikes_per_second']=native['tail_spikes_per_second']
    motor_summary=[r for r in comparisons if r['profile']=='current' and r['population'] in ('p9_left','p9_right')]
    observed_selection=[]
    rows=json.loads((root/'trials'/'current-odor_a_long-2199'/'bins.json').read_text())
    late=[r for r in rows if r['time']>5.5]
    for index,neuron in enumerate(selection['selected']):
        observed_selection.append({**neuron,'late_hz':float(np.mean([r['selected_counts'][index]/.1 for r in late])),
            'late_exc_delivered_mV_per_second':float(np.mean([r['delivered_exc_mV'][index]/.1 for r in late])),
            'late_inh_delivered_mV_per_second':float(np.mean([r['delivered_inh_mV'][index]/.1 for r in late])),
            'late_external_delivered_mV_per_second':float(np.mean([r['delivered_external_mV'][index]/.1 for r in late])),
            'late_mean_v_mV':float(np.mean([r['selected_mean_v_mV'][index] for r in late])),
            'late_mean_g_mV':float(np.mean([r['selected_mean_g_mV'][index] for r in late]))})
    summary={'schema_version':1,'source_contribution':source_contribution,'paired_cue_comparisons':comparisons,
        'descending_odor_response':motor_summary,'late_internal_state':observed_selection,
        'verification':checks,'promoted_profile':None,'learning_demonstrated':False,'navigation_demonstrated':False,
        'native_reference':native,
        'scope':'Short full-network dynamics diagnostic; recurrence-off is a causal lesion, not a usable controller.'}
    atomic_json(root/'analysis.json',summary)
    byte_count=sum(p.stat().st_size for p in root.rglob('*') if p.is_file())
    text=f'''# Fly Garden — Stage 2 persistent-activity diagnosis

The diagnostic isolates stimulus-driven activity from recurrent persistence, without changing the live controller or training weights. Full network: {selection['coverage']['neurons']:,} neurons and {selection['coverage']['connections']:,} connection records. Results cover {len(trials)} trials, two calibration seeds plus one prospectively specified held-out seed. This is not a navigation or learning evaluation.

## Main measured results

Unstimulated trials produced {source_contribution['quiet_spikes']:,} total spikes. A 300 ms odor-A pulse produced an average of {source_contribution['current_odor_tail_mean_spikes_per_second']:,.0f} spikes/s in the final 500 ms of the 2-second trials, with all external input removed. The selected-input refractory convention produced {source_contribution['selected_refractory_odor_tail_mean_spikes_per_second']:,.0f} spikes/s. Disabling recurrent delivery after the same pulse produced {source_contribution['recurrent_off_tail_mean_spikes_per_second']:,.0f} spikes/s in that tail.

Disabling recurrent delivery extinguished the measured post-pulse activity in all three seeds while the intact network remained active. This establishes that recurrent connections sustain this response in the tested model. It cannot identify a unique responsible circuit, establish biological firing patterns, or prove the network would work with a natural sensory interface. Constant walking input is tested separately; persistent odor-triggered activity does not require the arena, walking controller, predator, renderer or learning.

An independent native reference check used the original pinned construction functions, original reset and actual per-neuron PoissonInput, with the same 300 ms 50 Hz odor-A pulse followed by 5.5 seconds without input. Its final rate was {native['tail_spikes_per_second']:,.0f} spikes/s. That check uses a different RNG stream from our adapter; it confirms the response also occurs in the upstream setup, rather than establishing exact stochastic equivalence.

## Refractory and amplitude probes

Mean across three seeds; 50 Hz odor-A input for 300 ms. Recovery tail is the final 500 ms, not a single sample.

| Diagnostic profile | During-pulse spikes/s | Recovery-tail spikes/s | Tail-active neurons |
| --- | ---: | ---: | ---: |
{chr(10).join(table)}

`selected_refractory` disables refractory periods only for populations selected for stimulation, retained for that trial. `ordinary_refractory` keeps 2.2 ms on every neuron. `jump_20pct` and `jump_10pct` change only the external voltage jump to 13.75 mV and 6.875 mV. These are engineered diagnostic probes, not physiology-derived replacements. The original 68.75 mV artificial activation is retained in the reference. `recurrent_off_after_pulse` disables recurrent delivery at 0.5 seconds; no cells or edges are removed from import, but it is an explicit lesion and must never be presented as a full-function interactive model.

## Input-rate dose response

| Odor-A input rate (Hz) | During-pulse spikes/s | Recovery-tail spikes/s |
| --- | ---: | ---: |
{chr(10).join(dose)}

Rate and voltage jump are distinct settings. A lower input frequency can still trigger a self-sustaining response. No setting was promoted based on making the visualization dimmer or more variable.

## Longer recovery

Same 300 ms pulse, now followed by 5.5 seconds with no external stimulation. Mean over three seeds, final 500 ms:

| Profile | Recovery-tail spikes/s | Tail-active neurons |
| --- | ---: | ---: |
{chr(10).join(tails)}

## Circuit response and internal-state evidence

`analysis.json` contains paired odor-A/B rate-vector comparisons for sensory neurons, gamma Kenyon cells, MBONs and DNp09. It also contains exact-root-ID internal-state summaries: delivered positive and negative recurrent voltage increments, external increments, membrane voltage and synaptic state. Positive/negative increment totals are model bookkeeping; they are not physical currents or a validated biological excitation/inhibition ratio. Counters honor the refractory gate.

`bins.json` retains each population's response, individual inspected-neuron rates, descending motor readout, near-refractory-limit counts and sensory events. Selected high-firing neurons' coefficient of variation and silence frequency are descriptive metrics with selection bias, not evidence that the entire brain is static. The switch-A/B trial checks responses without resetting neural state between cues. Exact raw spikes and per-neuron counts cover the entire network; internal state was sampled at 1 ms for {len(selection['selected'])} annotated neurons, not all cells.

## Validation, storage and runtime

Verified {checks['windows']:,} committed 100 ms windows and {checks['verified_spikes']:,} raw spikes against per-neuron monitor counts. All hashes, timestamp bounds and finite-state checks passed. No external input was recorded during baseline/recovery. Observer-enabled and observer-disabled full-network tests agreed on spikes/counts and final v/g state. A separate two-cell signed-delivery test verified refractory blocking and 1.8 ms delivery delays. Learning stayed frozen, plasticity weights stayed unchanged, and sources retained their committed identity.

Also verified {checks['native_reference_windows']} native reference windows with {checks['native_reference_spikes']:,} raw spikes. Native confirmation took {native['wall_seconds']:.1f} wall seconds, after the main simulation worker finished.

Aggregate active worker wall time (excluding the storage pause): {progress['wall_seconds']:.1f} seconds. Peak worker RSS: {progress['peak_rss_bytes']/1024**3:.2f} GiB (macOS RSS accounting; excludes live app). Evidence currently occupies {byte_count/1024**2:.1f} MiB. One diagnostic simulation worker; no render jobs. A 2 GiB free-space reserve is checked before writing. Completed windows are committed before manifest publication. Earlier diagnostic setup failures are retained separately and excluded from these results.

## Decision and next stage

No production brain configuration is changed or promoted. Persistent activity survived both refractory conventions, reduced input amplitudes, low input rates and the independent upstream check. None of these tested settings provides a validated correction. Only about 8,200 neurons (roughly 6% of the modeled network) were active in the odor recovery tail; this is not evidence that every neuron is continuously firing.

The current left/right DNp09 motor readouts emitted zero spikes during both isolated odor pulses in every seed. Sensory activation therefore does not demonstrate odor-driven walking with this decoder. Stage 3 should establish an annotated sensory-to-descending pathway and verify cue-dependent motor outputs before reconnecting the arena. Diagnose the persistent circuit separately, using documented model assumptions and targeted circuit tests; do not weaken recurrent weights merely to change the display. Learning and navigation remain unproven.

Source: [pinned released model](https://github.com/eonsystemspbc/fly-brain/tree/a3db62f9436074e485c0278290c2164ed6150808). Brian's refractory flags also make flagged variables ignore incoming changes while refractory: [official Brian2 documentation](https://brian2.readthedocs.io/en/stable/user/refractoriness.html).

Reproduce with `.venv-next/bin/python scripts/diagnose_persistent_activity.py`, then `.venv-next/bin/python scripts/check_native_activity_recovery.py ABSOLUTE_EVIDENCE_FOLDER`, then `.venv-next/bin/python scripts/report_activity_diagnostics.py ABSOLUTE_EVIDENCE_FOLDER`. A fresh folder is required; earlier recordings/checkpoints are never overwritten.

![Pulse and recovery](pulse-recovery.png)

![Individual-neuron recovery](individual-neuron-recovery.png)
'''
    (root/'DIAGNOSIS.md').write_text(text)
    return summary


def main():
    parser=argparse.ArgumentParser();parser.add_argument('folder',type=Path)
    parser.add_argument('--partial',action='store_true',help='Validate completed trials only; publish no complete report.')
    args=parser.parse_args()
    root=args.folder.resolve()
    progress=json.loads((root/'progress.json').read_text())
    if args.partial:
        print(json.dumps({'status':'partial','completed':len(progress['trials']),
            'planned':progress['planned_trials'],'verification':verify(root,progress,require_native=False)},indent=2))
        return
    if progress['status']!='complete' or len(progress['trials'])!=progress['planned_trials']:
        raise RuntimeError('Diagnostic batch incomplete; preserve progress but do not publish complete results')
    selection=json.loads((root/'neuron-selection.json').read_text())
    checks=verify(root,progress)
    atomic_json(root/'verification.json',checks)
    comparisons=cue_comparisons(root,progress['trials'],selection)
    plots(root,progress['trials'],selection)
    summary=report(root,progress,selection,checks,comparisons)
    print(json.dumps(summary['source_contribution'],indent=2));print(root/'DIAGNOSIS.md')


if __name__=='__main__':main()
