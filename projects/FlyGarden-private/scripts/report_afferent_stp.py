"""Audit raw paired trials and apply the prospective recovery prerequisite."""
import sys,json,time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from flygarden.recording import atomic_json
from flygarden.continuous_candidate import file_sha
from flygarden.descending import DescendingDecoder
OUT=ROOT/'reports/brain-integration/recovery/afferent-stp-full-network-v1'
protocol=json.loads((OUT/'protocol.json').read_text());mapping=protocol['mapping']
assert all(file_sha(ROOT/n)==h for n,h in protocol['sources'].items())
import pandas as pd
import brian2 as b
import hashlib
graph=pd.read_parquet(ROOT/'vendor/fly-brain/data/2025_Connectivity_783.parquet',columns=['Excitatory x Connectivity'])
expected_weights=np.asarray((graph['Excitatory x Connectivity'].to_numpy()*.275*b.mV)/b.mV).copy()
expected_weight_hashes={'original':hashlib.sha256(expected_weights.tobytes()).hexdigest()}
aff=json.loads((ROOT/'reports/brain-integration/recovery/olfactory-spiking-mechanism-v1/mapping.json').read_text())
expected_weights[np.array([e['row_position'] for e in aff['afferent_edges']])]=0
expected_weight_hashes['stp_only']=hashlib.sha256(expected_weights.tobytes()).hexdigest()
del graph,expected_weights
population={k:np.array([n['index'] for n in pop],dtype=int) for k,pop in mapping.items()}
summaries=[];pair_checks=[];trajectories={};manifest_collection=[];checkpoint_checks=[]
paired_dynamics=[]
for case in protocol['cases']:
    for seed in protocol['seeds']:
        folder=OUT/'trials'/f"{case['model']}-{case['cue']}-{seed}"
        m=json.loads((folder/'manifest.json').read_text());assert m['status']=='complete'
        assert m['storage_weights_sha256']==m['final_storage_weights_sha256']
        assert m['storage_weights_sha256']==expected_weight_hashes[case['model']]
        assert m['sources']==protocol['sources'];manifest_collection.append(m)
        rows=json.loads((folder/'rows.json').read_text());assert len(rows)==len(m['chunks'])==160
        rng=np.random.default_rng(seed);channels=np.array(m['controller']['input_channels']);targets=m['controller']['input_indices']
        decoder=DescendingDecoder();traces={k:[] for k in mapping}
        count_l1=0;identical_spike_windows=0
        for tick,chunk in enumerate(m['chunks']):
            path=folder/chunk['file'];assert file_sha(path)==chunk['sha256']
            with np.load(path) as z:
                assert np.array_equal(np.bincount(z['spike_i'],minlength=138639),z['counts'])
                assert np.isfinite(z['spike_t']).all() and np.all(z['spike_t']>=tick*.025-1e-9) and np.all(z['spike_t']<(tick+1)*.025+1e-9)
                rates=np.array([0,0,0,0,65,65,0,0.])
                if case['cue']!='none' and 12<=tick<32:rates[0 if case['cue']=='a_left' else 1]=50
                tt,ii=np.nonzero(rng.random((250,len(targets)))<rates[channels]*.0001)
                assert np.array_equal(ii,z['external_i'])
                assert np.allclose(tick*.025+tt*.0001,z['external_t'],rtol=0,atol=1e-12)
                pop={k:float(z['counts'][indices].mean()/.025) for k,indices in population.items()}
                assert all(abs(pop[k]-rows[tick]['population_hz'][k])<1e-10 for k in pop)
                assert np.array_equal(decoder.advance(.025,pop),z['motor'])
                if case['model']=='stp_only':
                    other=OUT/'trials'/f"original-{case['cue']}-{seed}"/chunk['file']
                    with np.load(other) as baseline:
                        assert np.array_equal(z['external_i'],baseline['external_i'])
                        assert np.array_equal(z['external_t'],baseline['external_t'])
                        count_l1+=int(np.abs(z['counts']-baseline['counts']).sum())
                        identical_spike_windows+=int(np.array_equal(z['spike_i'],baseline['spike_i']) and np.array_equal(z['spike_t'],baseline['spike_t']))
                for k in traces:traces[k].append(pop[k])
        traces={k:np.array(v) for k,v in traces.items()}
        traces['pn']=np.mean([traces[k] for k in ('DM1_lPN_left','DM1_lPN_right')],axis=0)
        traces['signed_dna']=traces['DNa02_left']-traces['DNa02_right']
        trajectories[(case['model'],case['cue'],seed)]=traces
        summaries.append({'model':case['model'],'cue':case['cue'],'seed':seed,
                          'pulse_pn_hz':float(traces['pn'][12:32].mean()),
                          'tail_pn_hz':float(traces['pn'][140:160].mean()),
                          'pulse_signed_dna_hz':float(traces['signed_dna'][12:32].mean()),
                          'tail_signed_dna_hz':float(traces['signed_dna'][140:160].mean())})
        if case['model']=='stp_only':paired_dynamics.append({'cue':case['cue'],'seed':seed,
            'whole_network_count_l1_difference':count_l1,'identical_spike_windows':identical_spike_windows,'windows':160})
        if case['cue']=='a_left' and seed==protocol['seeds'][0]:
            c=json.loads((folder/'continuation-result.json').read_text());assert c['status']=='passed';checkpoint_checks.append({'model':case['model'],**c})
            with np.load(folder/'window-0030.npz') as z:
                selected_pre={e['pre_index'] for e in json.loads((ROOT/'reports/brain-integration/recovery/olfactory-spiking-mechanism-v1/mapping.json').read_text())['afferent_edges']}
                pending=int(np.sum(np.isin(z['spike_i'],list(selected_pre)) & (z['spike_t']>=.775-.0018-1e-12)))
            checkpoint_checks[-1]['selected_presynaptic_spikes_in_delay_horizon']=pending
            if case['model']=='stp_only':
                with np.load(folder/'continuation-state.npz') as z:
                    checkpoint_checks[-1]['state_ranges_at_0_9s']={k:[float(z[k].min()),float(z[k].max())] for k in ('x','u')}
for model in ('original','stp_only'):
    for seed in protocol['seeds']:
        none=next(s for s in summaries if s['model']==model and s['seed']==seed and s['cue']=='none')
        for cue in ('a_left','a_right'):
            s=next(s for s in summaries if s['model']==model and s['seed']==seed and s['cue']==cue)
            pn_diff=s['tail_pn_hz']-none['tail_pn_hz'];dna_diff=s['tail_signed_dna_hz']-none['tail_signed_dna_hz'];pulse=s['pulse_pn_hz']-none['pulse_pn_hz']
            pair_checks.append({'model':model,'seed':seed,'cue':cue,'pulse_pn_increase_hz':pulse,
                                'tail_pn_difference_hz':pn_diff,'tail_signed_dna_difference_hz':dna_diff,
                                'recovery_prerequisite_passed':abs(pn_diff)<=5 and abs(dna_diff)<=5 and pulse>=10})
qualified=all(c['recovery_prerequisite_passed'] for c in pair_checks if c['model']=='stp_only')
results={'status':'complete','trial_count':12,'raw_chunks_audited':1920,'raw_spike_counts_match':True,
         'seeded_external_events_reconstructed':True,'paired_external_events_exact':True,
         'decoder_reconstruction_exact':True,'storage_weights_unchanged_during_trials':True,
         'all_nonselected_weights_match_source_graph':True,
         'checkpoints':checkpoint_checks,'summary':summaries,'recovery_checks':pair_checks,
         'paired_dynamics':paired_dynamics,
         'stp_only_recovery_prerequisite_passed':qualified,'controller_promoted':False,
         'navigation_validated':False,'learning_validated':False,
         'runtime_trial_seconds':sum(m['wall_seconds'] for m in manifest_collection),
         'peak_worker_rss_bytes':max(m['peak_rss_bytes'] for m in manifest_collection),
         'storage_bytes':sum(p.stat().st_size for p in OUT.rglob('*') if p.is_file()),
         'source_sha256':file_sha(Path(__file__))}
atomic_json(OUT/'results.json',results)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
fig,axes=plt.subplots(2,2,figsize=(12,7),sharex=True)
t=(np.arange(160)+1)*.025
for col,cue in enumerate(('a_left','a_right')):
    for model,color in [('original','#b45240'),('stp_only','#276ea6')]:
        for row,key in enumerate(('pn','signed_dna')):
            series=np.array([trajectories[(model,cue,s)][key] for s in protocol['seeds']])
            axes[row,col].plot(t,series.mean(axis=0),color=color,label=model)
            axes[row,col].fill_between(t,series.min(axis=0),series.max(axis=0),color=color,alpha=.13)
            axes[row,col].axvspan(.3,.8,color='grey',alpha=.12);axes[row,col].grid(alpha=.2)
    axes[0,col].set_title(cue.replace('_',' '));axes[1,col].set_xlabel('Simulated seconds');axes[0,col].legend()
axes[0,0].set_ylabel('DM1 projection-neuron mean Hz');axes[1,0].set_ylabel('DNa02 left minus right Hz')
fig.suptitle('Full-network STP-only hypothesis — two seeds; shading is their range, not confidence')
fig.tight_layout();fig.savefig(OUT/'comparison.png',dpi=160);plt.close(fig)
table='\n'.join(f"| {c['model']} | {c['seed']} | {c['cue']} | {c['pulse_pn_increase_hz']:.1f} | {c['tail_pn_difference_hz']:.1f} | {c['tail_signed_dna_difference_hz']:.1f} | {'pass' if c['recovery_prerequisite_passed'] else 'fail'} |" for c in pair_checks)
(OUT/'RESULTS.md').write_text(f'''# Full-network afferent STP-only comparison

Completed 12 four-second full-network trials and fresh-process checkpoint
continuation for both model identities. The recovery prerequisite for STP-only
{'passed' if qualified else 'failed'}. No controller was promoted and the application remains unchanged.

The engineered variant gives independent release-resource state to 351 exact
cognate DM1/DM2 ORN-to-lPN aggregated records, using generic published U=.24,
tauD=.1s and tauF=.05s. The rested first delivery is normalized to the original
voltage weight. Presynaptic inhibition is absent. This normalization, aggregated
per-edge state and parameter application are engineered assumptions, not validated
DM1/DM2 physiology. All 138,639 neurons and 15,091,983 effective graph records
remain represented; 351 original storage weights are zero and their matching
stateful synapses provide the effective delivery. Nonselected weights and decoder
remain fixed; learning is disabled.

## Prospective recovery screen

Compare each cue with same-model/same-seed support-only control. Recovery requires
both tail differences within 5Hz and a pulse PN increase of at least10Hz in every
STP-only case. This two-seed screen is a prerequisite, not statistical validation.
The PN measure is the mean over the two mapped DM1 lPNs.

| Model | Seed | Cue | Pulse PN increase Hz | Final PN difference Hz | Final signed DNa difference Hz | Gate |
| --- | ---: | --- | ---: | ---: | ---: | --- |
{table}

## Verification and limits

All1,920 chunk hashes, raw spike-count matches, population rates, input RNG
reconstructions and fixed decoder reconstructions passed. Both variants received
exactly identical external events for each matched cue/seed. Stored base weights
did not change during trials; only the STP release multiplier changed.
The two support-only recordings have identical spikes and counts in all160
windows. Pulse recordings differ substantially across the full network, so the
mechanism was active; that changed activity did not meet the recovery prerequisite.

Each checkpoint was saved during stimulation at .775s, then reopened in a new
process. The next five windows matched spikes, counts, inputs and motor output
exactly; whole-network v/g and STP x/u/last-update differences met the registered
tolerances. The receipt reports how many selected presynaptic spikes occurred in
the pending-delay horizon, rather than assuming every checkpoint tests that path.
The separate native micro-test explicitly verifies restoration with a queued delivery.
Brian2 stores event-driven x/u lazily together with their last-update times;
recovery between events is evaluated analytically on the next event. Stored x/u
values alone should not be read as current-time physiological measurements.

Trial wall time: {results['runtime_trial_seconds']:.1f}s; maximum worker RSS:
{results['peak_worker_rss_bytes']/1024**3:.2f}GiB; package storage before this report:
{results['storage_bytes']/1024**3:.2f}GiB. Initialization/checkpoint costs are included;
fresh-process continuation-worker times are not included in trial wall time.

{'Next: preregister bilateral direction checks before any physical navigation test.' if qualified else 'Next: retain this failed afferent-only result and preregister targeted downstream persistence diagnostics. Do not expand STP to arbitrary connections or proceed to navigation/learning.'}
No inference about consciousness, behavioral learning or autonomous navigation follows.

[Published source](https://www.frontiersin.org/journals/computational-neuroscience/articles/10.3389/fncom.2021.730431/full)
and prior isolated event/reference packages establish the mechanism basis, not
biological parameter validity for this application.
''')
progress_path=ROOT/'reports/brain-integration/recovery/progress.json';p=json.loads(progress_path.read_text())
p['phases']['4']['status']='stp_only_full_network_recovery_screen_'+('passed' if qualified else 'failed')
for file in ('RESULTS.md','results.json','protocol.json'):
    evidence=str((OUT/file).relative_to(ROOT))
    if evidence not in p['phases']['4']['evidence']:p['phases']['4']['evidence'].append(evidence)
p['next']='Preregister direction screen.' if qualified else 'Preregister targeted downstream persistence diagnostics; STP-only afferents failed recovery; no navigation promotion.'
p['next_action']=p['next'];p['updated']=time.time();atomic_json(progress_path,p)
ledger_path=ROOT/'reports/brain-integration/acceptance-ledger.json';ledger=json.loads(ledger_path.read_text())
for r in ledger['requirements']:
    if r['step']==8:
        r['status']=p['phases']['4']['status'];r['completion_proven']=False
        evidence=str((OUT/'RESULTS.md').relative_to(ROOT/'reports/brain-integration'))
        if evidence not in r['evidence']:r['evidence'].append(evidence)
ledger['updated']=time.time();atomic_json(ledger_path,ledger)
print(json.dumps({'recovery_prerequisite_passed':qualified,'checks':pair_checks,'checkpoints':checkpoint_checks},indent=2))
