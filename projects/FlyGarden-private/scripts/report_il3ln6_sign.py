"""Independent recording audit and frozen response/recovery screen."""
import sys,json,hashlib,time
from pathlib import Path
import numpy as np,pandas as pd
import brian2 as b
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from flygarden.continuous_candidate import file_sha
from flygarden.descending import DescendingDecoder
from flygarden.recording import atomic_json
OUT=ROOT/'reports/brain-integration/recovery/il3ln6-sign-full-network-v2'
p=json.loads((OUT/'protocol.json').read_text());assert all(file_sha(ROOT/n)==h for n,h in p['sources'].items())
graph=pd.read_parquet(ROOT/'vendor/fly-brain/data/2025_Connectivity_783.parquet');ids=pd.read_csv(ROOT/'vendor/fly-brain/data/2025_Completeness_783.csv',index_col=0).index.to_numpy(dtype=np.int64)
rows=np.array(p['sign_mapping']['rows'],dtype=int);pre=graph.Presynaptic_Index.to_numpy(dtype=int);post=graph.Postsynaptic_Index.to_numpy(dtype=int)
assert np.array_equal(ids[pre],graph.Presynaptic_ID.to_numpy()) and np.array_equal(ids[post],graph.Postsynaptic_ID.to_numpy())
assert np.array_equal(rows,np.flatnonzero(np.isin(ids[pre],[int(x) for x in p['sign_mapping']['root_ids']])))
weights=np.asarray(graph['Excitatory x Connectivity'].to_numpy()*.275*b.mV/b.mV);source=weights.copy();hashes={'original':hashlib.sha256(weights.tobytes()).hexdigest()};weights[rows]*=-1;hashes['sign_only']=hashlib.sha256(weights.tobytes()).hexdigest();assert np.array_equal(np.abs(source),np.abs(weights));del source,weights
edge_map=[{'row':int(i),'pre_root_id':str(ids[pre[i]]),'post_root_id':str(ids[post[i]]),'anatomical_synapses':int(graph.iloc[i].Connectivity),'source_signed_synapses':float(graph.iloc[i]['Excitatory x Connectivity'])} for i in rows]
atomic_json(OUT/'model-variants.json',{'scope':'Two-root source-supported hypothesis, not physiological validation or controller promotion.','neurons':len(ids),'aggregate_records':len(graph),'anatomical_synapses':int(graph.Connectivity.sum()),'all_absolute_weights_match':True,'weights_sha256':hashes,'changed_edges':edge_map,'neural_parameters_and_decoder_unchanged':True});del graph
mapping={k:np.array([n['index'] for n in v],dtype=int) for k,v in p['mapping'].items()};selected=np.array(p['selected']);col={int(n):i for i,n in enumerate(selected)}
matrices={}
changed_rows=set(rows.tolist())
for model in ('original','sign_only'):
 pos=np.zeros((len(ids),len(selected)));neg=np.zeros_like(pos)
 for e in p['incoming_edges']:
  w=e['signed_synapses']*.275*(-1 if model=='sign_only' and e['row'] in changed_rows else 1);pos[e['pre'],col[e['post']]]+=max(w,0);neg[e['pre'],col[e['post']]]+=max(-w,0)
 matrices[model]=(pos,neg)
all_manifests=[];summaries=[];traces_all={};audits=[];checks=[];differences=[]
for case in p['cases']:
 model,cue=case['model'],case['cue']
 for seed in p['seeds']:
  folder=OUT/'trials'/f'{model}-{cue}-{seed}';m=json.loads((folder/'manifest.json').read_text());assert m['status']=='complete' and m['sources']==p['sources'];assert m['initial_weights_sha256']==m['final_weights_sha256']==hashes[model];all_manifests.append(m)
  assert m['controller']['learning'] is False and m['controller']['signed_connectome_weights_modified']==(model=='sign_only')
  data_rows=json.loads((folder/'rows.json').read_text());assert len(m['chunks'])==len(data_rows)==240
  rng=np.random.default_rng(seed);channels=np.array(m['controller']['input_channels']);targets=np.array(m['controller']['input_indices']);decoder=DescendingDecoder();traces={k:[] for k in mapping};individual=[];previous_i=np.array([],dtype=int);previous_ticks=np.array([],dtype=int);max_error=0;count_difference=0;identical=0
  for tick,chunk in enumerate(m['chunks']):
   path=folder/chunk['file'];assert file_sha(path)==chunk['sha256']
   with np.load(path) as z:
    assert np.array_equal(np.bincount(z['spike_i'],minlength=len(ids)),z['counts']);assert np.array_equal(z['spike_counts'],z['counts'][selected]);assert np.array_equal(z['neuron_indices'],selected) and np.array_equal(z['root_ids'],ids[selected])
    assert np.isfinite(z['voltage_mV']).all() and np.isfinite(z['net_synaptic_conductance_equivalent_mV']).all();assert z['gate'].shape==(len(selected),250)
    assert np.allclose(z['gate_t'],(tick*250+np.arange(250))*.0001,rtol=0,atol=1e-12);assert np.allclose(z['time_seconds'],(tick*250+np.arange(0,250,10))*.0001,rtol=0,atol=1e-12)
    rates=np.array([0,0,0,0,65,65,0,0.])
    if cue!='none' and (12<=tick<32 or 132<=tick<152):
     if cue in ('a_left','a_both'):rates[0]=50
     if cue in ('a_right','a_both'):rates[1]=50
    tt,ii=np.nonzero(rng.random((250,len(targets)))<rates[channels]*.0001);expected_ticks=tick*250+tt
    assert np.array_equal(ii,z['external_i']) and np.array_equal(ii,z['delivered_external_indices'])
    assert np.array_equal(expected_ticks,np.rint(z['external_t']/.0001).astype(np.int64));assert np.array_equal(expected_ticks,np.rint(z['delivered_external_times_seconds']/.0001).astype(np.int64))
    assert np.allclose(expected_ticks*.0001,z['external_t'],rtol=0,atol=1e-12) and np.allclose(expected_ticks*.0001,z['delivered_external_times_seconds'],rtol=0,atol=1e-12)
    pop={k:float(z['counts'][indices].mean()/.025) for k,indices in mapping.items()};assert all(abs(pop[k]-data_rows[tick]['population_hz'][k])<1e-10 for k in mapping);assert np.array_equal(decoder.advance(.025,pop),z['motor']);assert np.isfinite(z['motor']).all() and np.all((z['motor']>=0)&(z['motor']<=1.2));assert abs(chunk['end']-(tick+1)*.025)<1e-9
    for k in mapping:traces[k].append(pop[k])
    individual.append(z['counts'][selected]/.025)
    current_ticks=np.rint(z['spike_t']/.0001).astype(np.int64);assert np.all((current_ticks>=tick*250)&(current_ticks<(tick+1)*250));source_i=np.concatenate([previous_i,z['spike_i']]);arrivals=np.concatenate([previous_ticks,current_ticks])+18;within=(arrivals>=tick*250)&(arrivals<(tick+1)*250);source_i=source_i[within];arrivals=arrivals[within]-tick*250
    reconstructed=np.array([(matrix[source_i]*z['gate'][:,arrivals].T).sum(axis=0) for matrix in matrices[model]]);error=float(np.max(np.abs(reconstructed-z['delivered_mV'])));assert error<1e-6,(model,cue,seed,tick,error);max_error=max(max_error,error)
    carry=current_ticks>=(tick+1)*250-18;previous_i=z['spike_i'][carry].copy();previous_ticks=current_ticks[carry].copy()
    if model=='sign_only':
     with np.load(OUT/'trials'/f'original-{cue}-{seed}'/chunk['file']) as orig:
      assert np.array_equal(z['external_i'],orig['external_i']) and np.array_equal(z['external_t'],orig['external_t']);count_difference+=int(np.abs(z['counts']-orig['counts']).sum());identical+=int(np.array_equal(z['spike_i'],orig['spike_i']) and np.array_equal(z['spike_t'],orig['spike_t']))
  traces={k:np.array(v) for k,v in traces.items()};traces['pn']=np.mean([traces['DM1_lPN_left'],traces['DM1_lPN_right']],axis=0);traces['signed_dna']=traces['DNa02_left']-traces['DNa02_right'];traces['pn_contrast']=traces['DM1_lPN_left']-traces['DM1_lPN_right'];traces['individual']=np.array(individual);traces_all[(model,cue,seed)]=traces
  summary={'model':model,'cue':cue,'seed':seed}
  for name,(lo,hi) in [('pulse1',(12,32)),('recovery1',(100,120)),('pulse2',(132,152)),('recovery2',(220,240))]:summary[name]={k:float(traces[k][lo:hi].mean()) for k in ('pn','signed_dna','pn_contrast','residual_local','ALLN','ALPN','il3LN6')}
  summaries.append(summary);audits.append({'model':model,'cue':cue,'seed':seed,'chunks':240,'delivery_max_error_mV':max_error})
  if model=='sign_only':differences.append({'cue':cue,'seed':seed,'whole_count_l1_difference':count_difference,'identical_spike_windows':identical,'total_windows':240})
  if cue=='a_left' and seed==p['seeds'][0]:
   result=json.loads((folder/'continuation-result.json').read_text());assert result['status']=='passed';checks.append({'model':model,**result})
   with np.load(folder/'window-0150.npz') as z:checks[-1]['selected_source_spikes_pending_at_checkpoint']=int(np.sum(np.isin(z['spike_i'],[pre[i] for i in rows[:1]]+[pre[rows[-1]]])&(z['spike_t']>=3.775-.0018-1e-12)))
# Observer parity, including all selected state/gates.
folder=OUT/'trials'/f'sign_only-a_left-{p["seeds"][0]}-observer-off';m=json.loads((folder/'manifest.json').read_text());assert m['status']=='complete' and m['sources']==p['sources'];assert m['initial_weights_sha256']==m['final_weights_sha256']==hashes['sign_only'];all_manifests.append(m);assert len(m['chunks'])==240
for chunk in m['chunks']:
 assert file_sha(folder/chunk['file'])==chunk['sha256']
 with np.load(folder/chunk['file']) as z,np.load(OUT/'trials'/f'sign_only-a_left-{p["seeds"][0]}'/chunk['file']) as orig:
  assert all(np.array_equal(z[k],orig[k]) for k in ('counts','spike_i','spike_t','external_i','external_t','motor','voltage_mV','net_synaptic_conductance_equivalent_mV','gate')) and not z['delivered_mV'].any()
recovery=[];contrast=[]
local_cols=[col[x['index']] for x in p['selected_local_targets']]
for model in ('original','sign_only'):
 for seed in p['seeds']:
  none=traces_all[(model,'none',seed)]
  for cue in ('a_left','a_right','a_both'):
   t=traces_all[(model,cue,seed)]
   for pulse,(pl,ph),(rl,rh) in zip((1,2),p['pulse_windows'],p['recovery_windows']):
    pn_diff={k:float(t[k][rl:rh].mean()-none[k][rl:rh].mean()) for k in ('DM1_lPN_left','DM1_lPN_right')};local_diff=(t['individual'][rl:rh,local_cols].mean(axis=0)-none['individual'][rl:rh,local_cols].mean(axis=0));dna_diff=float(t['signed_dna'][rl:rh].mean()-none['signed_dna'][rl:rh].mean());base=none['pn'][pl:ph].mean() if pulse==1 else t['pn'][100:120].mean();gain=float(t['pn'][pl:ph].mean()-base)
    recovered=all(abs(x)<=5 for x in pn_diff.values()) and np.all(np.abs(local_diff)<=5) and abs(dna_diff)<=5;responded=gain>=10
    recovery.append({'model':model,'cue':cue,'seed':seed,'pulse':pulse,'pn_recovery_difference_hz':pn_diff,'local_individual_recovery_difference_hz':{str(x['root_id']):float(v) for x,v in zip(p['selected_local_targets'],local_diff)},'signed_dna_recovery_difference_hz':dna_diff,'response_gain_hz':gain,'recovered':bool(recovered),'response_passed':bool(responded),'response_and_recovery_passed':bool(recovered and responded)})
  for pulse,(lo,hi) in zip((1,2),p['pulse_windows']):
   left=float(traces_all[(model,'a_left',seed)]['pn_contrast'][lo:hi].mean());right=float(traces_all[(model,'a_right',seed)]['pn_contrast'][lo:hi].mean());neutral=float(none['pn_contrast'][lo:hi].mean());passed=(left-right>=10 and left-neutral>0 and right-neutral<0)
   contrast.append({'model':model,'seed':seed,'pulse':pulse,'left_pn_contrast_hz':left,'right_pn_contrast_hz':right,'none_contrast_hz':neutral,'passed':bool(passed)})
qualified=all(x['response_and_recovery_passed'] for x in recovery if x['model']=='sign_only');contrast_passed=all(x['passed'] for x in contrast if x['model']=='sign_only')
r={'status':'complete','trials':17,'raw_chunks_audited':4080,'raw_spikes_match_counts':True,'owned_input_events_reconstructed':True,'external_integer_clock_ticks_exact':True,'paired_external_events_exact':True,'decoder_reconstruction_exact':True,'delayed_gated_delivery_reconstruction_passed':True,'all_nonselected_weights_match_source':True,'all_absolute_weights_match_source':True,'weights_unchanged_during_trials':True,'instrumentation_parity_exact':True,'checkpoint_checks':checks,'summary':summaries,'recovery_checks':recovery,'contrast_checks':contrast,'paired_dynamics':differences,'audits':audits,'candidate_response_recovery_passed':qualified,'candidate_sensory_contrast_passed':contrast_passed,'promoted':False,'navigation_validated':False,'learning_validated':False,'biological_correction_validated':False,'wall_seconds_trials':sum(x['wall_seconds'] for x in all_manifests),'peak_rss_bytes':max(x['peak_rss_bytes'] for x in all_manifests),'report_sha256':file_sha(Path(__file__))}
atomic_json(OUT/'results.json',r)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
fig,axes=plt.subplots(3,3,figsize=(15,10),sharex=True);times=np.arange(1,241)*.025
for j,cue in enumerate(('a_left','a_right','a_both')):
 for model,color in [('original','#b45240'),('sign_only','#276ea6')]:
  for i,key in enumerate(('pn','signed_dna','residual_local')):
   v=np.array([traces_all[(model,cue,s)][key] for s in p['seeds']]);axes[i,j].plot(times,v.mean(axis=0),color=color,label=model);axes[i,j].fill_between(times,v.min(axis=0),v.max(axis=0),color=color,alpha=.13)
 for i in range(3):
  axes[i,j].axvspan(.3,.8,color='gray',alpha=.15);axes[i,j].axvspan(3.3,3.8,color='gray',alpha=.15);axes[i,j].grid(alpha=.2)
 axes[0,j].set_title(cue.replace('_',' '));axes[0,j].legend();axes[2,j].set_xlabel('Simulated seconds')
axes[0,0].set_ylabel('DM1 projection mean Hz');axes[1,0].set_ylabel('Left minus right DNa02 Hz');axes[2,0].set_ylabel('Selected local mean Hz');fig.suptitle('Two-root sign hypothesis; two seed ranges, not confidence intervals');fig.tight_layout();fig.savefig(OUT/'comparison.png',dpi=150);plt.close(fig)
table='\n'.join(f"| {x['model']} | {x['cue']} | {x['seed']} | {x['pulse']} | {x['response_gain_hz']:.1f} | {max(abs(v) for v in x['pn_recovery_difference_hz'].values()):.1f} | {max(abs(v) for v in x['local_individual_recovery_difference_hz'].values()):.1f} | {x['signed_dna_recovery_difference_hz']:.1f} | {'pass' if x['response_and_recovery_passed'] else 'fail'} |" for x in recovery)
(OUT/'RESULTS.md').write_text(f'''# Two-il3LN6 sign candidate: response and recovery

Seventeen six-second full-network trials completed. Candidate response/recovery prerequisite: **{'passed' if qualified else 'failed'}**. Candidate sensory contrast prerequisite: **{'passed' if contrast_passed else 'failed'}**. No controller promotion or application change.

Exactly3,546 outgoing weights of roots720575940623636701 and720575940632403986 are multiplied by−1 before simulation. All138,639 neurons,15,091,983 aggregated connection records and absolute weight magnitudes are retained. No peptide dynamics, compartments, adaptation, other signs, neuron parameters, sensory encoding or decoder are changed. Learning is disabled. The sign change is a source-supported cell-type hypothesis, not validated biology.

Trials use support-only, left, right and bilateral synthetic odorA. The same condition is presented at .3–.8s and3.3–3.8s. Recovery windows are2.5–3.0s and5.5–6.0s. A response must increase mean DM1 PN firing by at least10Hz; for the second pulse the reference is the first recovery mean. Recovery requires both individual PNs and all four selected local neurons within5Hz of same-model/same-seed support-only, plus signed DNa02 within5Hz. These frozen gates apply to every non-none cue, both pulses and both diagnostic seeds.

| Model | Cue | Seed | Pulse | Response gain Hz | Max PN recovery difference Hz | Max individual local difference Hz | Signed steering difference Hz | Gate |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
{table}

Sensory contrast separately requires the left PN contrast to exceed the right contrast by10Hz and opposite signs relative to support-only in each seed/pulse. It does not validate physical movement direction or navigation.

## Verification and resources

All4,080 chunk hashes and raw spike/count identities passed. Owned input RNG events and exact integer ticks were reconstructed; decimal timestamps agreed within1e-12s. Paired input events and fixed decoding matched exactly. Positive/negative accepted voltage increments reconstructed from1.8ms-delayed spikes and100µs gates passed the1e-6mV tolerance across chunk boundaries; these increments are not biological currents. Whole weight hashes establish that only the selected signs changed and remained fixed.

The observer-disabled candidate replay matched all spikes, inputs, commands and selected state/gates exactly. Original and candidate checkpoints continued for five windows in fresh processes with exact events/counts/inputs/motor and whole-network v/g differences at most1e-10mV. Checkpoint receipts report selected-source spikes in the pending delay horizon; they do not invent pending events when none occurred.

Trial wall time: {r['wall_seconds_trials']:.1f}s. Peak worker memory: {r['peak_rss_bytes']/1024**3:.2f}GiB. One worker. Two diagnostic seeds provide a bounded screen, not population statistics. No body, reward or learning evaluation ran.

The original v1 one-chunk interruption is preserved. AMENDMENT.json documents the recording-only timestamp-check correction; no scientific criterion, seed or model parameter changed. The anatomical sign candidate retains versionv1; recording protocol isv2.
''')
print(json.dumps({'response_recovery':qualified,'sensory_contrast':contrast_passed,'checkpoint':checks,'summary':summaries},indent=2))
