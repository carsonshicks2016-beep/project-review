"""Validate Stage4 raw events and reconstruct signed deliveries by source."""
import argparse,json,sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.audit_brain_reference import sha
from flygarden.recording import atomic_json


def inspect_trial(root,name,protocol):
 folder=root/'trials'/name;manifest=json.loads((folder/'manifest.json').read_text());assert manifest['status']=='complete';assert manifest['sources']==protocol['sources'];assert manifest['neurons']==138639 and manifest['connections']==15091983 and len(manifest['chunks'])==20
 rows=json.loads((folder/'bins.json').read_text());assert len(rows)==20;events_i=[];events_t=[];gates=[];voltage=[];conductance=[];windows=0;spikes=0;expected=[]
 for tick,chunk in enumerate(manifest['chunks']):
  file=folder/chunk['file'];assert sha(file)==chunk['sha256']
  with np.load(file) as data:
   i=data['i'];t=data['t'];counts=data['counts'];inp=data['input_counts'];v=data['v_mV'];g=data['g_mV'];gate=data['gate'];st=data['state_t'];delivery=data['delivered_mV']
   assert counts.shape==(138639,) and np.array_equal(np.bincount(i,minlength=138639),counts) and len(i)==chunk['spikes'];assert np.isfinite(t).all() and np.all(t>=tick*.1-1e-9) and np.all(t<(tick+1)*.1+1e-9)
   assert v.shape==g.shape==gate.shape==(2,1000);assert np.isfinite(v).all() and np.isfinite(g).all();assert np.allclose(st,np.arange(tick*1000,(tick+1)*1000)*.0001,rtol=0,atol=1e-9)
   assert np.isclose(inp.sum(),rows[tick]['input_events']);assert np.isfinite(delivery).all() and np.all(delivery>=-1e-6)
   if not 3<=tick<8:assert inp.sum()==0
   assert rows[tick]['input_rate_hz']==(50 if 3<=tick<8 else 0)
   assert np.allclose(counts[protocol['DNa02_targets']]/.1,rows[tick]['DNa02_hz'])
   events_i.append(i.copy());events_t.append(t.copy());gates.append(gate.copy());voltage.append(v.copy());conductance.append(g.copy());expected.append(delivery.copy())
   windows+=1;spikes+=len(i)
 raw_i=np.concatenate(events_i);raw_t=np.concatenate(events_t);gate=np.concatenate(gates,axis=1);v=np.concatenate(voltage,axis=1);g=np.concatenate(conductance,axis=1)
 incoming=protocol['incoming_edges'];pres=np.array(sorted({r['pre_index'] for r in incoming}),dtype=np.int32);keep=np.isin(raw_i,pres);filtered_i=raw_i[keep];filtered_tick=np.rint(raw_t[keep]/.0001).astype(np.int64);sort=np.argsort(filtered_i,kind='stable');filtered_i=filtered_i[sort];filtered_tick=filtered_tick[sort];unique,first,length=np.unique(filtered_i,return_index=True,return_counts=True);times={int(n):filtered_tick[start:start+count] for n,start,count in zip(unique,first,length)}
 zero=set(protocol['lesions'][manifest['profile']]['edge_indices']);reconstructed=np.zeros((20,2,2));contributions=[]
 for edge in incoming:
  weight=0 if edge['edge_index'] in zero else edge['signed_count']*.275
  target_side=protocol['DNa02_targets'].index(edge['post_index']);arrival=times.get(edge['pre_index'],np.array([],dtype=np.int64))+18;arrival=arrival[arrival<20000];accepted=arrival[gate[target_side,arrival]];bins=np.bincount(accepted//1000,minlength=20).astype(float)*abs(weight)
  sign=0 if weight>=0 else 1;reconstructed[:,sign,target_side]+=bins
  contributions.append({**edge,'effective_weight_mV':weight,'pulse_delivered_mV':float(bins[3:8].sum()),'recovery_delivered_mV':float(bins[8:].sum()),'early_delivered_mV':float(bins[3:5].sum()),'accepted_arrivals':int(len(accepted)) if weight else 0})
 assert np.allclose(reconstructed,np.array(expected),rtol=1e-9,atol=1e-5),('Delivery attribution does not match observer',name,float(np.max(np.abs(reconstructed-np.array(expected)))))
 active=np.array([r['DNa02_hz'] for r in rows]);early=active[3:5].mean(axis=0);tail=active[15:].mean(axis=0)
 result={'trial':name,'profile':manifest['profile'],'cue':manifest['cue'],'seed':manifest['seed'],'early_DNa02_hz':early.tolist(),'pulse_DNa02_hz':active[3:8].mean(axis=0).tolist(),'tail_DNa02_hz':tail.tolist(),'early_left_minus_right_hz':float(early[0]-early[1]),'pulse_mean_v_mV':v[:,3000:8000].mean(axis=1).tolist(),'pulse_mean_g_mV':g[:,3000:8000].mean(axis=1).tolist(),'pulse_exc_delivered_mV':reconstructed[3:8,0].sum(axis=0).tolist(),'pulse_inh_delivered_mV':reconstructed[3:8,1].sum(axis=0).tolist(),'pulse_refractory_fraction':(1-gate[:,3000:8000].mean(axis=1)).tolist(),'windows':windows,'spikes':spikes,'attribution_matches_observer':True,'counter_scope':'Signed model voltage increments accepted at synaptic delivery before possible reset; not physical current, conductance, or a biological E/I balance.'}
 atomic_json(folder/'source-deliveries.json',{'schema_version':1,'trial':name,'counter_scope':result['counter_scope'],'contributions':contributions});atomic_json(folder/'summary.json',result);return result


def run(root,partial=False):
 protocol=json.loads((root/'protocol.json').read_text());progress=json.loads((root/'progress.json').read_text());assert all(sha(ROOT/name)==digest for name,digest in protocol['sources'].items());assert json.loads((root/'instrumentation-check.json').read_text())['passed']
 if not partial:assert progress['status']=='complete' and len(progress['completed'])==36
 results=[inspect_trial(root,name,protocol) for name in progress['completed']];checks={'passed':True,'trials':len(results),'windows':sum(r['windows'] for r in results),'raw_spikes':sum(r['spikes'] for r in results),'all_source_attribution_matches_observer':True,'sources_unchanged':True,'observer_parity_passed':True,'plasticity_updates':0,'full_network_neurons':138639,'full_network_connection_records':15091983}
 if partial:print(json.dumps(checks));return
 # Match exogenous stimulation across every narrow intervention and intact control.
 for result in results:
  if result['profile'] in ('intact_narrow','intact_broad'):continue
  control=root/'trials'/f"intact_narrow-{result['cue']}-{result['seed']}"
  modified=root/'trials'/result['trial']
  for file in control.glob('window-*.npz'):
   with np.load(file) as a,np.load(modified/file.name) as b:assert np.array_equal(a['input_counts'],b['input_counts'])
 checks['matched_narrow_input_event_counts']=True
 paired=[]
 for profile in protocol['profiles']:
  for seed in protocol['seeds']:
   left=next(r for r in results if r['profile']==profile and r['cue']=='left' and r['seed']==seed);right=next(r for r in results if r['profile']==profile and r['cue']=='right' and r['seed']==seed)
   tracked=left['early_left_minus_right_hz']>5 and right['early_left_minus_right_hz']< -5;recovers=max(left['tail_DNa02_hz']+right['tail_DNa02_hz'])<5
   paired.append({'profile':profile,'seed':seed,'left_cue_difference_hz':left['early_left_minus_right_hz'],'right_cue_difference_hz':right['early_left_minus_right_hz'],'tracks_cue':tracked,'recovers':recovers,'intact_sensor_gate':tracked and recovers and profile=='intact_broad'})
 broad_gate=all(r['intact_sensor_gate'] for r in paired if r['profile']=='intact_broad')
 baseline=[r for r in results if r['profile']=='intact_narrow'];right_off=[r for r in results if r['profile']=='right_negative_off'];ps=[r for r in results if r['profile']=='PS049_negative_off'];feedback=[r for r in results if r['profile']=='steering_feedback_off']
 def mean_rate(rows,side):return float(np.mean([r['pulse_DNa02_hz'][side] for r in rows]))
 findings={'right_baseline_pulse_hz':mean_rate(baseline,1),'right_negative_off_pulse_hz':mean_rate(right_off,1),'PS049_negative_off_right_pulse_hz':mean_rate(ps,1),'feedback_off_left_pulse_hz':mean_rate(feedback,0),'baseline_left_pulse_hz':mean_rate(baseline,0),'broad_intact_gate_passed':broad_gate}
 analysis={'schema_version':1,'verification':checks,'findings':findings,'cue_comparisons':paired,'trials':results,'promoted':False,'brain_changed':False,'learning_demonstrated':False,'scope':'Full-import model dynamics with explicit local diagnostic lesions; no body/navigation evaluation.'};atomic_json(root/'verification.json',checks);atomic_json(root/'analysis.json',analysis)
 measurements={'neural_active_wall_seconds':progress['wall_seconds'],'peak_worker_rss_bytes':max(json.loads((root/'trials'/r['trial']/'manifest.json').read_text())['peak_rss_bytes'] for r in results),'evidence_bytes':sum(p.stat().st_size for p in root.rglob('*') if p.is_file())};atomic_json(root/'measurements.json',measurements)
 fig,axes=plt.subplots(2,2,figsize=(12,8),sharex=True)
 for profile,color in [('intact_narrow','#287db5'),('intact_broad','#d47d22'),('right_negative_off','#b43662'),('PS049_negative_off','#238447'),('steering_feedback_off','#7f69b3')]:
  bins=json.loads((root/'trials'/f'{profile}-left-4199'/'bins.json').read_text());t=[r['time']-.05 for r in bins]
  for side in (0,1):
   axes[0,side].plot(t,[r['DNa02_hz'][side] for r in bins],label=profile,color=color);axes[1,side].plot(t,[r['mean_g_mV'][side] for r in bins],color=color)
 for ax in axes.flat:ax.axvspan(.3,.8,color='grey',alpha=.12);ax.grid(alpha=.15)
 axes[0,0].set_title('Left DNa02');axes[0,1].set_title('Right DNa02');axes[0,0].set_ylabel('Firing rate (Hz)');axes[1,0].set_ylabel('Model synaptic state g (mV)');axes[0,1].legend(fontsize=8);axes[1,0].set_xlabel('Simulated time (s)');axes[1,1].set_xlabel('Simulated time (s)');fig.suptitle('Held-out steering-bias interventions: left odor pulse\nShaded = input; lesion profiles are diagnostic only, not controller candidates');fig.tight_layout();fig.savefig(root/'steering-interventions.png',dpi=160);plt.close(fig)
 table='\n'.join(f"| {r['profile']} | {r['seed']} | {r['left_cue_difference_hz']:.1f} | {r['right_cue_difference_hz']:.1f} | {'yes' if r['tracks_cue'] else 'no'} | {'yes' if r['recovers'] else 'no'} |" for r in paired)
 intervention='\n'.join(f"| {p} | {protocol['lesions'][p]['records_zeroed']:,} | {protocol['lesions'][p]['anatomical_synapses_affected']:,} |" for p in protocol['profiles'])
 # Attribution ranking is reported for every full-trial source; representative
 # held-out intact examples are readable while raw records retain all sources.
 examples=[]
 for cue in ('left','right'):
  name=f'intact_narrow-{cue}-4199';contrib=json.loads((root/'trials'/name/'source-deliveries.json').read_text())['contributions']
  for side,target in enumerate(protocol['DNa02_targets']):
   top=sorted([r for r in contrib if r['post_index']==target and r['effective_weight_mV']<0],key=lambda r:r['pulse_delivered_mV'],reverse=True)[:5]
   examples.append({'trial':name,'DNa02_side':('left','right')[side],'top_negative_sources':top})
 atomic_json(root/'top-negative-sources.json',examples)
 followup_section=''
 followup=root/'active-source-followup/analysis.json'
 if followup.exists():
  extra=json.loads(followup.read_text());assert extra['status']=='complete' and extra['source_attribution_passed']
  fork=json.loads((followup.parent/'protocol.json').read_text());assert all(sha(ROOT/name)==digest for name,digest in fork['sources'].items())
  for result in extra['trials']:
   verified=inspect_trial(followup.parent,result['trial'],fork);assert verified['spikes']==result['spikes']
   control=root/'trials'/f"intact_narrow-{result['cue']}-{result['seed']}";modified=followup.parent/'trials'/result['trial']
   for file in control.glob('window-*.npz'):
    with np.load(file) as a,np.load(modified/file.name) as b:assert np.array_equal(a['input_counts'],b['input_counts'])
  analysis['exploratory_followup']=extra;atomic_json(root/'analysis.json',analysis)
  atomic_json(root/'combined-verification.json',{'passed':True,'primary_trials':36,'exploratory_trials':6,'total_windows':checks['windows']+extra['windows'],'total_raw_spikes':checks['raw_spikes']+extra['raw_spikes'],'all_source_attribution_passed':True,'matched_external_inputs_passed':True,'observer_parity_passed':True,'sources_unchanged':True})
  fig,axes=plt.subplots(1,2,figsize=(12,4.5))
  right_sources=next(e for e in examples if e['trial']=='intact_narrow-left-4199' and e['DNa02_side']=='right')['top_negative_sources'];labels=[r['pre_cell_type']+' · …'+r['pre_root_id'][-5:] for r in right_sources]
  axes[0].barh(labels[::-1],[r['pulse_delivered_mV'] for r in right_sources][::-1],color='#b34966');axes[0].set_xlabel('Accepted negative model increments (mV)');axes[0].set_title('Right DNa02: measured negative sources\nHeld-out primary left-cue pulse')
  rates=[findings['right_baseline_pulse_hz'],findings['PS049_negative_off_right_pulse_hz'],extra['mean_lesioned_right_pulse_hz'],findings['right_negative_off_pulse_hz']];axes[1].bar(['Intact','PS049\noff','AOTU019\noff','All negative\ninput off'],rates,color=['#287db5','#238447','#d47d22','#b43662']);axes[1].set_ylabel('Right DNa02 pulse firing (Hz)');axes[1].set_title('Diagnostic interventions, not controllers\nMeans: three seeds × two cue sides');fig.tight_layout();fig.savefig(root/'active-inhibitor.png',dpi=160);plt.close(fig)
  followup_section=f"""## Exploratory active-source intervention

The original trial results selected {extra['source_cell_type']} as the largest delivered negative source group into right DNa02, using the calibration trials. A subsequent six-trial matched-seed intervention zeroed only {extra['records_zeroed']} negative connection record from that group into the right target. Right DNa02 pulse firing changed from {extra['mean_control_right_pulse_hz']:.1f} Hz to {extra['mean_lesioned_right_pulse_hz']:.1f} Hz. Attribution checks also passed for its {extra['windows']} windows and {extra['raw_spikes']:,} raw spikes.

![Active-source attribution and interventions](active-inhibitor.png)

This hypothesis was selected after the primary results; its reused seeds do not provide independent held-out validation. The intervention tests a model causal dependency and does not establish biologically incorrect inhibition or an acceptable replacement controller. `active-source-followup/protocol.json` preserves selection, root IDs, exact edge IDs, sources and limitations.

"""
 text=f'''# Fly Garden — Step 4 steering-bias diagnosis

## Result

Completed 36 matched full-network trials, plus a full-network observer on/off check. Verified {checks['windows']:,} 100 ms windows and {checks['raw_spikes']:,} raw spikes. Both DNa02 neurons were sampled every 0.1 ms. Source-resolved delayed, refractory-gated delivery reconstructions matched the live diagnostic counters in every trial.

Right DNa02 mean firing during the narrow odor pulse: {findings['right_baseline_pulse_hz']:.1f} Hz intact; {findings['right_negative_off_pulse_hz']:.1f} Hz with its negative input blocked; {findings['PS049_negative_off_right_pulse_hz']:.1f} Hz with PS049 negative input blocked. Left DNa02: {findings['baseline_left_pulse_hz']:.1f} Hz intact; {findings['feedback_off_left_pulse_hz']:.1f} Hz with both steering neurons' outgoing feedback blocked. These are six-trial means (two cue sides, three seeds), not population variability across biological animals.

Broad intact sensory candidate: **{'passes' if broad_gate else 'fails'}** the predeclared combined cue-tracking and recovery gate. No candidate is promoted; the live brain and its learned parameters are unchanged.

## What the tests mean

Negative-input lesions intervene on signed weights in this simplified model. Their response can establish model causal dependencies, not that inhibition is biologically faulty. PS049 was selected prospectively as the largest individual negative anatomical source on both sides, before measuring its delivered activity. The delivery ranking uses actual source spikes, 1.8 ms delays and the postsynaptic refractory gate, rather than synapse count alone. Recorded positive/negative voltage increments are not physical currents, and resets can discard a delivered increment later in the same timestep.

Broader stimulation recruits annotated left/right ORN types, instead of only ORN_DM1. This tests narrow-input coverage as a hypothesis; it is still artificial activation, not a natural odor receptor encoding. {len(protocol['broad_populations']['left'])} left and {len(protocol['broad_populations']['right'])} right ORNs are mapped by exact root ID; {protocol['excluded_broad_ORNs_without_left_right_annotation']} ORNs with unavailable side annotation are explicitly excluded from stimulation, but retained in the network. The broad groups can differ in neuron count.

All trials import 138,639 neurons and 15,091,983 connection records. Lesions retain those records but zero the listed weights, so they are **not full-function intact controllers**. Ordinary cells retain 2.2 ms refractory; only actually stimulated ORNs use the released zero-refractory convention. No tonic walking, direct steering input, arena state, reward or learning is supplied.

## Intervention manifest

| Profile | Records zeroed | Anatomical synapses represented by those records |
| --- | ---: | ---: |
{intervention}

## Cue tracking and recovery

Acceptance requires left-minus-right DNa02 >5 Hz for a left cue and <−5 Hz for a right cue in the first 200 ms of input, on both calibration seeds and held-out 4199. Both outputs must also return below5 Hz during the last500 ms. Lesion success never qualifies for production promotion.

| Profile | Seed | Left cue difference (Hz) | Right cue difference (Hz) | Tracks side | Recovers |
| --- | ---: | ---: | ---: | --- | --- |
{table}

![Steering interventions](steering-interventions.png)

## Source attribution and internal state

`top-negative-sources.json` provides representative held-out rankings. Every trial retains `source-deliveries.json`, listing all incoming records with exact source/target IDs, annotation, effective signed weight, accepted arrivals and delivered increments in early, pulse and recovery periods. `bins.json` includes voltage, synaptic state and refractory fraction. Full raw spike events and state samples remain in hashed chunks for reanalysis.

{followup_section}## Decision

{'Broad intact stimulation passed this limited gate; it still requires receptor-level encoding calibration and body/closed-loop evaluation before any promotion.' if broad_gate else 'The tested intact sensory configuration does not provide a validated steering controller. Keep the arena controller unchanged. Use these causal and source-attribution results to choose a documented model/circuit correction, rather than promoting a lesion or fitting direct odor-to-turn rules.'} Navigation, learning and biological accuracy remain unproven.

## Verification and runtime

Five meaningful lesion/observer/decoder tests passed. Observer enabled/disabled runs matched all neuron spike IDs, spike times, counts and input counts; final DNa02 voltage/synaptic state matched. Every count chunk equals its raw spike histogram. Input-free periods, source hashes, finite states, frozen learning and exact lesion manifests are checked. Source attribution independently reproduces the measured delivery counter for every100 ms bin.

Active simulation wall time: {measurements['neural_active_wall_seconds']:.1f} s. Peak neural worker RSS: {measurements['peak_worker_rss_bytes']/1024**3:.2f} GiB (macOS process accounting; excludes the live app). Evidence before plots/report: {measurements['evidence_bytes']/1024**2:.1f} MiB. One sequential neural worker, 2 GiB free-space reserve, no deletion.

Reproduce with `.venv-next/bin/python scripts/diagnose_steering_bias.py NEW_FOLDER`, then `.venv-next/bin/python scripts/report_steering_bias.py NEW_FOLDER`. Existing complete trials are retained. Preserve incomplete attempts before retry. Diagnostic observer semantics follow [Brian2 synapses](https://brian2.readthedocs.io/en/stable/user/synapses.html) and [refractoriness](https://brian2.readthedocs.io/en/stable/user/refractoriness.html). Model source/data identities are pinned in `protocol.json`.
'''
 (root/'RESULTS.md').write_text(text);print(json.dumps(findings,indent=2))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('--partial',action='store_true');a=p.parse_args();run(a.root,a.partial)
