"""Audit the committed Step8 causal experiment, without retuning its controller."""
import sys,json,hashlib,shutil,platform,importlib.metadata as metadata
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from flygarden.recording import atomic_json
from flygarden.descending import DescendingDecoder
from flygarden.causal_metrics import trajectory_metrics,paired_bootstrap
from scripts.diagnose_causal_behavior import OUT,SEEDS,ARMS,DT,DURATION,sha,sources,cue_for

def read(folder,protocol,neural=True):
 m=json.loads((folder/'manifest.json').read_text());assert m['status']=='complete' and m['sources']==protocol['sources'];rows=json.loads((folder/'bins.json').read_text());frames=json.loads((folder/'frames.json').read_text());assert len(rows)==len(m['chunks'])==120 and len(frames)==90
 assert np.all(np.abs(np.array([f['time'] for f in frames])-np.arange(1,91)/30)<=.0005+1e-9)
 assert m['metrics']==trajectory_metrics(m['initial'],frames,rows,m['cue'])
 if neural:
  assert m['neurons']==138639 and m['connections']==15091983 and m['anatomical_synapses']==54492922 and not m['learning'] and m['weights_after_sha256']==m['expected_weights_sha256']
  if m['arm']!='steering_cut':assert m['expected_weights_sha256']==m['baseline_weights_sha256']
 decoder=DescendingDecoder();counts=np.zeros(138639,dtype=np.int64);physics=[];motor=[];external_i=[];external_t=[]
 for tick,(c,row) in enumerate(zip(m['chunks'],rows)):
  file=folder/c['file'];assert sha(file)==c['sha256'];start=tick*DT;end=(tick+1)*DT
  assert abs(row['motor_start']-start)<1e-9 and abs(row['time']-end)<1e-9
  with np.load(file) as archive:
   a={key:archive[key] for key in archive.files}
   assert np.array_equal(a['motor'],row['applied_motor']) and np.isfinite(a['physics']).all();physics.append(a['physics'].copy());motor.append(a['motor'].copy())
   if neural:
    assert len(a['spike_i'])==c['spikes']==int(a['counts'].sum());assert np.array_equal(np.bincount(a['spike_i'],minlength=138639),a['counts']);assert np.array_equal(np.bincount(a['external_i'],minlength=len(m['input_indices'])),a['input_counts'])
    for key in ('spike_t','external_t'):assert np.all(a[key]>=start-1e-9) and np.all(a[key]<end-1e-9)
    assert row['observation_time']==row['motor_start'] and row['image_time_min']<=start+1e-9 and row['image_time_max']<end+1e-9 and row['next_available_at']==row['time']
    expected_rates={k:float(a['counts'][[n['index'] for n in v]].mean()/DT) for k,v in m['mapping'].items()};assert expected_rates==row['population_hz'];assert np.array_equal(decoder.advance(DT,expected_rates),a['next_motor']);assert np.array_equal(a['next_motor'],row['next_motor'])
    assert row['applied_motor']==([0.,0.] if tick==0 else rows[tick-1]['next_motor'])
    counts+=a['counts'];external_i.append(a['external_i'].copy());external_t.append(a['external_t'].copy())
 return dict(manifest=m,rows=rows,frames=frames,counts=counts,physics=physics,motor=np.array(motor),external_i=np.concatenate(external_i) if neural else None,external_t=np.concatenate(external_t) if neural else None)

def lesion_reference(protocol):
 import pandas as pd
 comp=pd.read_csv(ROOT/'vendor/fly-brain/data/2025_Completeness_783.csv',index_col=0);ids=comp.index.to_numpy(dtype=np.int64)
 c=pd.read_parquet(ROOT/'vendor/fly-brain/data/2025_Connectivity_783.parquet',columns=['Presynaptic_Index','Postsynaptic_Index','Excitatory x Connectivity','Connectivity']);pre=c.Presynaptic_Index.to_numpy(dtype=np.int32);post=c.Postsynaptic_Index.to_numpy(dtype=np.int32);m=json.loads((OUT/'trials/8101/intact/manifest.json').read_text());target=np.array([n['index'] for k in ('DNa02_left','DNa02_right') for n in m['mapping'][k]]);indices=np.flatnonzero(np.isin(post,target));base=(c['Excitatory x Connectivity'].to_numpy()*.275*.001)/.001;reference={'indices':indices,'pre':pre[indices],'post':post[indices],'original_w_mV':base[indices],'pre_root_ids':ids[pre[indices]],'post_root_ids':ids[post[indices]]};original_hash=hashlib.sha256(base.tobytes()).hexdigest();base[indices]=0.;cut_hash=hashlib.sha256(base.tobytes()).hexdigest()
 return reference,original_hash,cut_hash,int(c.Connectivity.to_numpy()[indices].sum())

def report(partial=False):
 protocol=json.loads((OUT/'protocol.json').read_text());assert protocol['sources']==sources();progress=json.loads((OUT/'progress.json').read_text());seeds=progress['completed'] if partial else list(SEEDS)
 if not partial:assert progress['status']=='complete' and progress['completed']==list(SEEDS)
 if not seeds:return
 reference,baseline_hash,cut_hash,interrupted_synapses=lesion_reference(protocol);records=[];by_seed={}
 eye=ROOT/'reports/brain-integration/stage6-20261006/eye-stimuli';eyes=json.loads((eye/'manifest.json').read_text())
 for name,digest in eyes['files'].items():assert sha(eye/name)==digest
 for seed in seeds:
  cases={};folder=OUT/'trials'/str(seed)
  with np.load(folder/'interrupted-edges.npz') as z:
   for key,value in reference.items():assert np.array_equal(z[key],value)
  for arm in (*ARMS,'motor_replay','motor_zero'):
   case=read(folder/arm,protocol,neural=arm in ARMS);cases[arm]=case;m=case['manifest'];assert m['cue']==cue_for(seed) and m['seed']==seed
   if arm in ARMS:
    assert m['baseline_weights_sha256']==baseline_hash and m['expected_weights_sha256']==(cut_hash if arm=='steering_cut' else baseline_hash);assert sha(folder/'interrupted-edges.npz')==m['intervention']['edge_sha256'];assert m['eye_source_sha256']==sha(eye/f'{cue_for(seed)}.json')
   else:assert not m['brain_recomputed'] and m['source_manifest_sha256']==sha(folder/'intact/manifest.json')
   assert m['initial']==cases['intact']['manifest']['initial']
   pops=cases['intact']['manifest']['mapping'];pop_counts={k:int(case['counts'][[n['index'] for n in v]].sum()) for k,v in pops.items()} if arm in ARMS else None
   records.append({'seed':seed,'cue':m['cue'],'arm':arm,'metrics':m['metrics'],'spikes':int(case['counts'].sum()) if arm in ARMS else None,'external_events':len(case['external_t']) if arm in ARMS else None,'population_spikes':pop_counts,'wall_seconds':m['wall_seconds'],'peak_rss_bytes':m.get('peak_rss_bytes')})
  full=cases['intact'];off=cases['input_off'];cut=cases['steering_cut'];replay=cases['motor_replay'];zero=cases['motor_zero']
  assert np.array_equal(full['external_i'],cut['external_i']) and np.array_equal(full['external_t'],cut['external_t'])
  assert len(off['external_t'])==0 and not off['motor'].any() and not off['counts'].any()
  steering=[n['index'] for k in ('DNa02_left','DNa02_right') for n in full['manifest']['mapping'][k]];assert not cut['counts'][steering].any()
  assert all(np.array_equal(a,b) for a,b in zip(full['physics'],replay['physics'])) and full['frames']==replay['frames'] and all(a['body']==b['body'] for a,b in zip(full['rows'],replay['rows']))
  assert all(np.array_equal(a,b) for a,b in zip(off['physics'],zero['physics'])) and off['frames']==zero['frames'] and all(a['body']==b['body'] for a,b in zip(off['rows'],zero['rows']))
  final_delta=np.array(full['frames'][-1]['body']['position'])[:2]-np.array(off['frames'][-1]['body']['position'])[:2]
  by_seed[seed]={'seed':seed,'cue':cue_for(seed),'motor_rms_vs_input_off':float(np.sqrt(np.mean((full['motor']-off['motor'])**2))),'motor_rms_vs_steering_cut':float(np.sqrt(np.mean((full['motor']-cut['motor'])**2))),'away_heading_vs_input_off':full['manifest']['metrics']['away_heading_radians']-off['manifest']['metrics']['away_heading_radians'],'away_heading_vs_steering_cut':full['manifest']['metrics']['away_heading_radians']-cut['manifest']['metrics']['away_heading_radians'],'final_xy_departure_mm':float(np.linalg.norm(final_delta)),'no_flips':full['manifest']['metrics']['flipped_frames']==0,'stalled':full['manifest']['metrics']['stalled'],'replay_exact':True,'zero_matches_input_off_exact':True}
 effects=list(by_seed.values());worker_seconds=sum(json.loads((OUT/'trials'/str(seed)/'worker.json').read_text())['wall_seconds'] for seed in seeds);result={'status':'partial' if partial else 'complete','seed_count':len(seeds),'full_brain_trials':3*len(seeds),'body_only_trials':2*len(seeds),'interrupted_records':len(reference['indices']),'interrupted_anatomical_synapses':interrupted_synapses,'source_and_chunk_integrity_passed':True,'identical_intact_cut_external_events':True,'quiet_input_off_verified':True,'steering_neurons_silent_when_interrupted':True,'motor_replay_exact':True,'zero_motor_matches_input_off_exact':True,'records':records,'paired_effects':effects,'worker_wall_seconds':worker_seconds,'completed_neural_arm_wall_seconds':sum(r['wall_seconds'] for r in records if r['arm'] in ARMS),'body_control_wall_seconds':sum(r['wall_seconds'] for r in records if r['arm'] not in ARMS),'peak_worker_rss_bytes':max(r['peak_rss_bytes'] or 0 for r in records),'storage_bytes':sum(f.stat().st_size for f in OUT.rglob('*') if f.is_file()),'free_bytes':shutil.disk_usage(OUT).free}
 if not partial:
  sides=[r['cue'] for r in effects];stats={key:paired_bootstrap([r[key] for r in effects],sides) for key in ('motor_rms_vs_input_off','motor_rms_vs_steering_cut','away_heading_vs_input_off','away_heading_vs_steering_cut')};result['statistics']=stats
  useful={'away_at_least_0_05_rad_count':sum(r['away_heading_vs_input_off']>=.05 for r in effects),'departure_at_least_0_1_mm_count':sum(r['final_xy_departure_mm']>=.1 for r in effects),'no_flip_count':sum(r['no_flips'] for r in effects)}
  gates={'causal_motor_effect':all(stats[k]['positive'] for k in ('motor_rms_vs_input_off','motor_rms_vs_steering_cut')),'directional_avoidance':all(stats[k]['positive'] for k in ('away_heading_vs_input_off','away_heading_vs_steering_cut')),'useful_response':useful['away_at_least_0_05_rad_count']>=16 and useful['departure_at_least_0_1_mm_count']>=16 and useful['no_flip_count']>=18,'exact_motor_replay':True};gates['progression_ready']=all(gates.values());result.update(useful_response_counts=useful,gates=gates)
  atomic_json(OUT/'results.json',result);plot(result);write_report(result,protocol)
 else:atomic_json(OUT/'partial-audit.json',result)
 print(json.dumps({k:result[k] for k in ('status','seed_count','full_brain_trials','motor_replay_exact')}))

def plot(r):
 import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
 effects=r['paired_effects'];x=np.arange(len(effects));colors=['#3373b4' if e['cue']=='loom_left' else '#c47635' for e in effects];fig,axes=plt.subplots(2,2,figsize=(12,8),constrained_layout=True)
 axes[0,0].bar(x,[e['motor_rms_vs_input_off'] for e in effects],color=colors);axes[0,0].set(title='Sensory-dependent command change',ylabel='RMS drive difference vs input-off',xlabel='Independent paired seed')
 axes[0,1].bar(x,[e['away_heading_vs_input_off'] for e in effects],color=colors);axes[0,1].axhline(.05,linestyle=':',color='#777777',label='Predeclared useful threshold');axes[0,1].axhline(0,color='#777777',linewidth=.5);axes[0,1].set(title='Away heading response',ylabel='Radians vs matched input-off',xlabel='Independent paired seed');axes[0,1].legend()
 axes[1,0].bar(x,[e['final_xy_departure_mm'] for e in effects],color=colors);axes[1,0].axhline(.1,linestyle=':',color='#777777',label='Predeclared useful threshold');axes[1,0].set(title='Actual physical departure',ylabel='Final xy difference from input-off (mm)',xlabel='Independent paired seed');axes[1,0].legend()
 for arm,col in (('intact','#3373b4'),('input_off','#777777'),('steering_cut','#c47635')):
  values=[v for v in r['records'] if v['arm']==arm];d=np.array([v['population_spikes']['DNa02_left']+v['population_spikes']['DNa02_right'] for v in values]);axes[1,1].bar(np.arange(3)[('intact','input_off','steering_cut').index(arm)],d.mean()/6,color=col,label=arm)
 axes[1,1].set(title='Mapped steering population firing',ylabel='Mean Hz across 2 cells and 20 seeds',xticks=range(3),xticklabels=['Intact','Input-off','Steering cut'])
 for a in axes.flat:a.grid(axis='y',alpha=.15)
 fig.suptitle('Fly Garden · Step 8 · 20 matched recorded-eye response trials\nBlue = left loom · orange = right loom · fixed controller, learning off');fig.savefig(OUT/'causal-effects.png',dpi=160);plt.close(fig)

def write_report(r,p):
 def state(key):return 'PASS' if r['gates'][key] else 'FAIL'
 comparisons='\n'.join(f"| {k.replace('_',' ')} | {s['mean']:.7f} | [{s['ci_97_5'][0]:.7f}, {s['ci_97_5'][1]:.7f}] |" for k,s in r['statistics'].items())
 counts=r['useful_response_counts'];effects=r['paired_effects'];intact=[a for a in r['records'] if a['arm']=='intact']
 p9_spikes=sum(a['population_spikes']['DNp09_left']+a['population_spikes']['DNp09_right'] for a in intact)
 steering_spikes=sum(a['population_spikes']['DNa02_left']+a['population_spikes']['DNa02_right'] for a in intact)
 dnp01_spikes=sum(a['population_spikes']['DNp01_left']+a['population_spikes']['DNp01_right'] for a in intact)
 text=f'''# Step 8 — controlled causality and readiness

The experiment completed 20 new independent input/gait seeds, balanced between ten left and ten right recorded looming cues. Each seed has three continuous full-brain/body runs plus two physical motor controls: 60 full-brain trials and 40 body-only trials. The previously fixed 25 ms coupling, sensory encoding and decoder were not tuned during this experiment. Learning remained disabled.

| Predeclared gate | Result |
|---|---|
| Sensory- and steering-dependent motor effect | {state('causal_motor_effect')} |
| Consistent away-directed heading effect | {state('directional_avoidance')} |
| Useful physical response threshold | {state('useful_response')} |
| Exact motor replay | {state('exact_motor_replay')} |
| Ready for Step 9 behavioral progression | {state('progression_ready')} |

A detectable causal motor effect is distinct from useful avoidance. The result does not establish online visual navigation, predator escape, contact-driven retreat or learning. No experimental controller is promoted into the live arena by this report.

## What the controls do

1. **Intact:** actual eye-image features recorded in Step 6 stimulate exact-root LPLC2 cells, the complete fixed network advances continuously, and DNa02/DNp09 activity produces motor commands through the fixed candidate decoder.
2. **Input-off:** encoded visual rates are zero. Neural parameters, initial neural state, input random-number schedule, gait seed and body remain matched.
3. **Steering interruption:** every incoming anatomical connection onto the two exact-root DNa02 cells has its weight set to zero. The model retains all 138,639 neurons and 15,091,983 connection records. This affects {r['interrupted_records']} records containing {r['interrupted_anatomical_synapses']} anatomical synapses. Every other weight is fixed. Original signed weights, indices and exact pre/post root IDs are saved. This artificial intervention tests the necessity of the mapped steering readout; it does not identify a uniquely biological LPLC2-to-DNa02 route.
4. **Motor replay:** replay the intact trial's held commands in a fresh same-seed physical body. Every complete physical integration state, observed body state and 30 Hz pose matches exactly in all twenty replays.
5. **Zero motor:** reuse the intact neural recording as a labeled body-only counterfactual and apply zero commands to a fresh same-seed body. Every physical state and recorded pose matches the independent full-brain input-off run exactly. The brain is not recomputed for this control.

Intact and interrupted trials receive identical external event IDs/timestamps per seed. Input-off has no external events or neuronal spikes. The interrupted DNa02 cells have no spikes. Raw spike histograms, population rates, every fixed decoder output, causal command boundaries, frame timing, source hashes, connection-mask identity and unchanged weight hashes are independently reconstructed and checked.

## Prechosen effects and uncertainty

Away heading is negative heading change for a left loom and positive heading change for a right loom, with angle unwrapping. It is measured relative to each matched control, rather than counting passive settling as a sensory response. Motor effects use RMS differences between actual applied command timelines. All paired effects are retained, including wrong-way turns, inactive responses, falls and stalls.

The table uses 10,000 paired bootstrap resamples, preserving ten left and ten right seeds, with RNG 8200. Each two-comparison family uses 97.5% percentile intervals for nominal simultaneous 95% coverage by Bonferroni adjustment. Twenty runs provide limited precision; these intervals describe this fixed cue protocol and seed distribution, not a broad claim about fly behavior.

| Paired effect | Mean | 97.5% bootstrap interval |
|---|---|---|
{comparisons}

The engineered useful-response criterion was saved before the first run: at least 16/20 seeds must gain 0.05 radians of away heading over input-off, at least 16/20 must depart by 0.1 mm from its final xy position, and at least 18/20 must have no flipped frames. Actual counts: **{counts['away_at_least_0_05_rad_count']}/20 heading**, **{counts['departure_at_least_0_1_mm_count']}/20 departure**, **{counts['no_flip_count']}/20 without flips**. These are progression targets, not biological norms or an escape-success measure.

Maximum final departure from input-off is {max(e['final_xy_departure_mm'] for e in effects):.6f} mm. Mean paired away heading is {np.mean([e['away_heading_vs_input_off'] for e in effects]):.7f} radians. There are {sum(e['stalled'] for e in effects)} active-motor stalls under the predeclared definition (RMS drive above 1e-4 and travel below 0.1 mm during 1.5–3 seconds).

## What reaches the movement decoder

Across the twenty intact trials, the selected DNa02 cells emit {steering_spikes} spikes, DNp09 emits {p9_spikes} spikes, and the monitored DNp01 visual-response cells emit {dnp01_spikes} spikes. The fixed walking decoder derives forward drive from DNp09 and steering from DNa02; it does not turn DNp01 firing into a walking escape reflex. This separates visually responsive neural activity from the signals this body actually uses. No tonic drive or substituted escape controller was added after observing these runs.

## Scope and next decision

This is a recorded-eye stimulus→brain→command→body causal experiment. The original clips come from the actual physical body's eye cameras, but the stimulated brain does not receive new images from its own changing pose in this stage. Recorded input allows exact event matching and motor counterfactuals; it cannot validate visually closed-loop navigation. Contact-to-brain input is disabled, while the supplied gait controller retains its own contact corrections. The Step 7 touch limitation remains: the existing force proxy did not recruit MDN. No coordinates, route or game score reach this controller.

The brain parameters and learned weights are identical across the twenty initial states; seeds vary stochastic stimulus timing and gait initialization. They are not twenty independently trained animals. No gain, cue timing, interval, outcome threshold or decoder was fitted after reading these results.

Proceeding to navigation requires resolving any failed directional or useful-response gate first. If the causal motor gate passes but the useful gate fails, the evidence supports small activity-dependent movement, not a working foraging brain. Preserve this result rather than adding a hidden navigator, tonic drive or escape rule.

## Runtime and retained evidence

- Complete network: 138,639 neurons, 15,091,983 aggregated connection records, 54,492,922 anatomical synapses before the explicitly recorded intervention.
- 180 simulated seconds of full-brain/body computation. Completed neural-arm wall time: {r['completed_neural_arm_wall_seconds']:.1f} s; body-only control wall time: {r['body_control_wall_seconds']:.1f} s.
- Peak neural worker memory: {r['peak_worker_rss_bytes']/1024**3:.2f} GiB. Evidence storage: {r['storage_bytes']/1024**2:.1f} MiB. Free disk: {r['free_bytes']/1024**3:.1f} GiB; the 2 GiB reserve remains enforced.
- Seven focused measurement, physical-state, clock and checkpoint tests passed before the protocol was committed.
- Immutable protocol/source hashes, all-neuron raw spike chunks, external events, body states, 30 Hz poses, per-seed geometry, interrupted edges, motor replays and outcomes remain alongside this report. Interrupted attempts are preserved rather than silently overwritten. The battery shutdown interrupted one input-off attempt; it was archived with its complete chunks and restarted from the same initial protocol state. The seven finished matched seeds and the already complete eighth intact arm were retained without recomputation.

![Paired effects](causal-effects.png)

Run `.venv-next/bin/python scripts/diagnose_causal_behavior.py` for the committed sequential batch and `.venv-next/bin/python scripts/report_causal_behavior.py` to audit completed artifacts. Completed cases are retained; existing incomplete cases must be preserved before retrying. `--partial` audits completed seeds without publishing final statistics or pass/fail gates.
'''
 (OUT/'RESULTS.md').write_text(text)
 atomic_json(OUT/'runtime.json',{'python':platform.python_version(),'platform':platform.platform(),'packages':{k:metadata.version(k) for k in ('brian2','numpy','pandas','scipy','mujoco','matplotlib','flygym')},'production_controller_modified':False})

if __name__=='__main__':report('--partial' in sys.argv)
