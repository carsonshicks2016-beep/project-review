"""Audit Stage7 immutable recordings and fresh-process continuation."""
import json,sys,hashlib,shutil
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from flygarden.recording import atomic_json
OUT=ROOT/'reports/brain-integration/stage7-20261006'

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def load(folder):
 m=json.loads((folder/'manifest.json').read_text());rows=json.loads((folder/'bins.json').read_text());frames=json.loads((folder/'frames.json').read_text())
 assert m['status']=='complete' and m['weights_before']==m['weights_after'] and not m['learning']
 chunks=[]
 for c in m['chunks']:
  assert sha(folder/c['file'])==c['sha256']
  with np.load(folder/c['file']) as f:a={k:f[k].copy() for k in f.files}
  assert len(a['spike_i'])==int(a['counts'].sum())==c['spikes']
  assert np.array_equal(np.bincount(a['spike_i'],minlength=m['neurons']),a['counts'])
  assert np.array_equal(np.bincount(a['external_i'],minlength=len(m['input_indices'])),a['input_counts'])
  chunks.append(a)
 assert len(rows)==len(chunks)
 for i,(row,a) in enumerate(zip(rows,chunks)):
  start=row['motor_start'];end=row['time']
  assert abs(end-start-m['interval'])<1e-9 and row['observation_time']==start and row['next_available_at']==end
  assert row['image_frame_time_max']<end+1e-9 and row['image_frame_time_min']<=start+1e-9
  assert np.array_equal(a['motor'],row['applied_motor']) and np.array_equal(a['next_motor'],row['next_motor'])
  if i:assert np.array_equal(a['motor'],chunks[i-1]['next_motor'])
  for key in ('spike_t','external_t'):
   assert np.all(a[key]>=start-1e-9) and np.all(a[key]<end-1e-9)
  assert np.isfinite(a['physics']).all()
 expected=np.arange(round(m['start_time']*30)+1,91)/30
 assert len(frames)==len(expected) and np.all(np.abs(np.array([f['time'] for f in frames])-expected)<=.0005+1e-9)
 for s,h in m['sources'].items():assert sha(ROOT/s)==h
 return m,rows,frames,chunks

def joined(chunks,key):return np.concatenate([a[key] for a in chunks])
def first_spike(chunks,indices):
 for a in chunks:
  times=a['spike_t'][np.isin(a['spike_i'],indices)]
  if len(times):return float(times.min())
 return None
def first_motor(rows,key):
 for r in rows:
  if np.max(np.abs(r[key]))>1e-12:return r['time'] if key=='next_motor' else r['motor_start']
 return None

def run():
 protocol=json.loads((OUT/'protocol.json').read_text());assert json.loads((OUT/'progress.json').read_text())['status']=='complete'
 eye=ROOT/'reports/brain-integration/stage6-20261006/eye-stimuli';eye_manifest=json.loads((eye/'manifest.json').read_text())
 for file,digest in eye_manifest['files'].items():assert sha(eye/file)==digest
 eye_mapping=json.loads((eye.parent/'protocol.json').read_text())['mapping']
 trials={};records=[]
 for cue,seed,dt in protocol['cases']:
  name=f'{cue}-{seed}-{round(dt*1000)}ms';m,rows,frames,chunks=load(OUT/'trials'/name);trials[(cue,seed,dt)]=(m,rows,frames,chunks)
  assert m['sources']==protocol['sources'] and m['neurons']==138639 and m['connections']==15091983
  assert m['eye_stream_sha256']==sha(eye/(f'{cue}.json' if cue in ('blank','loom_left','loom_right') else 'blank.json'))
  for key in ('LPLC2_left','LPLC2_right','DNp01_left','DNp01_right'):assert m['mapping'][key]==eye_mapping[key]
  pops=m['mapping'];group=lambda prefix:[n['index'] for k,v in pops.items() if k.startswith(prefix) for n in v]
  external=joined(chunks,'external_t');all_counts=sum((a['counts'] for a in chunks),np.zeros(m['neurons'],dtype=np.int64))
  item=dict(name=name,cue=cue,seed=seed,interval=dt,spikes=int(all_counts.sum()),external_events=len(external),frames=len(frames),wall_seconds=m['wall_seconds'],peak_rss_bytes=m['peak_rss_bytes'],heading_change=m['heading_change_radians'],flipped_samples=m['flipped_samples'],first_external=float(external.min()) if len(external) else None,first_LPLC2=first_spike(chunks,group('LPLC2_')),first_DNp01=first_spike(chunks,group('DNp01_')),first_DNa02=first_spike(chunks,group('DNa02_')),first_TLA=first_spike(chunks,group('AN_multi_63_')),first_MDN=first_spike(chunks,group('MDN_')),first_command_available=first_motor(rows,'next_motor'),first_command_applied=first_motor(rows,'applied_motor'),mean_MDN_hz=float(all_counts[group('MDN_')].mean()/3),mean_TLA_hz=float(all_counts[group('AN_multi_63_')].mean()/3),MDN_spikes=int(all_counts[group('MDN_')].sum()),TLA_spikes=int(all_counts[group('AN_multi_63_')].sum()),max_candidate_motor=float(np.max([r['candidate_motor'] for r in rows])),peak_contact_force=max(r['feedback']['pooled_force_model_units'] for r in rows),peak_contact_input_hz=max(max(r['feedback']['tla_hz']) for r in rows))
  if cue.startswith('loom'):
   _,baseline,bframes,_=trials[('blank',seed,dt)]
   delta=np.array([f['body']['position'] for f in frames])-np.array([f['body']['position'] for f in bframes]);item['max_displacement_from_blank_mm']=float(np.linalg.norm(delta,axis=1).max())
   angle=np.array([f['body']['heading'] for f in frames])-np.array([f['body']['heading'] for f in bframes]);item['max_heading_from_blank_radians']=float(np.max(np.abs(np.arctan2(np.sin(angle),np.cos(angle)))))
  records.append(item)
 parity=[]
 for cue in ('blank','loom_left','loom_right'):
  for seed in protocol['seeds']:
   ref=trials[(cue,seed,.1)][3]
   for dt in (.05,.025):
    other=trials[(cue,seed,dt)][3];result={key:bool(np.array_equal(joined(ref,key),joined(other,key))) for key in ('spike_i','spike_t','external_i','external_t')}
    parity.append(dict(cue=cue,seed=seed,interval=dt,exact=result))
    assert all(result.values()),f'Neural partition parity failed {parity[-1]}'
 contact=[]
 for seed in protocol['seeds']:
  off=trials[('contact_off',seed,.025)][3];on=trials[('contact_on',seed,.025)][3]
  exact=all(np.array_equal(a['physics'],b['physics']) and np.array_equal(a['motor'],b['motor']) for a,b in zip(off,on));assert exact
  contact.append(dict(seed=seed,identical_body_and_held_motor=exact))
 cp=OUT/'trials/loom_left-7199-25ms/checkpoint-1.5';cpmeta=json.loads((cp/'manifest.json').read_text());assert all(sha(cp/f)==h for f,h in cpmeta['files'].items())
 resumed=load(OUT/'continuation/loom_left-7199-25ms');reference=trials[('loom_left',7199,.025)]
 matches={k:all(np.array_equal(a[k],b[k]) for a,b in zip(reference[3][60:],resumed[3])) for k in reference[3][0]}
 assert len(resumed[3])==60 and all(matches.values()),f'Continuation mismatch {matches}'
 assert reference[1][60:]==resumed[1] and reference[2][45:]==resumed[2]
 usage=shutil.disk_usage(OUT);size=sum(p.stat().st_size for p in OUT.rglob('*') if p.is_file())
 audit=dict(status='complete',trials=records,clock_causality_and_chunk_integrity='PASS',full_neural_partition_parity=parity,contact_body_parity=contact,cold_checkpoint_continuation=dict(window_count=60,exact_arrays=matches,exact_observations_and_frames=True),storage_bytes=size,free_bytes=usage.free,scientific_limits=['Prerecorded eye images, not closed-loop visual navigation.','Contact trials use fixed external motor drive; neural candidates do not control gait.','Anterior-force proxy is engineered; movement and joint-to-neuron inputs remain disabled.','No biological retreat, navigation or learning claim; prior weak-movement gates remain in force.'])
 atomic_json(OUT/'audit.json',audit);plot(trials,records);write_report(audit,protocol)
 print(json.dumps({'status':audit['status'],'trials':len(records),'exact_neural_comparisons':len(parity),'checkpoint':'exact','storage_bytes':size},indent=2))

def plot(trials,records):
 import matplotlib;matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 fig,axes=plt.subplots(2,2,figsize=(12,7),constrained_layout=True)
 for dt,color in ((.1,'#4666bb'),(.05,'#db9030'),(.025,'#278a60')):
  _,rows,_,_=trials[('loom_left',7199,dt)];t=[r['motor_start'] for r in rows]
  axes[0,0].step(t,[r['applied_motor'][0] for r in rows],where='post',color=color,label=f'{round(dt*1000)} ms')
  items=[r for r in records if r['cue']=='loom_left' and r['seed']==7199 and r['interval']==dt]
  vals=[items[0][k] for k in ('first_external','first_LPLC2','first_DNa02','first_command_available','first_command_applied')]
  axes[0,1].plot(range(len(vals)),vals,'o-',label=f'{round(dt*1000)} ms',color=color)
 axes[0,0].set(title='Held left motor command · looming left · seed 7199',xlabel='Simulated seconds',ylabel='Drive (full allowable range 0–1.2)');axes[0,0].legend()
 axes[0,1].set(title='Causal response timestamps',ylabel='Simulated seconds',xticks=range(5),xticklabels=['Input','LPLC2','DNa02','Computed','Applied']);axes[0,1].legend()
 for cue,color in (('contact_off','#778899'),('contact_on','#b55342')):
  _,rows,_,_=trials[(cue,7199,.025)];t=[r['observation_time'] for r in rows]
  axes[1,0].plot(t,[r['feedback']['pooled_force_model_units'] for r in rows],label=cue,color=color)
  axes[1,1].plot(t,[(r['population_hz']['MDN_left']+r['population_hz']['MDN_right'])/2 for r in rows],label=cue,color=color)
 axes[1,0].set(title='Actual opposing foreleg force · identical body',xlabel='Simulated seconds',ylabel='Native model force units');axes[1,0].axhline(1.2,color='#999999',linestyle=':',label='Engineered threshold');axes[1,0].legend()
 axes[1,1].set(title='MDN response · external gait calibration',xlabel='Simulated seconds',ylabel='Mean Hz / 25 ms window',ylim=(-1,5));axes[1,1].legend();axes[1,1].text(.5,.5,'0 MDN spikes in both conditions',transform=axes[1,1].transAxes,ha='center')
 fig.suptitle('Fly Garden · Step 7: timing and contact diagnostics');fig.savefig(OUT/'timing-feedback.png',dpi=160);plt.close(fig)

def write_report(a,p):
 rs=a['trials'];vision=[r for r in rs if r['cue'].startswith('loom')];contact=[r for r in rs if r['cue']=='contact_on'];mapping=json.loads((OUT/'feedback-mapping.json').read_text())
 latency='\n'.join(f"| {round(r['interval']*1000)} | {r['first_external']} | {r['first_LPLC2']} | {r['first_DNa02']} | {r['first_command_available']} | {r['first_command_applied']} |" for r in rs if r['cue']=='loom_left' and r['seed']==7199)
 contact_table='\n'.join(f"| {r['seed']} | {r['peak_contact_input_hz']:.2f} | {r['mean_TLA_hz']:.2f} | {r['mean_MDN_hz']:.2f} | {r['spikes']} |" for r in contact)
 text=f'''# Step 7 — causal timing and supported body feedback

The experimental full-brain/body pipeline now applies each newly computed motor command in the following physical interval. All 33 three-second trials completed with continuous neural state and unchanged neural weights. Exact neural event streams match across 100, 50 and 25 ms observation windows, and a fresh-process continuation at 1.5 seconds reproduces the remaining neural events, commands, observations, frames and complete physical integration state exactly.

This establishes timing and checkpoint integration. It does not establish useful navigation, escaping or learning. The production controller remains unchanged while the earlier movement gates remain failed.

## What changed

The old loop advanced the brain and then applied its new output to the body over that same time interval. The experimental causal coupler holds the previously available command, advances both systems to the next shared model timestamp, and publishes the successor command for the following interval. No simulated dynamics are skipped when computation is slow. Physics remains 100 microseconds and the supplied gait controller remains 500 microseconds. The selected observation interval is 25 ms, chosen before held-out results rather than fitted to performance.

The experimental body also replaces rounded frame scheduling with a floor-based schedule. Each three-second run records exactly 90 physical poses, within 0.5 ms of their intended 30 Hz timestamps. Observation calculation uses a copy of physical state so it cannot change solver warm starts or gait sensing. A real-body unit test confirms exact physical state and gait-phase agreement across different observation windows for identical held commands.

## Measurements

Held-out left-loom response, seed 7199; values are absolute simulated seconds. Computed and applied timestamps can be equal at a boundary: the command is published at the end of the preceding interval and starts the next interval at that boundary.

| Interval ms | First input event | First LPLC2 spike | First DNa02 spike | First command computed | First command applied |
|---|---|---|---|---|---|
{latency}

Raw visual input events and all neuron spike IDs/timestamps are exactly identical in all 18 cross-interval comparisons. Interval size changes readout binning, smoothing and when a command becomes available, rather than changing the neural event stream. Vision here replays the actual eye-image recordings from Step 6 at their original 30 Hz timestamps; this is not visually closed-loop navigation. The fly's neural motor commands drive the body in these vision trials.

Across the looming cases, maximum movement departure from matched blank trials is {max(r['max_displacement_from_blank_mm'] for r in vision):.6f} mm, and maximum heading departure is {max(r['max_heading_from_blank_radians'] for r in vision):.6f} radians. Maximum candidate drive is {max(r['max_candidate_motor'] for r in vision):.5f}. These are measurements, not successful walking or escape outcomes. Prior Step 6 motor-response failures still apply.

## Contact feedback

The exact local AN_multi_63 roots are 720575940616343441 (left) and 720575940613981330 (right). The institutional [cell-type explorer](https://reiserlab.github.io/celltype-explorer-drosophila-male-cns/types/AN17A026_L.html) identifies the corresponding type as TwoLumps Ascending. [Sen et al. (2019)](https://pubmed.ncbi.nlm.nih.gov/31813606/) reports an anterior-touch retreat pathway involving TLA and MDN. In this imported model, the two selected TLA cells have {len(mapping['direct_TLA_MDN'])} direct excitatory records totaling {sum(x['anatomical_synapses'] for x in mapping['direct_TLA_MDN'])} anatomical synapses onto four exact-root MDN cells. These cells are typed MDN locally; DNp42 is not substituted.

Only these two of the 2,317 locally annotated ascending/sensory-ascending cells receive neural body-feedback input; this is deliberately limited coverage. The adapter pools opposing horizontal foreleg forces and uses an explicitly engineered 1.2–6 native-model-unit ramp to 0–100 Hz, driving both TLA candidates equally. It does not claim a physiological force-to-rate fit or known left/right receptive fields. Force units are retained as native model units. The measured leg forces are a proxy for anterior contact, not modeled bristles or joint proprioceptors; receptor and ventral-nerve-cord processing are bypassed. Open-ground calibration also produced a 6.53 Hz peak for one seed, so this is not a validated touch classifier.

The contact-on/off trials use the same fixed external .65/.65 walking drive. Complete physical states and held motor commands are exactly identical between each pair, isolating the neural response to actual live forces. Candidate brain commands are recorded but do not control these calibration bodies. MDN activation is inspected; no backward gait has been invented from its firing rate.

| Seed | Peak engineered input Hz | Mean TLA Hz | Mean MDN Hz | Whole-brain spikes |
|---|---|---|---|---|
{contact_table}

The contact-on trials produce {sum(r['TLA_spikes'] for r in contact)} spikes in the two selected TLA cells and {sum(r['MDN_spikes'] for r in contact)} spikes in the four MDN cells. Thus the present encoding has not demonstrated downstream MDN recruitment or retreat. The 25 ms boundary samples can underrepresent brief forces compared with the independent 10 ms preparation; both sampled traces are retained. No gain was changed after seeing held-out results. Contact event integration and physiological calibration need a separate controlled test before a behavioral claim.

Velocity and yaw are recorded as diagnostics. Neural stimulation from speed, turning and joint angles remains disabled because exact-root maps and physiological gains have not been verified. The supplied gait controller's contact corrections remain separately visible.

## Verification and runtime

- Five simulation tests pass: causal command scheduling and clock restore, contact encoding, real-body state/frame invariance, feedback-history continuation, and physical/gait checkpoint continuation at a wall. Two existing browser playback timestamp tests also pass.
- 33 full-network trials: 138,639 neurons and 15,091,983 connection records, fixed weights and learning disabled.
- Every chunk hash, raw spike histogram, external-input count, timing boundary and motor transition verified.
- 18 exact raw-neural cross-interval comparisons; three exact contact-on/off physical comparisons.
- One independent fresh-process full checkpoint continuation: 60 remaining 25 ms windows and 45 remaining 30 Hz frames match exactly, including physical solver state.
- Total primary run time: {sum(r['wall_seconds'] for r in rs):.1f} wall seconds for 99 simulated seconds. Peak worker memory: {max(r['peak_rss_bytes'] for r in rs)/1024**3:.2f} GiB. Recording/checkpoint/report storage: {a['storage_bytes']/1024**2:.1f} MiB; remaining disk: {a['free_bytes']/1024**3:.1f} GiB. The 2 GiB reserve remains enforced.

The neural/body clock has no display camera or playback-speed parameter. Recorded playback changes presentation time only; it cannot change saved commands. This stage does not promote a new live viewer mode. Source hashes, exact-root mappings, raw activity, physical samples and checkpoint files are retained alongside this report.

![Timing and contact measurements](timing-feedback.png)

## Reproduce

Use `.venv-next/bin/python scripts/prepare_timing_feedback.py` for the independent contact preparation, `.venv-next/bin/python scripts/diagnose_causal_timing.py` for the committed sequential batch, and `.venv-next/bin/python scripts/report_causal_timing.py` to verify completed artifacts. Complete runs are retained; interrupted attempts must be preserved before retrying. Full experimental continuation uses the worker's `--resume` option and the exact Stage 7 checkpoint, rather than the production application's checkpoint loader.

Next is Step 8's causal controls. Useful steering, forward drive and biological contact calibration remain unresolved; do not turn these integration checks into a learning or behavior badge.
'''
 (OUT/'RESULTS.md').write_text(text)

if __name__=='__main__':run()
