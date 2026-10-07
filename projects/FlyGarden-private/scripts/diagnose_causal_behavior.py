"""Stage8: twenty matched recorded-eye causal comparisons; no policy tuning."""
import sys,json,time,hashlib,subprocess,resource,argparse,copy
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from flygarden.recording import atomic_json,space_check
from flygarden.synchronized import SynchronizedBody,CausalCoupling
from flygarden.descending import DescendingDecoder
from flygarden.causal_metrics import trajectory_metrics
from scripts.diagnose_causal_timing import mapping
OUT=ROOT/'reports/brain-integration/stage8-20261006';DT=.025;DURATION=3.;SEEDS=tuple(range(8101,8121));ARMS=('intact','input_off','steering_cut')
SOURCES=('scripts/diagnose_causal_behavior.py','flygarden/causal_metrics.py','flygarden/synchronized.py','flygarden/descending.py','flygarden/brain.py','flygarden/body.py','flygarden/plasticity.py','scripts/diagnose_causal_timing.py','data/annotations.tsv','vendor/fly-brain/data/2025_Completeness_783.csv','vendor/fly-brain/data/2025_Connectivity_783.parquet','reports/brain-integration/stage7-20261006/feedback-mapping.json','reports/brain-integration/stage6-20261006/protocol.json','reports/brain-integration/stage6-20261006/eye-stimuli/manifest.json')
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for block in iter(lambda:f.read(8*1024**2),b''):h.update(block)
 return h.hexdigest()
def sources():return {s:sha(ROOT/s) for s in SOURCES}
def cue_for(seed):return 'loom_left' if seed%2 else 'loom_right'
def ready(folder):
 if not folder.exists():return False
 if not (folder/'manifest.json').exists():raise RuntimeError(f'Preserve incomplete attempt before retry: {folder}')
 m=json.loads((folder/'manifest.json').read_text())
 if m['status']!='complete':raise RuntimeError(f'Preserve incomplete attempt before retry: {folder}')
 assert m['sources']==sources();return True
def worker(seed):
 import brian2 as b
 from flygarden.brain import FullBrain
 started=time.perf_counter();cue=cue_for(seed);pops=mapping();brain=FullBrain(seed=seed,learning=False,record_spikes=True)
 keys=('LPLC2_left','LPLC2_right','AN_multi_63_left','AN_multi_63_right');groups=[np.array([n['index'] for n in pops[k]],dtype=np.int32) for k in keys];targets=np.concatenate(groups);channels=np.concatenate([np.full(len(g),i,dtype=np.int32) for i,g in enumerate(groups)])
 assert all(str(brain.ids[n['index']])==n['root_id'] for cells in pops.values() for n in cells)
 gen=b.SpikeGeneratorGroup(len(targets),np.array([],dtype=np.int32),np.array([])*b.second,clock=brain.clock);ext=b.Synapses(gen,brain.neurons,on_pre='v_post += 68.75*mV',clock=brain.clock);ext.connect(i=np.arange(len(targets)),j=targets);im=b.SpikeMonitor(gen,record=False);brain.network.add(gen,ext,im);brain.inputs.active=False;brain.input.active=False;brain.neurons.rfc=2.2*b.ms;brain.neurons.rfc[groups[0 if cue=='loom_left' else 1]]=0*b.ms
 pre=np.asarray(brain.synapses.i[:],dtype=np.int32);post=np.asarray(brain.synapses.j[:],dtype=np.int32);steering=np.array([n['index'] for k in ('DNa02_left','DNa02_right') for n in pops[k]],dtype=np.int32);cut=np.flatnonzero(np.isin(post,steering));base=np.asarray(brain.synapses.w[:]/b.mV).copy();base_hash=hashlib.sha256(base.tobytes()).hexdigest();expected=base.copy();expected[cut]=0;cut_hash=hashlib.sha256(expected.tobytes()).hexdigest()
 seed_folder=OUT/'trials'/str(seed);seed_folder.mkdir(parents=True,exist_ok=True)
 lesion_file=seed_folder/'interrupted-edges.npz'
 if not lesion_file.exists():
  with lesion_file.with_suffix('.tmp').open('wb') as f:np.savez_compressed(f,indices=cut,pre=pre[cut],post=post[cut],original_w_mV=base[cut],pre_root_ids=brain.ids[pre[cut]],post_root_ids=brain.ids[post[cut]])
  lesion_file.with_suffix('.tmp').replace(lesion_file)
 eye_file=ROOT/'reports/brain-integration/stage6-20261006/eye-stimuli'/f'{cue}.json';eye_rows=json.loads(eye_file.read_text());eye_rates=np.array([[r['features'][s]['lplc2_hz'] for s in range(2)]+[0.,0.] for r in eye_rows]);brain.network.store('stage8_initial')
 for arm in ARMS:
  folder=seed_folder/arm
  if ready(folder):continue
  brain.network.restore('stage8_initial',restore_random_state=False);brain.clear_recorded_spikes();assert float(brain.network.t/b.second)==0 and not np.asarray(brain.monitor.count[:]).any()
  if arm=='steering_cut':brain.synapses.w[cut]=0*b.mV
  expected_hash=cut_hash if arm=='steering_cut' else base_hash;assert hashlib.sha256(np.asarray(brain.synapses.w[:]/b.mV).tobytes()).hexdigest()==expected_hash
  body=SynchronizedBody(seed=seed);coupling=CausalCoupling(DT);decoder=DescendingDecoder();rng=np.random.default_rng(seed);previous=np.zeros(brain.n,dtype=np.int64);prior_input=np.zeros(len(targets),dtype=np.int64);frames=[];rows=[];initial=body.observation();origin=body.sim.mj_data.time
  folder.mkdir(exist_ok=False);t0=time.perf_counter();m={'schema_version':1,'status':'running','seed':seed,'cue':cue,'arm':arm,'sources':sources(),'neurons':brain.n,'connections':brain.edges,'anatomical_synapses':brain.anatomical_synapses,'learning':False,'mapping':pops,'input_indices':targets.tolist(),'input_root_ids':brain.ids[targets].astype(str).tolist(),'channels':channels.tolist(),'interval':DT,'duration':DURATION,'initial':initial,'eye_source_sha256':sha(eye_file),'baseline_weights_sha256':base_hash,'expected_weights_sha256':expected_hash,'intervention':{'description':'Zero incoming anatomical connection weights onto both exact-root DNa02 cells; all other weights fixed.' if arm=='steering_cut' else 'All visual input rates zero; same neural parameters and random-number schedule.' if arm=='input_off' else 'Intact fixed network and fixed decoder.','interrupted_records':len(cut) if arm=='steering_cut' else 0,'edge_file':'../interrupted-edges.npz','edge_sha256':sha(lesion_file)},'sensory_scope':'Recorded stationary-pose actual eye images; no live scene/pose feedback or contact-to-brain input.','motor_identity':decoder.version,'chunks':[]};atomic_json(folder/'manifest.json',m);atomic_json(folder/'geometry.json',body.geometry())
  try:
   for tick in range(round(DURATION/DT)):
    t=coupling.time;assert abs(float(brain.network.t/b.second)-t)<1e-9 and abs(body.steps*body.dt-t)<1e-9 and abs(body.sim.mj_data.time-origin-t)<1e-8
    micro=np.arange(coupling.ticks,coupling.ticks+coupling.ticks_per_window);fi=micro*3//1000;rates=eye_rates[fi].copy()
    if arm=='input_off':rates[:]=0
    assert np.all(np.array([eye_rows[i]['time'] for i in fi])<=micro*.0001+1e-9)
    sample=rng.random((len(micro),len(targets)))<rates[:,channels]*.0001;local,ii=np.nonzero(sample);event_times=micro[local]*.0001;gen.set_spikes(ii,event_times*b.second,sorted=True);block={}
    def advance(dt):
     nonlocal previous,prior_input
     brain.network.run(dt*b.second,namespace={});counts=np.asarray(brain.monitor.count[:],dtype=np.int64);block['counts']=counts-previous;previous=counts.copy();ic=np.asarray(im.count[:],dtype=np.int64);block['input_counts']=ic-prior_input;prior_input=ic.copy();block['spike_i']=np.asarray(brain.monitor.i[:],dtype=np.int32).copy();block['spike_t']=np.asarray(brain.monitor.t[:]/b.second).copy();assert block['counts'].sum()==len(block['spike_i']);block['rates']={k:float(block['counts'][[n['index'] for n in v]].mean()/dt) for k,v in pops.items()};return decoder.advance(dt,block['rates'])
    tr=coupling.advance(advance,lambda dt,motor:body.advance(dt,motor,capture=lambda stamp,pose:frames.append({'time':stamp,**pose})))
    assert abs(float(brain.network.t/b.second)-coupling.time)<1e-9 and abs(body.steps*body.dt-coupling.time)<1e-9 and abs(body.sim.mj_data.time-origin-coupling.time)<1e-8
    rows.append({'time':coupling.time,'observation_time':t,'image_time_min':eye_rows[int(fi[0])]['time'],'image_time_max':eye_rows[int(fi[-1])]['time'],'motor_start':tr['start'],'motor_end':tr['end'],'applied_motor':tr['applied_motor'],'next_motor':tr['next_motor'],'next_available_at':tr['next_available_at'],'population_hz':block['rates'],'body':tr['body'],'input_hz_mean':rates.mean(axis=0).tolist(),'total_spikes':len(block['spike_i'])})
    file=folder/f'window-{tick:04d}.npz';space_check(folder,block['counts'].nbytes+12*len(block['spike_i'])+1024**2)
    with file.with_suffix('.tmp').open('wb') as f:np.savez_compressed(f,counts=block['counts'],input_counts=block['input_counts'],spike_i=block['spike_i'],spike_t=block['spike_t'],external_i=ii.astype(np.int32),external_t=event_times,physics=body.snapshot()['physics'],motor=np.array(tr['applied_motor']),next_motor=np.array(tr['next_motor']))
    file.with_suffix('.tmp').replace(file);brain.clear_recorded_spikes();m['chunks'].append({'file':file.name,'sha256':sha(file),'time':coupling.time,'spikes':len(block['spike_i'])});atomic_json(folder/'manifest.json',m);atomic_json(folder/'bins.json',rows);atomic_json(folder/'frames.json',frames)
   after=hashlib.sha256(np.asarray(brain.synapses.w[:]/b.mV).tobytes()).hexdigest();assert after==expected_hash and brain.plasticity.updates==0 and np.isfinite(brain.neurons.v[:]/b.mV).all() and np.isfinite(brain.neurons.g[:]/b.mV).all()
   m.update(status='complete',weights_after_sha256=after,wall_seconds=time.perf_counter()-t0,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,frames=len(frames),metrics=trajectory_metrics(initial,frames,rows,cue));atomic_json(folder/'manifest.json',m);print(f'{seed} {cue} {arm} complete',flush=True)
  finally:body.close()
 atomic_json(seed_folder/'worker.json',{'status':'complete','seed':seed,'wall_seconds':time.perf_counter()-started,'peak_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'sources':sources()})
def body_controls(seed):
 source=OUT/'trials'/str(seed)/'intact';m=json.loads((source/'manifest.json').read_text());assert m['status']=='complete' and m['sources']==sources();rows=json.loads((source/'bins.json').read_text());reference_frames=json.loads((source/'frames.json').read_text());reference_physics=[]
 for c in m['chunks']:
  assert sha(source/c['file'])==c['sha256']
  with np.load(source/c['file']) as f:reference_physics.append(f['physics'].copy())
 for arm in ('motor_replay','motor_zero'):
  folder=source.parent/arm
  if ready(folder):continue
  body=SynchronizedBody(seed=seed);frames=[];out_rows=[];initial=body.observation();folder.mkdir(exist_ok=False);start=time.perf_counter();meta={'schema_version':1,'status':'running','arm':arm,'seed':seed,'cue':m['cue'],'sources':sources(),'source_manifest_sha256':sha(source/'manifest.json'),'neural_source':'../intact','brain_recomputed':False,'scope':'Body-only counterfactual; fixed prerecorded eye input makes neural recordings reusable. No claim of closed-loop sensory replay.','interval':DT,'duration':DURATION,'initial':initial,'chunks':[]};atomic_json(folder/'manifest.json',meta)
  try:
   for tick,row in enumerate(rows):
    motor=np.array(row['applied_motor']) if arm=='motor_replay' else np.zeros(2);obs=body.advance(DT,motor,capture=lambda t,pose:frames.append({'time':t,**pose}));physics=body.snapshot()['physics'];exact=bool(np.array_equal(physics,reference_physics[tick]))
    if arm=='motor_replay':assert exact and obs==row['body']
    out_rows.append({'time':row['time'],'motor_start':row['motor_start'],'applied_motor':motor.tolist(),'body':obs});file=folder/f'window-{tick:04d}.npz'
    space_check(folder,physics.nbytes+1024**2)
    with file.with_suffix('.tmp').open('wb') as f:np.savez_compressed(f,physics=physics,motor=motor)
    file.with_suffix('.tmp').replace(file);meta['chunks'].append({'file':file.name,'sha256':sha(file),'time':row['time'],'exact_reference_physics':exact});atomic_json(folder/'manifest.json',meta);atomic_json(folder/'frames.json',frames);atomic_json(folder/'bins.json',out_rows)
   if arm=='motor_replay':assert frames==reference_frames
   meta.update(status='complete',wall_seconds=time.perf_counter()-start,frames=len(frames),exact_reference_frames=frames==reference_frames,metrics=trajectory_metrics(initial,frames,out_rows,m['cue']));atomic_json(folder/'manifest.json',meta)
  finally:body.close()
def commit_protocol():
 OUT.mkdir(parents=True,exist_ok=True);file=OUT/'protocol.json'
 if not file.exists():
  atomic_json(file,{'schema_version':1,'sources':sources(),'seeds':SEEDS,'cue_assignment':{str(s):cue_for(s) for s in SEEDS},'main_seeded_trials':20,'neural_arms':ARMS,'body_only_arms':['motor_replay','motor_zero'],'duration_seconds':DURATION,'interval_seconds':DT,'fixed_parameters':'Identical Step7 visual encodings, neural equations and fixed decoder; no calibration or performance tuning.','intervention':'Incoming anatomical weights to both DNa02 roots zeroed only in steering_cut. All neurons/records retained; full interruption indices and original signed weights recorded.','predeclared_metrics':{'causal_output':'RMS difference of applied motor timelines: intact vs input_off and intact vs steering_cut. Both paired bootstrap lower limits must exceed zero.','direction':'Away-signed final unwrapped heading: intact minus input_off and intact minus steering_cut. Both paired bootstrap lower limits must exceed zero.','useful_response':'At least16/20 seeds gain ≥0.05rad away heading over input_off; at least16/20 move ≥0.1mm from the matched input_off final xy position; at least18/20 have no flipped frames. Engineered progression criteria, not biological norms.','replay':'All twenty held-command body replays must reproduce every complete physical integration state and every30Hz pose exactly.','secondary':['Travel distance','Signed heading','Motor RMS','Stalls: active motor RMS>1e-4 and travel<0.1mm in1.5–3.0s','Raw population activity and stimulus→output timing'],'bootstrap':'10000 paired resamples stratified to preserve10left/10right, RNG8200;97.5% percentile intervals for simultaneous95% coverage of two comparisons within each metric family.'},'scope':'Causal response to fixed recorded-eye cues, not online perception, navigation, escape success, contact behavior or learning. Twenty independent stochastic input/gait seeds share fixed initial neural parameters and learned weights. No production promotion.'})
 assert json.loads(file.read_text())['sources']==sources()
def run():
 commit_protocol();progress={'status':'running','planned_seeds':20,'planned_full_brain_trials':60,'completed':[]};started=time.perf_counter()
 try:
  for seed in SEEDS:
   progress['current']={'seed':seed,'cue':cue_for(seed)};atomic_json(OUT/'progress.json',progress);space_check(OUT,128*1024**2);log=OUT/'logs'/f'{seed}.log';log.parent.mkdir(exist_ok=True)
   if not all(ready(OUT/'trials'/str(seed)/arm) for arm in ARMS):
    with log.open('a') as f:subprocess.run([sys.executable,str(Path(__file__).resolve()),'--worker',str(seed)],stdout=f,stderr=subprocess.STDOUT,cwd=ROOT,env={**__import__('os').environ,'OPENBLAS_NUM_THREADS':'1'},check=True)
   body_controls(seed);progress['completed'].append(seed);progress['wall_seconds']=time.perf_counter()-started;atomic_json(OUT/'progress.json',progress);print(f'{len(progress["completed"])}/20 matched seeds; {3*len(progress["completed"])}/60 full-brain trials',flush=True)
  progress['status']='complete';atomic_json(OUT/'progress.json',progress)
 except Exception as e:progress.update(status='interrupted',error=repr(e));atomic_json(OUT/'progress.json',progress);raise
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--worker',type=int);a=p.parse_args()
 if a.worker:worker(a.worker)
 else:run()
