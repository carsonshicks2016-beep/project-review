"""Stage7 full brain + physical body, causal command application and touch audit."""
import argparse,json,sys,time,subprocess,resource,pickle,copy,hashlib
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from flygarden.recording import atomic_json,space_check
from scripts.audit_brain_reference import sha
from flygarden.synchronized import SynchronizedBody,CausalCoupling
from flygarden.body_feedback import BodyFeedback
from flygarden.descending import DescendingDecoder
OUT=ROOT/'reports/brain-integration/stage7-20261006'
SEEDS=(7101,7102,7199);INTERVALS=(.1,.05,.025);CUES=('blank','loom_left','loom_right')
SOURCES=('scripts/diagnose_causal_timing.py','scripts/prepare_timing_feedback.py','flygarden/synchronized.py','flygarden/body_feedback.py','flygarden/descending.py','flygarden/brain.py','flygarden/body.py','flygarden/engine.py','data/annotations.tsv','vendor/fly-brain/data/2025_Completeness_783.csv','vendor/fly-brain/data/2025_Connectivity_783.parquet','reports/brain-integration/stage7-20261006/feedback-mapping.json','reports/brain-integration/stage6-20261006/eye-stimuli/manifest.json')
def mapping():
 d=json.loads((OUT/'feedback-mapping.json').read_text());p=d['mapping']
 # Visual mappings are pinned to the existing exact-root stage6 artifact.
 v=json.loads((ROOT/'reports/brain-integration/stage6-20261006/protocol.json').read_text())['mapping']
 for key in ('LPLC2_left','LPLC2_right','DNp01_left','DNp01_right'):p[key]=v[key]
 return p
def sources():return {s:sha(ROOT/s) for s in SOURCES}
def worker(folder,cue,seed,interval,resume=None):
 import brian2 as b
 from flygarden.brain import FullBrain
 start=time.perf_counter();brain=FullBrain(seed=seed,learning=False,record_spikes=True);pops=mapping();body=SynchronizedBody(seed=seed,blocks=[{'x':4,'y':0,'w':.8,'h':12,'z':2}] if cue.startswith('contact') else []);feedback=BodyFeedback();coupling=CausalCoupling(interval);decoder=DescendingDecoder()
 keys=('LPLC2_left','LPLC2_right','AN_multi_63_left','AN_multi_63_right');groups=[np.array([n['index'] for n in pops[k]],dtype=np.int32) for k in keys];targets=np.concatenate(groups);channels=np.concatenate([np.full(len(g),i,dtype=np.int32) for i,g in enumerate(groups)])
 assert all(str(brain.ids[n['index']])==n['root_id'] for k in keys for n in pops[k])
 gen=b.SpikeGeneratorGroup(len(targets),np.array([],dtype=np.int32),np.array([])*b.second,clock=brain.clock);ext=b.Synapses(gen,brain.neurons,on_pre='v_post += 68.75*mV',clock=brain.clock);ext.connect(i=np.arange(len(targets)),j=targets);im=b.SpikeMonitor(gen,record=False);brain.network.add(gen,ext,im);brain.inputs.active=False;brain.input.active=False;brain.neurons.rfc=2.2*b.ms
 selected=np.zeros(4,dtype=bool)
 if cue=='loom_left':selected[0]=True
 if cue=='loom_right':selected[1]=True
 if cue=='contact_on':selected[2:]=True
 brain.neurons.rfc[targets[selected[channels]]]=0*b.ms
 eye_cue=cue if cue in CUES else 'blank';eye_path=ROOT/'reports/brain-integration/stage6-20261006/eye-stimuli'/f'{eye_cue}.json';eye_rows=json.loads(eye_path.read_text());eye_rates=np.array([[r['features'][s]['lplc2_hz'] for s in range(2)] for r in eye_rows]);rng=np.random.default_rng(seed);previous=np.zeros(brain.n,dtype=np.int64);prior_input=np.zeros(len(targets),dtype=np.int64)
 if cue.startswith('contact'):coupling.motor[:]=.65
 if resume:
  metadata=json.loads((resume/'manifest.json').read_text());assert metadata['sources']==sources() and metadata['cue']==cue and metadata['seed']==seed and metadata['interval']==interval
  assert all(sha(resume/name)==digest for name,digest in metadata['files'].items()), 'Checkpoint integrity failure'
  brain.load(resume/'brain.state')
  with (resume/'driver.pkl').open('rb') as f:state=pickle.load(f)
  body.restore(state['body']);coupling.restore(state['coupling']);feedback.restore(state['feedback']);rng.bit_generator.state=state['rng'];decoder.motor=state['decoder_motor'];previous=state['previous'];prior_input=state['prior_input']
 origin=body.sim.mj_data.time-body.steps*body.dt;manifest={'status':'running','cue':cue,'seed':seed,'interval':interval,'neurons':brain.n,'connections':brain.edges,'learning':False,'sources':sources(),'input_indices':targets.tolist(),'channels':channels.tolist(),'input_root_ids':[str(brain.ids[i]) for i in targets],'mapping':pops,'chunks':[],'frames_file':'frames.json','resumed_from':str(resume) if resume else None,'start_time':coupling.time,'eye_stream_sha256':sha(eye_path),'weights_before':hashlib.sha256(np.asarray(brain.synapses.w[:]/b.mV).tobytes()).hexdigest(),'motor_identity':'Fixed .65/.65 body calibration; neural candidate logged only' if cue.startswith('contact') else 'Fixed candidate DNa02/DNp09 decoder; no tonic drive or escape override','vision_identity':'Recorded actual eye-image feature stream; stationary source pose, not visually closed-loop navigation','contact_identity':'Live physical foreleg forces to pooled engineered TLA rates' if cue=='contact_on' else 'Neural contact input disabled; gait feedback still active'}
 folder.mkdir(parents=True,exist_ok=False);atomic_json(folder/'manifest.json',manifest);rows=[];frames=[];initial=body.observation();steps_total=round(3/interval);first=round(coupling.time/interval)
 try:
  for tick in range(first,steps_total):
   t=coupling.time;assert abs(float(brain.network.t/b.second)-t)<1e-9 and abs(body.steps*body.dt-t)<1e-9 and abs(body.sim.mj_data.time-origin-t)<1e-8
   obs=body.observation();fb=feedback.advance(obs,t,enabled=cue=='contact_on');micro=np.arange(coupling.ticks,coupling.ticks+coupling.ticks_per_window,dtype=np.int64);frame_indices=micro*3//1000;rate_matrix=np.zeros((len(micro),4));rate_matrix[:,:2]=eye_rates[frame_indices];rate_matrix[:,2:]=fb['tla_hz'];assert np.all(np.array([eye_rows[i]['time'] for i in frame_indices])<=micro*.0001+1e-9)
   samples=rng.random((len(micro),len(targets)))<rate_matrix[:,channels]*.0001;local_ticks,ii=np.nonzero(samples);event_times=micro[local_ticks]*.0001;gen.set_spikes(ii,event_times*b.second,sorted=True)
   block={}
   def advance_brain(dt):
    nonlocal previous,prior_input
    brain.network.run(dt*b.second,namespace={});counts=np.asarray(brain.monitor.count[:],dtype=np.int64);block['counts']=counts-previous;previous=counts.copy();ic=np.asarray(im.count[:],dtype=np.int64);block['input_counts']=ic-prior_input;prior_input=ic.copy();block['spike_i']=np.asarray(brain.monitor.i[:],dtype=np.int32).copy();block['spike_t']=np.asarray(brain.monitor.t[:]/b.second).copy();assert len(block['spike_i'])==block['counts'].sum();block['population_hz']={k:float(block['counts'][[n['index'] for n in v]].mean()/dt) if v else None for k,v in pops.items()};block['candidate_motor']=decoder.advance(dt,block['population_hz']);return [.65,.65] if cue.startswith('contact') else block['candidate_motor']
   transition=coupling.advance(advance_brain,lambda dt,m:body.advance(dt,m,capture=lambda stamp,pose:frames.append({'time':stamp,**pose})))
   assert abs(float(brain.network.t/b.second)-coupling.time)<1e-9 and abs(body.steps*body.dt-coupling.time)<1e-9
   assert np.isfinite(brain.neurons.v[:]/b.mV).all() and np.isfinite(brain.neurons.g[:]/b.mV).all()
   row={'time':coupling.time,'observation_time':t,'image_frame_time_min':float(eye_rows[int(frame_indices[0])]['time']),'image_frame_time_max':float(eye_rows[int(frame_indices[-1])]['time']),'motor_start':transition['start'],'motor_end':transition['end'],'applied_motor':transition['applied_motor'],'candidate_motor':block['candidate_motor'].tolist(),'next_motor':transition['next_motor'],'next_available_at':transition['next_available_at'],'population_hz':block['population_hz'],'feedback':fb,'body':transition['body'],'input_hz_mean':rate_matrix.mean(axis=0).tolist(),'total_spikes':len(block['spike_i'])};rows.append(row)
   file=folder/f'window-{tick:04d}.npz';space_check(folder,block['counts'].nbytes+block['spike_i'].nbytes+block['spike_t'].nbytes+1024**2)
   with file.with_suffix('.tmp').open('wb') as f:np.savez_compressed(f,counts=block['counts'],input_counts=block['input_counts'],spike_i=block['spike_i'],spike_t=block['spike_t'],external_i=ii.astype(np.int32),external_t=event_times,physics=body.snapshot()['physics'],motor=np.array(transition['applied_motor']),next_motor=np.array(transition['next_motor']))
   file.with_suffix('.tmp').replace(file);brain.clear_recorded_spikes();manifest['chunks'].append({'file':file.name,'sha256':sha(file),'time':coupling.time,'spikes':len(block['spike_i'])});atomic_json(folder/'manifest.json',manifest);atomic_json(folder/'bins.json',rows);atomic_json(folder/'frames.json',frames)
   if not resume and cue=='loom_left' and seed==7199 and interval==.025 and abs(coupling.time-1.5)<1e-9:
    checkpoint=folder/'checkpoint-1.5';checkpoint.mkdir();space_check(checkpoint,600*1024**2);brain.save(checkpoint/'brain.state')
    with (checkpoint/'driver.pkl').open('wb') as f:pickle.dump({'body':body.snapshot(),'coupling':coupling.snapshot(),'feedback':feedback.snapshot(),'rng':copy.deepcopy(rng.bit_generator.state),'decoder_motor':decoder.motor.copy(),'previous':previous.copy(),'prior_input':prior_input.copy()},f)
    atomic_json(checkpoint/'manifest.json',{'sources':sources(),'cue':cue,'seed':seed,'interval':interval,'time':coupling.time,'scope':'Experimental pipeline checkpoint; resume with identical stage7 worker, not the production application.','files':{p.name:sha(p) for p in checkpoint.iterdir() if p.is_file()}})
  manifest['weights_after']=hashlib.sha256(np.asarray(brain.synapses.w[:]/b.mV).tobytes()).hexdigest();assert manifest['weights_after']==manifest['weights_before'] and brain.plasticity.updates==0
  headings=np.unwrap([initial['heading']]+[r['body']['heading'] for r in rows]);manifest.update(status='complete',wall_seconds=time.perf_counter()-start,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,heading_change_radians=float(headings[-1]-headings[0]),flipped_samples=sum(r['body']['flipped'] for r in rows),body_frames=len(frames),end_time=coupling.time);atomic_json(folder/'manifest.json',manifest)
 finally:body.close()
def cases():
 return [(cue,seed,dt) for dt in INTERVALS for cue in CUES for seed in SEEDS]+[(cue,seed,.025) for cue in ('contact_off','contact_on') for seed in SEEDS]
def label_case(cue,seed,dt):return f'{cue}-{seed}-{round(dt*1000)}ms'
def run():
 protocol=OUT/'protocol.json'
 if not protocol.exists():atomic_json(protocol,{'schema_version':1,'seeds':SEEDS,'held_out_seed':7199,'intervals':INTERVALS,'selected_interval_seconds':.025,'selection':'Prechosen smallest supported interval, not fit to held-out steering performance.','sources':sources(),'cases':cases(),'gates':{'clock':'Every neural/body/physics boundary shares integer model ticks; commands apply only at or after their computation end.','repeatability':'Full raw neural event streams invariant across intervals for fixed recorded eye input and seed; independent fresh-process checkpoint continuation must match raw events, commands and complete physical integration state.','recording':'Exactly90 physical30Hz samples per3s at every interval; observation/capture cannot perturb physical dynamics.','feedback':'Exact-root anterior-touch candidates only; force-to-rate remains engineered. Live contact-on/off uses identical external motor calibration to isolate feedback; velocity/joint-to-neuron mappings remain blocked.','behavior':'No navigation/escape/learning claim or automatic production promotion.'},'scope':'Continuous full brain plus physical body. Vision is prerecorded; contact feedback is live but contact trials use fixed motor calibration. No neural restart per window, no skipped simulated time, no render clock input.'})
 p=json.loads(protocol.read_text());assert p['sources']==sources();progress={'status':'running','planned':33,'completed':[]};started=time.perf_counter()
 try:
  for cue,seed,dt in cases():
   name=label_case(cue,seed,dt);folder=OUT/'trials'/name
   if (folder/'manifest.json').exists() and json.loads((folder/'manifest.json').read_text())['status']=='complete':progress['completed'].append(name);continue
   if folder.exists():raise RuntimeError('Preserve incomplete attempt before retry '+str(folder))
   space_check(OUT,650*1024**2);progress['current']={'cue':cue,'seed':seed,'interval':dt};atomic_json(OUT/'progress.json',progress);log=OUT/'logs'/f'{name}.log';log.parent.mkdir(exist_ok=True)
   with log.open('w') as f:subprocess.run([sys.executable,str(Path(__file__).resolve()),str(folder),'--cue',cue,'--seed',str(seed),'--interval',str(dt)],stdout=f,stderr=subprocess.STDOUT,cwd=ROOT,check=True)
   progress['completed'].append(name);progress['wall_seconds']=time.perf_counter()-started;atomic_json(OUT/'progress.json',progress);print(f"{len(progress['completed'])}/33 {name}",flush=True)
  progress['status']='complete';atomic_json(OUT/'progress.json',progress)
 except Exception as e:progress.update(status='interrupted',error=repr(e));atomic_json(OUT/'progress.json',progress);raise
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('folder',type=Path,nargs='?');p.add_argument('--cue',choices=(*CUES,'contact_on','contact_off'));p.add_argument('--seed',type=int);p.add_argument('--interval',type=float);p.add_argument('--resume',type=Path);a=p.parse_args()
 if a.cue:worker(a.folder,a.cue,a.seed,a.interval,a.resume)
 else:run()
