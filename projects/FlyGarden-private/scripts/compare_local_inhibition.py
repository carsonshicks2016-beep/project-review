"""Frozen one-worker full-network comparison; no live-controller promotion."""
import argparse,fcntl,hashlib,json,os,resource,shutil,subprocess,sys,time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from flygarden.continuous_candidate import ContinuousCandidate,file_sha
from flygarden.recording import atomic_json,space_check
OUT=ROOT/'reports/brain-integration/recovery/local-inhibition-full-network-v1'
PRIOR=ROOT/'reports/brain-integration/recovery/patchy-sign-factorial-v1/protocol.json'
ROUTES=ROOT/'reports/brain-integration/recovery/local-inhibition-adapter-v1/routes.json'
DT=.025;STEPS=240;SEEDS=[11801,11802]
CASES=[{'model':m,'cue':c} for m in ['original','local'] for c in ['none','a_left','a_right','a_both']]
SOURCES=['scripts/compare_local_inhibition.py','flygarden/local_inhibition_candidate.py','flygarden/local_inhibition_bridge.py','flygarden/local_inhibition_adapter.py','flygarden/brain.py','flygarden/continuous_candidate.py','flygarden/candidate_inputs.py','flygarden/descending.py','data/annotations.tsv','vendor/fly-brain/data/2025_Completeness_783.csv','vendor/fly-brain/data/2025_Connectivity_783.parquet',str(PRIOR.relative_to(ROOT)),str(ROUTES.relative_to(ROOT))]

def prepare():
 OUT.mkdir(exist_ok=True);hashes={n:file_sha(ROOT/n) for n in SOURCES}
 if (OUT/'protocol.json').exists():
  p=json.loads((OUT/'protocol.json').read_text());assert p['sources']==hashes,'Frozen sources changed';return p
 prior=json.loads(PRIOR.read_text());p={'version':1,'sources':hashes,'mapping':prior['mapping'],'selected':prior['selected'],'selected_local_targets':prior['selected_local_targets'],'pulse_windows':prior['pulse_windows'],'recovery_windows':prior['recovery_windows'],'cases':CASES,'seeds':SEEDS,'dt':DT,'steps':STEPS,
 'fixed_candidate':{'tau_s':.015,'normalization_hz':100,'input_filter_tau_s':.005,'delay_s':.0018,'clock_s':.0001,'continuous_injection':'Negative average voltage-equivalent g increment; no refractory masking beyond original equations','coupling':'Anatomical local-to-local paths receive 1.8-ms delay; no inferred cable coupling'},
 'gates':'Unchanged diagnostic criteria: pulse mean DM1 PN increase>=10Hz; each DM1 PN, selected local neuron and signed DNa02 recovery difference<=5Hz; left-minus-right PN contrast>=10Hz with opposite deviations from none, each pulse/seed.',
 'input_schedule':'Same owned seeded Bernoulli inputs/profile as prior original baseline: two 50Hz odorA pulses .3-.8s and3.3-3.8s,65Hz walking support.',
 'continuation':'original and local a_left seed11801 checkpoint at3.775s; fresh process five windows exact counts/events/inputs/motor; full v/g and local state<=1e-10',
 'scope':'Two diagnostic seeds, not learning/navigation or measured physiology. Filter, normalized activity, limited coupling, missing MIP/receptor kinetics and hybrid residual are engineered assumptions. No tuning against these outcomes and no application promotion.'}
 atomic_json(OUT/'protocol.json',p)
 for n in SOURCES:
  if n.startswith(('data/','vendor/','reports/')):continue
  dest=OUT/'source'/n;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/n,dest)
 return p

def construct(p,case,seed):
 if case['model']=='original':return ContinuousCandidate(p['mapping'],seed)
 from flygarden.local_inhibition_candidate import LocalInhibitionCandidate
 return LocalInhibitionCandidate(p['mapping'],json.loads(ROUTES.read_text()),seed)

def advance(c,case,tick):
 odor=np.zeros((2,2))
 if case['cue']!='none' and (12<=tick<32 or 132<=tick<152):
  if case['cue'] in ('a_left','a_both'):odor[0,0]=1
  if case['cue'] in ('a_right','a_both'):odor[1,0]=1
 return c.advance(DT,odor,[0,0])

def arrays(c,p,command):
 import brian2 as b
 out={'counts':c.last['spike_counts'],'spike_i':c.last_spikes[0],'spike_t':c.last_spikes[1],'external_i':c.last['external_indices'],'external_t':c.last['external_times'],'motor':command,'v_mV':np.asarray(c.brain.neurons.v[p['selected']]/b.mV).copy(),'g_mV':np.asarray(c.brain.neurons.g[p['selected']]/b.mV).copy()}
 if hasattr(c,'bridge'):out.update(local_activity=c.bridge.adapter.activity.copy(),local_filtered_drive=np.asarray(c.bridge.local.r[:]).copy(),local_delay_buffer=c.bridge.buffer.copy())
 return out

def fullstate(c):
 import brian2 as b
 return {k:np.asarray(getattr(c.brain.neurons,k)[:]/b.mV).copy() for k in ('v','g')}

def worker(i,seed,off=False,resume=False):
 p=prepare();case=CASES[i];name=f"{case['model']}-{case['cue']}-{seed}"+('-observer-off' if off else '');folder=OUT/'trials'/name;start=time.monotonic();c=construct(p,case,seed)
 if resume:
  c.load(folder/'checkpoint')
  for tick in range(151,156):
   data=arrays(c,p,advance(c,case,tick))
   with np.load(folder/f'window-{tick:04d}.npz') as expected:
    assert all(np.array_equal(expected[k],v) for k,v in data.items()),('Continuation',tick)
  with np.load(folder/'continuation-state.npz') as expected:errors={k:float(np.max(np.abs(expected[k]-v))) for k,v in fullstate(c).items()}
  assert all(v<=1e-10 for v in errors.values())
  atomic_json(folder/'continuation-result.json',{'status':'passed','fresh_process':True,'windows':5,'state_errors_mV':errors});return
 folder.mkdir(parents=True,exist_ok=False);weight_hash=hashlib.sha256(np.asarray(c.brain.synapses.w[:]).tobytes()).hexdigest()
 m={'status':'running','case':case,'seed':seed,'controller':c.manifest(),'sources':p['sources'],'chunks':[],'initial_weight_hash':weight_hash,'observer_off':off};atomic_json(folder/'manifest.json',m);rows=[]
 try:
  for tick in range(STEPS):
   command=advance(c,case,tick)
   if hasattr(c,'bridge') and not off:
    before=c.bridge.checkpoint();view=c.bridge.adapter.release_hz();assert np.isfinite(view).all();assert before==c.bridge.checkpoint()
   data=arrays(c,p,command);assert np.array_equal(np.bincount(data['spike_i'],minlength=c.brain.n),data['counts']);assert all(np.isfinite(v).all() for v in data.values())
   space_check(folder,8*1024**2);file=folder/f'window-{tick:04d}.npz'
   with file.with_suffix('.tmp').open('wb') as f:np.savez_compressed(f,**data)
   file.with_suffix('.tmp').replace(file);m['chunks'].append({'file':file.name,'sha256':file_sha(file),'end':c.time});rows.append({'end':c.time,'population_hz':c.last['population_hz'],'motor':command.tolist()});atomic_json(folder/'rows.json',rows);atomic_json(folder/'manifest.json',m)
   atomic_json(OUT/'current-worker.json',{'trial':name,'completed_windows':tick+1,'planned_windows':STEPS,'simulated_seconds':c.time,'wall_seconds':time.monotonic()-start})
   checkpoint=case['cue']=='a_left' and seed==SEEDS[0] and not off
   if checkpoint and tick==150:c.save(folder/'checkpoint')
   if checkpoint and tick==155:np.savez_compressed(folder/'continuation-state.npz',**fullstate(c))
  final=hashlib.sha256(np.asarray(c.brain.synapses.w[:]).tobytes()).hexdigest();assert final==weight_hash and c.brain.plasticity.updates==0
  assert all(file_sha(ROOT/n)==h for n,h in p['sources'].items())
  m.update(status='complete',wall_seconds=time.monotonic()-start,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,final_weight_hash=final);atomic_json(folder/'manifest.json',m);print(name,'complete',round(m['wall_seconds'],1),'s',flush=True)
 except BaseException as e:m.update(status='interrupted',error=repr(e));atomic_json(folder/'manifest.json',m);raise

def analyze(p):
 traces={};audited=0;external={}
 for case in CASES:
  for seed in SEEDS:
   name=f"{case['model']}-{case['cue']}-{seed}";folder=OUT/'trials'/name;m=json.loads((folder/'manifest.json').read_text());assert m['status']=='complete'
   rows=json.loads((folder/'rows.json').read_text());pop={k:np.array([r['population_hz'][k] for r in rows]) for k in p['mapping']}
   for chunk in m['chunks']:
    file=folder/chunk['file'];assert file_sha(file)==chunk['sha256']
    with np.load(file) as z:
     assert np.array_equal(np.bincount(z['spike_i'],minlength=138639),z['counts']);audited+=1
     digest=hashlib.sha256(z['external_i'].tobytes()+z['external_t'].tobytes()).hexdigest();key=(case['cue'],seed,chunk['file'])
     if key in external:assert external[key]==digest
     else:external[key]=digest
   local=np.array([[np.load(folder/ch['file'])['counts'][x['index']]/DT for x in p['selected_local_targets']] for ch in m['chunks']])
   pop['individual_local']=local;pop['pn']=(pop['DM1_lPN_left']+pop['DM1_lPN_right'])/2;pop['pn_contrast']=pop['DM1_lPN_left']-pop['DM1_lPN_right'];pop['signed_dna']=pop['DNa02_left']-pop['DNa02_right'];traces[(case['model'],case['cue'],seed)]=pop
 recovery=[];contrast=[]
 for model in ['original','local']:
  for seed in SEEDS:
   none=traces[(model,'none',seed)]
   for cue in ['a_left','a_right','a_both']:
    t=traces[(model,cue,seed)]
    for pulse,(pl,ph),(rl,rh) in zip([1,2],p['pulse_windows'],p['recovery_windows']):
     pn=[float(t[k][rl:rh].mean()-none[k][rl:rh].mean()) for k in ['DM1_lPN_left','DM1_lPN_right']];local=t['individual_local'][rl:rh].mean(axis=0)-none['individual_local'][rl:rh].mean(axis=0);dna=float(t['signed_dna'][rl:rh].mean()-none['signed_dna'][rl:rh].mean());base=none['pn'][pl:ph].mean() if pulse==1 else t['pn'][100:120].mean();gain=float(t['pn'][pl:ph].mean()-base)
     recovery.append({'model':model,'cue':cue,'seed':seed,'pulse':pulse,'pn_difference_hz':pn,'local_difference_hz':local.tolist(),'signed_dna_difference_hz':dna,'response_gain_hz':gain,'passed':bool(max(abs(x) for x in pn)<=5 and np.all(abs(local)<=5) and abs(dna)<=5 and gain>=10)})
   for pulse,(lo,hi) in zip([1,2],p['pulse_windows']):
    left=float(traces[(model,'a_left',seed)]['pn_contrast'][lo:hi].mean());right=float(traces[(model,'a_right',seed)]['pn_contrast'][lo:hi].mean());neutral=float(none['pn_contrast'][lo:hi].mean());contrast.append({'model':model,'seed':seed,'pulse':pulse,'left':left,'right':right,'none':neutral,'passed':bool(left-right>=10 and left-neutral>0 and right-neutral<0)})
 off=OUT/'trials'/f'local-a_left-{SEEDS[0]}-observer-off';original=OUT/'trials'/f'local-a_left-{SEEDS[0]}'
 for ch in json.loads((off/'manifest.json').read_text())['chunks']:
  with np.load(off/ch['file']) as a,np.load(original/ch['file']) as z:assert all(np.array_equal(a[k],z[k]) for k in a.files)
 checks={}
 for model in ['original','local']:
  d=json.loads((OUT/'trials'/f'{model}-a_left-{SEEDS[0]}'/'continuation-result.json').read_text());assert d['status']=='passed';checks[model]=d
 result={'status':'complete','trials':17,'raw_chunks_audited':audited+240,'paired_external_events_exact':True,'spike_count_parity':True,'observer_parity_exact':True,'continuation_checks':checks,'recovery_checks':recovery,'contrast_checks':contrast,'response_recovery_passed':{m:all(x['passed'] for x in recovery if x['model']==m) for m in ['original','local']},'sensory_contrast_passed':{m:all(x['passed'] for x in contrast if x['model']==m) for m in ['original','local']},'promoted':False,'navigation_validated':False,'learning_validated':False}
 atomic_json(OUT/'results.json',result);print(json.dumps({k:result[k] for k in ['response_recovery_passed','sensory_contrast_passed']},indent=2))

def run():
 with (ROOT/'.runtime/experiment.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);p=prepare();done=[]
  # Interleave the two models by cue so failures/progress are comparable early.
  jobs=[(CASES.index({'model':m,'cue':cue}),seed,False) for seed in SEEDS for cue in ['none','a_left','a_right','a_both'] for m in ['original','local']]+[(5,SEEDS[0],True)]
  try:
   for i,seed,off in jobs:
    case=CASES[i];name=f"{case['model']}-{case['cue']}-{seed}"+('-observer-off' if off else '');folder=OUT/'trials'/name
    if (folder/'manifest.json').exists():assert json.loads((folder/'manifest.json').read_text())['status']=='complete','Interrupted trial preserved: requires explicit continuation decision'
    else:
     space_check(OUT,1200*1024**2);atomic_json(OUT/'progress.json',{'status':'running','completed':done,'current':name,'planned':17,'pid':os.getpid()})
     args=[sys.executable,__file__,'--case',str(i),'--seed',str(seed)]
     if off:args.append('--unobserved')
     subprocess.run(args,cwd=ROOT,check=True)
    if case['cue']=='a_left' and seed==SEEDS[0] and not off and not (folder/'continuation-result.json').exists():subprocess.run([sys.executable,__file__,'--case',str(i),'--seed',str(seed),'--resume'],cwd=ROOT,check=True)
    done.append(name)
   analyze(p);atomic_json(OUT/'progress.json',{'status':'complete','completed':done,'planned':17})
  except BaseException as e:atomic_json(OUT/'progress.json',{'status':'interrupted','completed':done,'error':repr(e),'planned':17});raise
if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('--prepare',action='store_true');a.add_argument('--case',type=int);a.add_argument('--seed',type=int);a.add_argument('--unobserved',action='store_true');a.add_argument('--resume',action='store_true');a=a.parse_args()
 if a.prepare:prepare()
 elif a.case is not None:worker(a.case,a.seed,a.unobserved,a.resume)
 else:run()
