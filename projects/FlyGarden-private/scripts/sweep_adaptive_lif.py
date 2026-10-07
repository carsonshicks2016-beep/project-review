"""Frozen nine-point adaptation sweep. Two calibration seeds; fresh held-out seeds."""
import argparse,fcntl,hashlib,json,os,resource,shutil,subprocess,sys,time
from pathlib import Path
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from flygarden.continuous_candidate import ContinuousCandidate,file_sha
from flygarden.adaptive_candidate import AdaptiveCandidate
from flygarden.recording import atomic_json,space_check
OUT=ROOT/'reports/brain-integration/recovery/adaptive-parameter-sweep-v1'
MAP=ROOT/'reports/brain-integration/recovery/local-inhibition-full-network-v1/protocol.json'
DOMAINS=['projection','sensory_projection','olfactory_hypothesis','global'];MODELS=['original']+DOMAINS
SEED=12001;DT=.025;STEPS=240
SOURCES=['scripts/screen_adaptive_lif.py','flygarden/adaptive_neurons.py','flygarden/adaptive_brain.py','flygarden/adaptive_candidate.py','flygarden/brain.py','flygarden/continuous_candidate.py','flygarden/candidate_inputs.py','flygarden/descending.py','data/annotations.tsv','vendor/fly-brain/data/2025_Completeness_783.csv','vendor/fly-brain/data/2025_Connectivity_783.parquet',str(MAP.relative_to(ROOT)),'reports/brain-integration/recovery/adaptive-cell-reference-v1/results.json']

def prepare():
 p=json.loads((OUT/'protocol.json').read_text())
 assert all(file_sha(ROOT/n)==h for n,h in p['sources'].items()),'Frozen sources changed'
 return p

def construct(p,model,seed):
 return ContinuousCandidate(p['mapping'],seed) if model=='original' else AdaptiveCandidate(p['mapping'],p['domains'][model],seed,tau_s=p['parameters'][model]['tau_s'],step_mv=p['parameters'][model]['step_mv'])

def advance(c,case,tick):
 odor=np.zeros((2,2))
 if 12<=tick<32 or 132<=tick<152:
  if case=='a_left':odor[0,0]=1
  elif case=='a_right':odor[1,0]=1
  elif case=='a_both':odor[:,0]=1
  elif case in ['equal','g60','g40']:odor[:,0]={'equal':[.5,.5],'g60':[.6,.4],'g40':[.4,.6]}[case]
 return c.advance(DT,odor,[0,0])

def arrays(c,p,motor):
 import brian2 as b
 d={'counts':c.last['spike_counts'],'spike_i':c.last_spikes[0],'spike_t':c.last_spikes[1],'external_i':c.last['external_indices'],'external_t':c.last['external_times'],'motor':motor,'v_mV':np.asarray(c.brain.neurons.v[p['selected']]/b.mV).copy(),'g_mV':np.asarray(c.brain.neurons.g[p['selected']]/b.mV).copy()}
 if hasattr(c.brain.neurons,'adapt'):d['adapt_mV']=np.asarray(c.brain.neurons.adapt[p['selected']]/b.mV).copy()
 return d

def fullstate(c):
 import brian2 as b
 keys=['v','g']+(['adapt'] if hasattr(c.brain.neurons,'adapt') else [])
 return {key:np.asarray(getattr(c.brain.neurons,key)[:]/b.mV).copy() for key in keys}

def worker(model,case,seed,resume=False):
 p=prepare();folder=OUT/'trials'/f'{model}-{case}-{seed}';start=time.monotonic();c=construct(p,model,seed)
 if resume:
  c.load(folder/'checkpoint')
  for tick in range(151,156):
   d=arrays(c,p,advance(c,case,tick))
   with np.load(folder/f'window-{tick:04d}.npz') as z:assert all(np.array_equal(z[k],v) for k,v in d.items()),('Continuation',model,tick)
  with np.load(folder/'continuation-state.npz') as z:errors={k:float(np.max(abs(v-z[k]))) for k,v in fullstate(c).items()}
  assert max(errors.values())<=1e-10;atomic_json(folder/'continuation-result.json',{'status':'passed','fresh_process':True,'errors_mV':errors});return
 folder.mkdir(parents=True,exist_ok=False);weights_hash=hashlib.sha256(np.asarray(c.brain.synapses.w[:]).tobytes()).hexdigest()
 m={'status':'running','model':model,'case':case,'seed':seed,'controller':c.manifest(),'sources':p['sources'],'initial_weight_hash':weights_hash,'chunks':[]};rows=[];atomic_json(folder/'manifest.json',m)
 try:
  for tick in range(STEPS):
   motor=advance(c,case,tick);d=arrays(c,p,motor);assert all(np.isfinite(v).all() for v in d.values());assert np.array_equal(np.bincount(d['spike_i'],minlength=c.brain.n),d['counts']);space_check(folder,8*1024**2);file=folder/f'window-{tick:04d}.npz'
   with file.with_suffix('.tmp').open('wb') as f:np.savez_compressed(f,**d)
   file.with_suffix('.tmp').replace(file);m['chunks'].append({'file':file.name,'sha256':file_sha(file),'end':c.time});rows.append({'end':c.time,'population_hz':c.last['population_hz'],'motor':motor.tolist()});atomic_json(folder/'rows.json',rows);atomic_json(folder/'manifest.json',m)
   atomic_json(OUT/'current-worker.json',{'trial':folder.name,'completed_windows':tick+1,'planned_windows':240,'simulated_seconds':c.time,'wall_seconds':time.monotonic()-start})
   if case=='a_left' and seed in [12001,12002,12111,12112] and tick==150:c.save(folder/'checkpoint')
   if case=='a_left' and seed in [12001,12002,12111,12112] and tick==155:np.savez_compressed(folder/'continuation-state.npz',**fullstate(c))
  final=hashlib.sha256(np.asarray(c.brain.synapses.w[:]).tobytes()).hexdigest();assert final==weights_hash and c.brain.plasticity.updates==0
  assert all(file_sha(ROOT/n)==h for n,h in p['sources'].items());m.update(status='complete',final_weight_hash=final,wall_seconds=time.monotonic()-start,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss);atomic_json(folder/'manifest.json',m);print(folder.name,'complete',round(m['wall_seconds'],1),'s',flush=True)
 except BaseException as e:m.update(status='interrupted',error=repr(e));atomic_json(folder/'manifest.json',m);raise

def evaluate(p,models,seeds,cues):
 traces={};external={};audited=0
 for model in models:
  for seed in seeds:
   for cue in cues:
    folder=OUT/'trials'/f'{model}-{cue}-{seed}';m=json.loads((folder/'manifest.json').read_text());assert m['status']=='complete';rows=json.loads((folder/'rows.json').read_text());pop={k:np.array([r['population_hz'][k] for r in rows]) for k in p['mapping']};local=[]
    for ch in m['chunks']:
     file=folder/ch['file'];assert file_sha(file)==ch['sha256']
     with np.load(file) as z:
      assert np.array_equal(np.bincount(z['spike_i'],minlength=138639),z['counts']);local.append(z['counts'][[x['index'] for x in p['selected_local_targets']]]/DT);key=(seed,cue,ch['file']);digest=hashlib.sha256(z['external_i'].tobytes()+z['external_t'].tobytes()).hexdigest()
      if key in external:assert external[key]==digest
      else:external[key]=digest
     audited+=1
    pop.update(pn=(pop['DM1_lPN_left']+pop['DM1_lPN_right'])/2,pn_contrast=pop['DM1_lPN_left']-pop['DM1_lPN_right'],signed_dna=pop['DNa02_left']-pop['DNa02_right'],local=np.array(local));traces[(model,seed,cue)]=pop
 recovery=[];contrast=[];bursts=[];support=[];gradients=[]
 for model in models:
  for seed in seeds:
   none=traces[(model,seed,'none')]
   support.append({'model':model,'seed':seed,'passed':all(none[k].mean()>=.8*traces[('original',seed,'none')][k].mean() for k in ['DNp09_left','DNp09_right'])})
   for cue in [c for c in cues if c!='none']:
    t=traces[(model,seed,cue)]
    for pulse,(pl,ph),(rl,rh) in zip([1,2],p['pulse_windows'],p['recovery_windows']):
     pn=[float(t[k][rl:rh].mean()-none[k][rl:rh].mean()) for k in ['DM1_lPN_left','DM1_lPN_right']];local=t['local'][rl:rh].mean(axis=0)-none['local'][rl:rh].mean(axis=0);dna=float(t['signed_dna'][rl:rh].mean()-none['signed_dna'][rl:rh].mean());base=none['pn'][pl:ph].mean() if pulse==1 else t['pn'][100:120].mean();gain=float(t['pn'][pl:ph].mean()-base)
     recovery.append({'model':model,'seed':seed,'cue':cue,'pulse':pulse,'gain_hz':gain,'pn_recovery_difference_hz':pn,'local_recovery_difference_hz':local.tolist(),'signed_dna_recovery_difference_hz':dna,'passed':bool(gain>=10 and max(abs(x) for x in pn)<=5 and np.max(abs(local))<=5 and abs(dna)<=5)})
    for lo,hi in p['burst_windows']:
     excess=t['pn']-none['pn'];maximum=max(float(excess[i:i+4].mean()) for i in range(lo,hi-3));bursts.append({'model':model,'seed':seed,'cue':cue,'window':[lo,hi],'max_100ms_PN_excess_hz':maximum,'passed':maximum<=20})
   for pulse,(lo,hi) in zip([1,2],p['pulse_windows']):
    left=float(traces[(model,seed,'a_left')]['pn_contrast'][lo:hi].mean());right=float(traces[(model,seed,'a_right')]['pn_contrast'][lo:hi].mean());neutral=float(none['pn_contrast'][lo:hi].mean());contrast.append({'model':model,'seed':seed,'pulse':pulse,'left':left,'right':right,'none':neutral,'passed':left-right>=10 and left-neutral>0 and right-neutral<0})
    if 'equal' in cues:
     equal=float(traces[(model,seed,'equal')]['pn_contrast'][lo:hi].mean());g60=float(traces[(model,seed,'g60')]['pn_contrast'][lo:hi].mean());g40=float(traces[(model,seed,'g40')]['pn_contrast'][lo:hi].mean());gradients.append({'model':model,'seed':seed,'pulse':pulse,'equal':equal,'g60':g60,'g40':g40,'passed':g60-g40>=5 and g60-equal>0 and g40-equal<0})
 eligible={model:all(x['passed'] for group in [recovery,contrast,bursts,support,gradients] for x in group if x['model']==model) for model in models}
 return {'status':'complete','chunks_audited':audited,'paired_inputs_exact':True,'recovery':recovery,'contrast':contrast,'bursts':bursts,'support':support,'gradient':gradients,'eligible':eligible,'promoted':False}

def run():
 with (ROOT/'.runtime/experiment.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);p=prepare();done=[]
  def execute(model,cue,seed,phase):
   name=f'{model}-{cue}-{seed}';folder=OUT/'trials'/name
   atomic_json(OUT/'progress.json',{'status':'running','completed':done,'current':name,'phase':phase,'pid':os.getpid(),'planned_calibration_trials':153})
   if (folder/'manifest.json').exists():assert json.loads((folder/'manifest.json').read_text())['status']=='complete','Interrupted trial requires explicit recovery'
   else:
    space_check(OUT,1200*1024**2)
    subprocess.run([sys.executable,__file__,'--model',model,'--cue',cue,'--seed',str(seed)],cwd=ROOT,check=True)
   if cue=='a_left' and not (folder/'continuation-result.json').exists():
    subprocess.run([sys.executable,__file__,'--model',model,'--cue',cue,'--seed',str(seed),'--resume'],cwd=ROOT,check=True)
   done.append(name)
  try:
   execute('original','a_left',12001,'regression');execute('zero','a_left',12001,'regression')
   original=OUT/'trials'/'original-a_left-12001';zero=OUT/'trials'/'zero-a_left-12001'
   for ch in json.loads((original/'manifest.json').read_text())['chunks']:
    with np.load(original/ch['file']) as a,np.load(zero/ch['file']) as z:
     for k in ['counts','spike_i','spike_t','external_i','external_t','motor','v_mV','g_mV']:assert np.array_equal(a[k],z[k]),('Zero parity',k)
     assert not z['adapt_mV'].any()
   with np.load(original/'continuation-state.npz') as a,np.load(zero/'continuation-state.npz') as z:
    assert np.array_equal(a['v'],z['v']) and np.array_equal(a['g'],z['g']) and not z['adapt'].any()
   atomic_json(OUT/'zero-regression.json',{'status':'passed','events_and_whole_v_g_exact':True})
   for seed in p['calibration_seeds']:
    for model in p['models']:
     for cue in ['none','a_left','a_right','a_both']:
      if model=='original' and cue=='a_left' and seed==12001:continue
      execute(model,cue,seed,'calibration')
   cal=evaluate(p,p['models'],p['calibration_seeds'],['none','a_left','a_right','a_both'])
   atomic_json(OUT/'calibration-results.json',cal)
   selected=next((m for m in p['selection_order'] if cal['eligible'][m]),None)
   atomic_json(OUT/'selection.json',{'selected':selected,'parameters':p['parameters'].get(selected),'basis':'Every gate on both calibration seeds; frozen domain then step then tau order','calibration_only':True})
   if selected:
    for seed in p['held_out_seeds']:
     for model in ['original',selected]:
      for cue in ['none','a_left','a_right','a_both','equal','g60','g40']:execute(model,cue,seed,'heldout')
    result=evaluate(p,['original',selected],p['held_out_seeds'],['none','a_left','a_right','a_both','equal','g60','g40'])
    atomic_json(OUT/'heldout-results.json',result)
   atomic_json(OUT/'progress.json',{'status':'complete','completed':done,'selected':selected,'heldout_run':bool(selected),'application_changed':False,'promoted':False})
  except BaseException as e:
   atomic_json(OUT/'progress.json',{'status':'interrupted','completed':done,'error':repr(e)});raise
if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('--prepare',action='store_true');a.add_argument('--model');a.add_argument('--cue');a.add_argument('--seed',type=int);a.add_argument('--resume',action='store_true');a=a.parse_args()
 if a.prepare:prepare()
 elif a.model:worker(a.model,a.cue,a.seed,a.resume)
 else:run()
