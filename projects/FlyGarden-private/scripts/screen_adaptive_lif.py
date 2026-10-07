"""Frozen one-point domain screen. Separate calibration and held-out seeds."""
import argparse,fcntl,hashlib,json,os,resource,shutil,subprocess,sys,time
from pathlib import Path
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from flygarden.continuous_candidate import ContinuousCandidate,file_sha
from flygarden.adaptive_candidate import AdaptiveCandidate
from flygarden.recording import atomic_json,space_check
OUT=ROOT/'reports/brain-integration/recovery/adaptive-domain-screen-v1'
MAP=ROOT/'reports/brain-integration/recovery/local-inhibition-full-network-v1/protocol.json'
DOMAINS=['projection','sensory_projection','olfactory_hypothesis','global'];MODELS=['original']+DOMAINS
SEED=11901;DT=.025;STEPS=240
SOURCES=['scripts/screen_adaptive_lif.py','flygarden/adaptive_neurons.py','flygarden/adaptive_brain.py','flygarden/adaptive_candidate.py','flygarden/brain.py','flygarden/continuous_candidate.py','flygarden/candidate_inputs.py','flygarden/descending.py','data/annotations.tsv','vendor/fly-brain/data/2025_Completeness_783.csv','vendor/fly-brain/data/2025_Connectivity_783.parquet',str(MAP.relative_to(ROOT)),'reports/brain-integration/recovery/adaptive-cell-reference-v1/results.json']

def prepare():
 OUT.mkdir(exist_ok=True);hashes={name:file_sha(ROOT/name) for name in SOURCES}
 if (OUT/'protocol.json').exists():
  p=json.loads((OUT/'protocol.json').read_text());assert p['sources']==hashes,'Frozen sources changed';return p
 assert json.loads((ROOT/'reports/brain-integration/recovery/adaptive-cell-reference-v1/results.json').read_text())['passed']
 prior=json.loads(MAP.read_text());ids=pd.read_csv(ROOT/'vendor/fly-brain/data/2025_Completeness_783.csv',index_col=0).index.to_numpy(dtype=np.int64);ann=pd.read_csv(ROOT/'data/annotations.tsv',sep='\t',dtype={'root_id':str},low_memory=False).fillna('').set_index('root_id').reindex([str(x) for x in ids]).fillna('')
 pn=ann.cell_class.eq('ALPN').to_numpy();orn=ann.cell_type.str.startswith('ORN_').to_numpy();alln=ann.cell_class.eq('ALLN').to_numpy();patchy=ann.cell_type.isin(['lLN2P_a','lLN2P_b','lLN2P_c']).to_numpy();assert patchy.sum()==34
 masks={'projection':pn,'sensory_projection':pn|orn,'olfactory_hypothesis':pn|orn|(alln&~patchy),'global':np.ones(len(ids),bool),'zero':np.ones(len(ids),bool)}
 descriptions={'projection':'ALPN annotation class; type-level spiking hypothesis, not exact-cell physiology','sensory_projection':'ALPN plus explicit ORN_ types; no ALLN inclusion','olfactory_hypothesis':'ALPN, ORN_, and ALLN except34 lLN2P_a/b/c. Remaining local-cell spiking identity is an explicit unvalidated engineering assumption','global':'All modeled spiking units, including biologically nonspiking types; engineering benchmark only','zero':'All units, adaptation increment zero: numerical baseline regression'}
 domains={name:{'name':name,'indices':np.flatnonzero(mask).tolist(),'root_ids':[str(x) for x in ids[mask]],'description':descriptions[name]} for name,mask in masks.items()}
 p={'version':1,'sources':hashes,'mapping':prior['mapping'],'selected':prior['selected'],'selected_local_targets':prior['selected_local_targets'],'pulse_windows':[[12,32],[132,152]],'recovery_windows':[[100,120],[220,240]],'models':MODELS,'domains':domains,'calibration_seed':SEED,'held_out_seeds':[11911,11912],'tau_s':.150,'step_mv':1.5,
 'design':'Initial fixed single-parameter-point domain screen, not completion of a parameter sweep. No parameter tuning in this package. A wider sweep requires its own frozen protocol and separate acceptance seeds.',
 'original_inputs':'50Hz odorA .3-.8s and3.3-3.8s;65Hz walking support; same encoder and decoder',
 'gates':{'response_gain_hz':10,'individual_PN_and_local_recovery_difference_hz':5,'signed_DNa02_recovery_difference_hz':5,'multi_trial_contrast_hz':10,'burst_100ms_recovery_PN_difference_hz':20,'walking_support_retention_fraction':.8,'state_precision_mV':1e-10},
 'burst_windows':[[44,132],[164,240]],'calibration_selection':'Require every response/recovery/contrast/burst/support criterion in the one calibration seed. Select narrowest eligible domain by ordered projection,sensory_projection,olfactory_hypothesis,global. This is calibration only, never proof of success.',
 'heldout_rule':'Only a calibration-eligible domain advances, parameter values stay150ms/1.5mV. Compare chosen domain against original on11911/11912 with none,left,right,both,equal(.5/.5),g60(.6/.4),g40(.4/.6). Preserve existing gates; gradient contrast mirror difference>=5Hz and opposite deviations from equal in each pulse/seed.',
 'determinism':'All a_left arms fresh-process checkpoint continuation five windows; exact events/counts/inputs/motor, continuous state<=1e-10mV. Adaptation state explicitly included. Zero-adaptation whole-network regression must pass before adapting arms. Learning frozen.',
 'mechanism_isolation':'No local-inhibition bridge, sign changes, graph pruning, encoder tuning or decoder changes.',
 'promotion':'No automatic application promotion. Embodied/escape checks are later prerequisites.'}
 atomic_json(OUT/'protocol.json',p)
 atomic_json(OUT/'domain-inventory.json',{'counts':{name:len(d['indices']) for name,d in domains.items()},'descriptions':descriptions,'known_nonspiking_patchy_roots_excluded_from_olfactory_hypothesis':[str(x) for x in ids[patchy]]})
 for name in SOURCES:
  if name.startswith(('vendor/','data/','reports/')):continue
  dest=OUT/'source'/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/name,dest)
 return p

def construct(p,model,seed):
 return ContinuousCandidate(p['mapping'],seed) if model=='original' else AdaptiveCandidate(p['mapping'],p['domains'][model],seed,tau_s=p['tau_s'],step_mv=0 if model=='zero' else p['step_mv'])

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
   if case=='a_left' and seed in [SEED,11911] and tick==150:c.save(folder/'checkpoint')
   if case=='a_left' and seed in [SEED,11911] and tick==155:np.savez_compressed(folder/'continuation-state.npz',**fullstate(c))
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
  def execute(model,cue,seed):
   name=f'{model}-{cue}-{seed}';folder=OUT/'trials'/name;atomic_json(OUT/'progress.json',{'status':'running','completed':done,'current':name,'phase':'heldout' if seed!=SEED else 'calibration','pid':os.getpid()})
   if (folder/'manifest.json').exists():assert json.loads((folder/'manifest.json').read_text())['status']=='complete'
   else:
    space_check(OUT,1200*1024**2);subprocess.run([sys.executable,__file__,'--model',model,'--cue',cue,'--seed',str(seed)],cwd=ROOT,check=True)
   if cue=='a_left' and seed in [SEED,11911] and not (folder/'continuation-result.json').exists():subprocess.run([sys.executable,__file__,'--model',model,'--cue',cue,'--seed',str(seed),'--resume'],cwd=ROOT,check=True)
   done.append(name)
  try:
   execute('original','a_left',SEED);execute('zero','a_left',SEED)
   original=OUT/'trials'/f'original-a_left-{SEED}';zero=OUT/'trials'/f'zero-a_left-{SEED}';maxerr=0.
   for ch in json.loads((original/'manifest.json').read_text())['chunks']:
    with np.load(original/ch['file']) as a,np.load(zero/ch['file']) as z:
     for k in ['counts','spike_i','spike_t','external_i','external_t','motor']:assert np.array_equal(a[k],z[k]),('Zero adaptation parity',k)
     for k in ['v_mV','g_mV']:maxerr=max(maxerr,float(np.max(abs(a[k]-z[k]))))
     assert not z['adapt_mV'].any()
   assert maxerr<=1e-10;atomic_json(OUT/'zero-regression.json',{'status':'passed','events_exact':True,'selected_state_max_error_mV':maxerr})
   for cue in ['none','a_left','a_right','a_both']:
    for model in MODELS:
     if model=='original' and cue=='a_left':continue
     execute(model,cue,SEED)
   cal=evaluate(p,MODELS,[SEED],['none','a_left','a_right','a_both']);atomic_json(OUT/'calibration-results.json',cal)
   selected=next((m for m in DOMAINS if cal['eligible'][m]),None);atomic_json(OUT/'selection.json',{'selected':selected,'tau_s':.150,'step_mv':1.5,'basis':'Frozen domain rank and all gates; no parameter fitting','calibration_only':True})
   if selected:
    for seed in p['held_out_seeds']:
     for cue in ['none','a_left','a_right','a_both','equal','g60','g40']:
      for model in ['original',selected]:execute(model,cue,seed)
    result=evaluate(p,['original',selected],p['held_out_seeds'],['none','a_left','a_right','a_both','equal','g60','g40']);atomic_json(OUT/'heldout-results.json',result)
   atomic_json(OUT/'progress.json',{'status':'complete','completed':done,'selected':selected,'heldout_run':bool(selected),'application_changed':False})
  except BaseException as e:atomic_json(OUT/'progress.json',{'status':'interrupted','completed':done,'error':repr(e)});raise
if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('--prepare',action='store_true');a.add_argument('--model');a.add_argument('--cue');a.add_argument('--seed',type=int);a.add_argument('--resume',action='store_true');a=a.parse_args()
 if a.prepare:prepare()
 elif a.model:worker(a.model,a.cue,a.seed,a.resume)
 else:run()
