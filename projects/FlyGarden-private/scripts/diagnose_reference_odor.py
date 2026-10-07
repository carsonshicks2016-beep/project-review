"""Stage 5 preregistered full-network measured odor-rate diagnostic."""
import argparse,json,sys,time,subprocess,resource
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from flygarden.recording import atomic_json,space_check
from scripts.audit_brain_reference import sha
from scripts.diagnose_sensorimotor_pathway import mapping
from flygarden.odor_encoding import OdorReference
SEEDS=(5101,5102,5199)
CONDITIONS=('baseline','oil_left','oil_right','acetate_left','acetate_right','pentanoate_left','pentanoate_right','acetate_switch','quiet','legacy_left','legacy_right')
SOURCES=('scripts/diagnose_reference_odor.py','scripts/prepare_odor_reference.py','flygarden/odor_encoding.py','flygarden/brain.py','data/annotations.tsv','vendor/fly-brain/data/2025_Completeness_783.csv','vendor/fly-brain/data/2025_Connectivity_783.parquet','reports/brain-integration/stage5-20261006/odor-reference.json')
def rates_for(cue,t,encoder):
 if cue=='quiet':return np.zeros(len(encoder.targets))
 if cue.startswith('legacy'):
  return np.array([50. if .5<=t<1 and u['glomerulus']=='DM1' and n['side']==cue.split('_')[1] else 0. for u,n in encoder.entries])
 chemical=None;left=right=0.
 if cue not in ('baseline',):
  chemical={'oil':'oil','acetate':'ethyl acetate','pentanoate':'ethyl pentanoate'}[cue.split('_')[0]]
  if .5<=t<1:
   if cue.endswith(('left','switch')):left=1.
   else:right=1.
  if cue=='acetate_switch' and 1.5<=t<2:right=1.
 return encoder.rates(chemical,left,right)
def worker(folder,cue,seed):
 import brian2 as b
 from flygarden.brain import FullBrain
 start=time.perf_counter();brain=FullBrain(seed=seed,learning=False,record_spikes=True)
 _,pops,_=mapping();artifact=json.loads((ROOT/SOURCES[-1]).read_text());encoder=OdorReference(artifact,brain.ids);targets=encoder.targets
 generator=b.SpikeGeneratorGroup(len(targets),np.array([],dtype=np.int32),np.array([])*b.second,clock=brain.clock)
 external=b.Synapses(generator,brain.neurons,on_pre='v_post += 68.75*mV',clock=brain.clock);external.connect(i=np.arange(len(targets)),j=targets)
 im=b.SpikeMonitor(generator,record=False);brain.network.add(generator,external,im);brain.inputs.active=False;brain.input.active=False
 brain.neurons.rfc=2.2*b.ms
 selected=np.any(np.array([rates_for(cue,tick*.1+1e-7,encoder) for tick in range(30)])>0,axis=0);brain.neurons.rfc[targets[selected]]=0*b.ms
 rng=np.random.default_rng(seed);previous=np.zeros(brain.n,dtype=np.int64);before_input=np.zeros(len(targets),dtype=np.int64);rows=[];weights=brain.learned_state()['weights'].copy()
 folder.mkdir(parents=True,exist_ok=False);manifest={'schema_version':1,'status':'running','cue':cue,'seed':seed,'neurons':brain.n,'connections':brain.edges,'chunks':[],'learning':False,'sources':{name:sha(ROOT/name) for name in SOURCES},'input_root_ids':[str(brain.ids[i]) for i in targets],'input_indices':targets.tolist(),'zero_refractory_indices':targets[selected].tolist(),'clock_seconds':.0001,'bin_seconds':.1,'synapse_changes':0,'intrinsic_parameter_changes':0}
 atomic_json(folder/'manifest.json',manifest)
 for tick in range(30):
  t=tick*.1;rates=rates_for(cue,t+1e-7,encoder);samples=rng.random((1000,len(targets)))<rates*.0001;times,ii=np.nonzero(samples);generator.set_spikes(ii,(t+times*.0001)*b.second,sorted=True)
  brain.network.run(.1*b.second,namespace={});counts=np.asarray(brain.monitor.count[:],dtype=np.int64);delta=counts-previous;previous=counts.copy();ic=np.asarray(im.count[:],dtype=np.int64);inp=ic-before_input;before_input=ic.copy()
  spike_i=np.asarray(brain.monitor.i[:],dtype=np.int32).copy();spike_t=np.asarray(brain.monitor.t[:]/b.second).copy();assert len(spike_i)==delta.sum()
  population_rates={k:float(delta[[r['index'] for r in v]].mean()/.1) if v else None for k,v in pops.items()}
  assert np.isfinite(brain.neurons.v[:]/b.mV).all() and np.isfinite(brain.neurons.g[:]/b.mV).all()
  rows.append({'time':(tick+1)*.1,'population_hz':population_rates,'total_spikes':int(delta.sum())});file=folder/f'window-{tick:04d}.npz';space_check(folder,delta.nbytes+inp.nbytes+spike_i.nbytes+spike_t.nbytes+1024**2)
  with file.with_suffix('.tmp').open('wb') as f:np.savez_compressed(f,counts=delta,input_counts=inp,requested_hz=rates,spike_i=spike_i,spike_t=spike_t)
  file.with_suffix('.tmp').replace(file);brain.clear_recorded_spikes();manifest['chunks'].append({'file':file.name,'sha256':sha(file),'time':rows[-1]['time'],'spikes':int(delta.sum())});atomic_json(folder/'manifest.json',manifest);atomic_json(folder/'bins.json',rows)
 assert np.array_equal(weights,brain.learned_state()['weights']) and brain.plasticity.updates==0
 manifest.update(status='complete',wall_seconds=time.perf_counter()-start,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss);atomic_json(folder/'manifest.json',manifest);print(cue,seed,'complete',flush=True)
def run(root):
 root.mkdir(parents=True,exist_ok=True);protocol=root/'protocol.json'
 if not protocol.exists():atomic_json(protocol,{'seeds':SEEDS,'held_out_seed':5199,'conditions':CONDITIONS,'sources':{name:sha(ROOT/name) for name in SOURCES},'protocol':'3s continuous trial; background at0; odor0.5-1s; second opposite pulse1.5-2s for switch; 100us Bernoulli external events; reference external voltage jump68.75mV; only stimulated roots zero refractory. Learning and synapses fixed.','gates':{'steering':'For both named odors, left-minus-right DNa02 change relative to same-seed baseline must exceed +5Hz for left cue and -5Hz for right cue during0.5-1s on all three seeds. Oil comparisons separately reported.','recovery':'Report odor-minus-baseline whole-network activity1-1.5s; baseline firing is not expected to be zero.','promotion':'No automatic production promotion; sensory input fidelity and body closed-loop validation remain separate requirements.'}})
 p=json.loads(protocol.read_text());assert all(sha(ROOT/k)==v for k,v in p['sources'].items())
 progress={'status':'running','planned':33,'completed':[]};started=time.perf_counter()
 try:
  for cue in CONDITIONS:
   for seed in SEEDS:
    folder=root/'trials'/f'{cue}-{seed}'
    if (folder/'manifest.json').exists() and json.loads((folder/'manifest.json').read_text())['status']=='complete':progress['completed'].append(folder.name);continue
    if folder.exists():raise RuntimeError('Preserve incomplete attempt before retry '+str(folder))
    space_check(root,64*1024**2);progress['current']={'cue':cue,'seed':seed};atomic_json(root/'progress.json',progress);log=root/'logs'/f'{cue}-{seed}.log';log.parent.mkdir(exist_ok=True)
    with log.open('w') as out:subprocess.run([sys.executable,str(Path(__file__).resolve()),str(folder),'--cue',cue,'--seed',str(seed)],cwd=ROOT,stdout=out,stderr=subprocess.STDOUT,check=True)
    progress['completed'].append(folder.name);progress['wall_seconds']=time.perf_counter()-started;atomic_json(root/'progress.json',progress);print(f"{len(progress['completed'])}/33 {folder.name}",flush=True)
  progress['status']='complete';atomic_json(root/'progress.json',progress)
 except Exception as e:progress.update(status='interrupted',error=repr(e));atomic_json(root/'progress.json',progress);raise
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('folder',type=Path);p.add_argument('--cue',choices=CONDITIONS);p.add_argument('--seed',type=int);a=p.parse_args()
 if a.cue:worker(a.folder,a.cue,a.seed)
 else:run(a.folder)
