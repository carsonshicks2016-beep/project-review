"""Stage 3 full-network, lateralized sensory-to-descending diagnostic.

No production edits, learning, recurrence changes, or arena coordinates.
One independently reconstructed full network per trial; continuous state within it.
"""
import argparse,json,sys,time,subprocess,resource
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from flygarden.recording import atomic_json,space_check
from scripts.audit_brain_reference import sha
SEEDS=(3101,3102,3199)
CONDITIONS=('quiet','a_left','a_right','a_both','b_left','b_right','a_switch','walking','walking_a_left','walking_a_right','direct_left','direct_right')
SOURCES=('flygarden/brain.py','flygarden/body.py','data/annotations.tsv','vendor/fly-brain/data/2025_Completeness_783.csv','vendor/fly-brain/data/2025_Connectivity_783.parquet','scripts/diagnose_sensorimotor_pathway.py','flygarden/descending.py')

def mapping():
 import pandas as pd
 ids=pd.read_csv(ROOT/'vendor/fly-brain/data/2025_Completeness_783.csv',index_col=0).index.to_numpy(dtype=np.int64)
 ann=pd.read_csv(ROOT/'data/annotations.tsv',sep='\t',low_memory=False).fillna('');ann=ann[ann.root_id.isin(ids)]
 order={int(v):i for i,v in enumerate(ids)};pops={}
 for cell in ('ORN_DM1','ORN_DM2','DM1_lPN','DM2_lPN','DNa02','DNp09','DNg13'):
  for side in ('left','right'):
   rows=ann[ann.cell_type.eq(cell)&ann.side.eq(side)]
   pops[f'{cell}_{side}']=[{'root_id':str(int(r.root_id)),'index':order[int(r.root_id)],'cell_type':cell,'side':side} for r in rows.itertuples()]
 for cell in ('ORN_DM1','ORN_DM2','DNa02','DNp09'):
  if not all(pops[f'{cell}_{side}'] for side in ('left','right')):raise ValueError('Required exact annotation absent: '+cell)
 return ids,pops,ann

def inputs(cue,t):
 values=np.zeros(8) # A left/right, B left/right, P9 left/right, DNa02 left/right
 pulse=.3<=t<.8
 if cue.startswith('walking') and t>=.3:values[4:6]=65
 if pulse:
  if cue in ('a_left','a_switch','walking_a_left'):values[0]=50
  if cue in ('a_right','walking_a_right'):values[1]=50
  if cue=='a_both':values[:2]=50
  if cue=='b_left':values[2]=50
  if cue=='b_right':values[3]=50
  if cue=='direct_left':values[6]=50
  if cue=='direct_right':values[7]=50
 if cue=='a_switch' and 1.3<=t<1.8:values[1]=50
 return values

def worker(folder,cue,seed):
 import brian2 as b
 from flygarden.brain import FullBrain
 from flygarden.descending import DescendingDecoder
 start=time.perf_counter();brain=FullBrain(seed=seed,learning=False,record_spikes=False)
 _,pops,_=mapping()
 keys=('ORN_DM1_left','ORN_DM1_right','ORN_DM2_left','ORN_DM2_right','DNp09_left','DNp09_right','DNa02_left','DNa02_right')
 groups=[np.array([r['index'] for r in pops[k]],dtype=np.int32) for k in keys]
 targets=np.concatenate(groups);channels=np.concatenate([np.full(len(g),i,dtype=np.int32) for i,g in enumerate(groups)])
 generator=b.SpikeGeneratorGroup(len(targets),np.array([],dtype=np.int32),np.array([])*b.second,clock=brain.clock)
 external=b.Synapses(generator,brain.neurons,on_pre='v_post += 68.75*mV',clock=brain.clock);external.connect(i=np.arange(len(targets)),j=targets)
 im=b.SpikeMonitor(generator,record=False);brain.network.add(generator,external,im);brain.inputs.active=False
 # Match the reference stimulation convention only for neurons actually stimulated.
 brain.neurons.rfc=2.2*b.ms
 selected=np.any(np.array([inputs(cue,tick*.1+.00001) for tick in range(30)])>0,axis=0)
 brain.neurons.rfc[targets[selected[channels]]]=0*b.ms
 rng=np.random.default_rng(seed);previous=np.zeros(brain.n,dtype=np.int64);before_input=np.zeros(len(targets),dtype=np.int64);decoder=DescendingDecoder();rows=[];weights=brain.learned_state()['weights'].copy()
 folder.mkdir(parents=True,exist_ok=False);manifest={'schema_version':1,'status':'running','cue':cue,'seed':seed,'neurons':brain.n,'connections':brain.edges,'chunks':[],'learning':False,'sources':{name:sha(ROOT/name) for name in SOURCES},'input_keys':keys,'input_root_ids':[str(brain.ids[i]) for i in targets],'input_mapping_indices':targets.tolist(),'refractory_zero_root_ids':[str(brain.ids[i]) for i in targets[selected[channels]]],'clock_seconds':.0001,'bin_seconds':.1}
 atomic_json(folder/'manifest.json',manifest)
 for tick in range(30):
  t=tick*.1;rates=inputs(cue,t+1e-7);samples=rng.random((1000,len(targets)))<rates[channels]*.0001;times,ii=np.nonzero(samples);generator.set_spikes(ii,(t+times*.0001)*b.second,sorted=True)
  brain.network.run(.1*b.second,namespace={});counts=np.asarray(brain.monitor.count[:],dtype=np.int64);delta=counts-previous;previous=counts.copy();ic=np.asarray(im.count[:],dtype=np.int64);inp=ic-before_input;before_input=ic.copy()
  population_rates={k:float(delta[[r['index'] for r in v]].mean()/.1) if v else None for k,v in pops.items()}
  motor=decoder.advance(.1,population_rates)
  assert np.isfinite(brain.neurons.v[:]/b.mV).all() and np.isfinite(brain.neurons.g[:]/b.mV).all()
  row={'time':(tick+1)*.1,'input_hz':rates.tolist(),'input_events':inp.tolist(),'population_hz':population_rates,'candidate_motor':motor.tolist(),'candidate_turn':float(motor[1]-motor[0])}
  rows.append(row);file=folder/f'window-{tick:04d}.npz';space_check(folder,delta.nbytes+inp.nbytes+1024**2)
  with file.with_suffix('.tmp').open('wb') as f:np.savez_compressed(f,counts=delta,input_counts=inp)
  file.with_suffix('.tmp').replace(file);manifest['chunks'].append({'file':file.name,'sha256':sha(file),'time':row['time'],'spikes':int(delta.sum())});atomic_json(folder/'manifest.json',manifest)
  atomic_json(folder/'bins.json',rows)
 assert np.array_equal(weights,brain.learned_state()['weights']) and brain.plasticity.updates==0
 manifest.update(status='complete',wall_seconds=time.perf_counter()-start,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss);atomic_json(folder/'manifest.json',manifest)
 print(cue,seed,'complete',flush=True)

def run(root):
 root.mkdir(parents=True,exist_ok=True);_,pops,_=mapping()
 protocol=root/'protocol.json'
 if not protocol.exists():
  atomic_json(protocol,{'schema_version':1,'seeds':SEEDS,'held_out_seed':3199,'conditions':CONDITIONS,'neurons':138639,'connections':15091983,'sources':{name:sha(ROOT/name) for name in SOURCES},'mapping':pops,'claim':'Exact-root-ID engineered bilateral odor encoding; DNa02 candidate steering; artificial direct-output controls separate.','sources_supporting_mapping':['https://elifesciences.org/articles/102230','https://pmc.ncbi.nlm.nih.gov/articles/PMC12778575/'],'protocol':'3s trial; 0.3s rested baseline; 0.3-0.8s 50Hz pulse; switch second pulse1.3-1.8s; walking tonic65Hz each P9 from0.3s; continuous neural state; frozen learning.','gates':{'odor_steering':'Held-out lateral cue difference must reverse sign on left/right, exceed 5Hz during0.3-0.5s, and be consistent on both calibration seeds. Recovery0.8-1.3s is separately reported. No gain tuning.','decoder':'Fixed P9 mean/100Hz forward and DNa02(left-right)/100Hz steering, bounded + smoothed; direct controls do not establish sensory behavior.','promotion':'No arena reconnection unless cue gate and body direction validation both pass.'}})
 p=json.loads(protocol.read_text());assert all(sha(ROOT/k)==v for k,v in p['sources'].items()),'Source changed after protocol commitment'
 progress={'status':'running','planned':len(SEEDS)*len(CONDITIONS),'completed':[]};started=time.perf_counter()
 try:
  for cue in CONDITIONS:
   for seed in SEEDS:
    folder=root/'trials'/f'{cue}-{seed}'
    if (folder/'manifest.json').exists() and json.loads((folder/'manifest.json').read_text())['status']=='complete':progress['completed'].append(folder.name);continue
    if folder.exists():raise RuntimeError('Preserve incomplete attempt before retry: '+str(folder))
    space_check(root,32*1024**2);progress['current']={'cue':cue,'seed':seed};atomic_json(root/'progress.json',progress)
    log=root/'logs'/f'{cue}-{seed}.log';log.parent.mkdir(exist_ok=True)
    with log.open('w') as out:subprocess.run([sys.executable,str(Path(__file__).resolve()),str(folder),'--cue',cue,'--seed',str(seed)],cwd=ROOT,stdout=out,stderr=subprocess.STDOUT,check=True)
    progress['completed'].append(folder.name);progress['wall_seconds']=time.perf_counter()-started;atomic_json(root/'progress.json',progress);print(f"{len(progress['completed'])}/{progress['planned']} {folder.name}",flush=True)
  progress['status']='complete';atomic_json(root/'progress.json',progress)
 except Exception as e:
  progress.update(status='interrupted',error=repr(e));atomic_json(root/'progress.json',progress);raise
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('folder',type=Path);p.add_argument('--cue',choices=CONDITIONS);p.add_argument('--seed',type=int);a=p.parse_args()
 if a.cue:worker(a.folder,a.cue,a.seed)
 else:run(a.folder)
