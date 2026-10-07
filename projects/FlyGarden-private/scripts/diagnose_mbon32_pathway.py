"""Annotated downstream circuit capability, never sensory/navigation substitution."""
import sys,json,subprocess,fcntl,argparse,time,resource,shutil
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from flygarden.continuous_candidate import file_sha
from flygarden.recording import atomic_json,space_check
OUT=ROOT/'reports/brain-integration/recovery/mbon32-capability-v1'
SOURCES=('scripts/diagnose_mbon32_pathway.py','flygarden/continuous_candidate.py','flygarden/brain.py','flygarden/candidate_inputs.py','flygarden/descending.py','flygarden/neural_probe.py','data/annotations.tsv','vendor/fly-brain/data/2025_Connectivity_783.parquet','vendor/fly-brain/data/2025_Completeness_783.csv')
def prepare():
 import pandas as pd
 from scripts.diagnose_recovery_interfaces import mappings
 ids,mapping=mappings();ann=pd.read_csv(ROOT/'data/annotations.tsv',sep='\t',low_memory=False).fillna('');index={int(n):i for i,n in enumerate(ids)};roots=[];targets=[]
 for side in ('left','right'):
  rows=ann[ann.cell_type.eq('MBON32')&ann.side.eq(side)&ann.root_id.isin(ids)];assert len(rows)==1
  root=int(rows.iloc[0].root_id);roots.append(str(root));targets.append(index[root])
 selected=sorted(set(targets+[n['index'] for key,pop in mapping.items() if key.startswith(('DNa','DNp')) for n in pop]))
 spec={'schema_version':1,'seeds':[10201,10202],'cases':['none','left','right','both'],'duration':1.5,'interval':.025,'mapping':mapping,'roots':roots,'targets':targets,'selected':selected,'sources':{n:file_sha(ROOT/n) for n in SOURCES},'schedule':'65Hz bilateral DNp09 support from0; MBON32 directly driven50Hz .3-.8; recovery.8-1.5; no odors or vision. Original full network fixed, no learning.', 'scope':'Downstream capability diagnostic only. Not olfactory processing, navigation, acquisition or a proposed production controller. Zero refractory original possible input targets plus the two directly stimulated MBON32 roots in all cases.', 'prospective_gate':'Both seeds: pulse signed DNa02 left-right difference relative to matched none >=5Hz for left and <=-5Hz for right; bilateral within5Hz; every recovery within5Hz. No promotion even if passed.', 'source':'https://elifesciences.org/articles/102230'}
 OUT.mkdir(parents=True,exist_ok=True);p=OUT/'protocol.json'
 if p.exists():assert json.loads(p.read_text())==spec
 else:
  atomic_json(p,spec)
  for n in SOURCES:
   if n.startswith(('data/','vendor/')):continue
   dest=OUT/'sources'/n;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/n,dest)
 return spec

def worker(cue,seed):
 import brian2 as b
 from flygarden.continuous_candidate import ContinuousCandidate
 from flygarden.neural_probe import NeuralProbe
 spec=prepare();folder=OUT/f'{cue}-{seed}';folder.mkdir(exist_ok=False);start=time.monotonic();cand=ContinuousCandidate(spec['mapping'],seed)
 gen=b.SpikeGeneratorGroup(2,[],[]*b.second,clock=cand.brain.clock);ext=b.Synapses(gen,cand.brain.neurons,on_pre='v_post+=68.75*mV',clock=cand.brain.clock);ext.connect(i=[0,1],j=spec['targets']);cand.brain.neurons.rfc[spec['targets']]=0*b.ms;cand.brain.network.add(gen,ext)
 probe=NeuralProbe(cand.brain,spec['selected'],gen);rng=np.random.default_rng(seed+100000);m={'status':'running','case':cue,'seed':seed,'sources':spec['sources'],'chunks':[],'controller':cand.manifest(),'roots':spec['roots'],'additional_zero_refractory_indices':spec['targets'],'learning':False};atomic_json(folder/'manifest.json',m);rows=[]
 for k in range(60):
  rates=np.zeros(2)
  if 12<=k<32:
   if cue in ('left','both'):rates[0]=50
   if cue in ('right','both'):rates[1]=50
  ticks,ii=np.nonzero(rng.random((250,2))<rates*.0001);times=(k*250+ticks)*.0001;gen.set_spikes(ii,times*b.second,sorted=True);command=cand.advance(.025,np.zeros((2,2)),[0,0]);data=probe.drain();assert np.array_equal(data['delivered_external_indices'],ii);assert np.allclose(data['delivered_external_times_seconds'],times,atol=1e-12,rtol=0)
  path=folder/f'window-{k:04d}.npz';space_check(folder,8*1024**2)
  with path.with_suffix('.tmp').open('wb') as stream:np.savez_compressed(stream,**{n:v for n,v in data.items() if not isinstance(v,str)},spike_i=cand.last_spikes[0],spike_t=cand.last_spikes[1],counts=cand.last['spike_counts'],support_i=cand.last['external_indices'],support_t=cand.last['external_times'],mbon_i=ii,mbon_t=times)
  path.with_suffix('.tmp').replace(path);m['chunks'].append({'file':path.name,'sha256':file_sha(path)});rows.append({'time':cand.time,'population_hz':cand.last['population_hz'],'motor':command.tolist()});atomic_json(folder/'manifest.json',m);atomic_json(folder/'rows.json',rows)
 assert cand.brain.plasticity.updates==0;m.update(status='complete',wall_seconds=time.monotonic()-start,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss);atomic_json(folder/'manifest.json',m);probe.close();print(cue,seed,'complete',flush=True)

def run():
 with (ROOT/'.runtime/experiment.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);s=prepare()
  for cue in s['cases']:
   for seed in s['seeds']:
    p=OUT/f'{cue}-{seed}'/'manifest.json'
    if p.exists():assert json.loads(p.read_text())['status']=='complete';continue
    subprocess.run([sys.executable,__file__,'--cue',cue,'--seed',str(seed)],check=True,cwd=ROOT)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--prepare',action='store_true');p.add_argument('--cue');p.add_argument('--seed',type=int);a=p.parse_args()
 if a.prepare:prepare()
 elif a.cue:worker(a.cue,a.seed)
 else:run()
