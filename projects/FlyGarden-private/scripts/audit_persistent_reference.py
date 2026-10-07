"""Longer shared-input reference equation replay; no controller promotion."""
import sys,json,time,fcntl,subprocess,argparse,resource,shutil
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from flygarden.recording import atomic_json,space_check
from flygarden.continuous_candidate import file_sha
OUT=ROOT/'reports/brain-integration/recovery/persistence-reference-v1'
SOURCES=('scripts/audit_persistent_reference.py','scripts/audit_brain_reference.py','flygarden/brain.py','flygarden/continuous_candidate.py','flygarden/candidate_inputs.py','flygarden/neural_probe.py','data/annotations.tsv','vendor/fly-brain/code/run_brian2_cuda.py','vendor/fly-brain/data/2025_Completeness_783.csv','vendor/fly-brain/data/2025_Connectivity_783.parquet')
def prepare():
 from scripts.diagnose_recovery_interfaces import mappings
 _,mapping=mappings();old=json.loads((ROOT/'reports/brain-integration/recovery/odor-bias-diagnosis/diagnostic-protocol.json').read_text())
 selected=set(old['selected_probe_indices'])
 import pandas as pd
 ann=pd.read_csv(ROOT/'data/annotations.tsv',sep='\t',low_memory=False).fillna('');ids=pd.read_csv(ROOT/'vendor/fly-brain/data/2025_Completeness_783.csv',index_col=0).index.to_numpy(np.int64);index={int(n):i for i,n in enumerate(ids)}
 selected.update(index[int(n)] for n in ann.loc[ann.cell_type.isin(['MBON32','APL','AOTU019']), 'root_id'] if int(n) in index)
 spec={'schema_version':1,'duration':4.,'interval':.025,'seeds':[10101,10102],'supports':[0,65],'backends':['candidate','reference'],'mapping':mapping,'selected':sorted(selected),'sources':{n:file_sha(ROOT/n) for n in SOURCES},'schedule':'OdorA left at5Hz per root during .3-.8s; support0/65Hz from0; no other sensory input;3.2s recovery. Shared external events; full fixed network; learning off.', 'reference_scope':'Released parameter/equation/weight construction independent of FullBrain. Shared Bernoulli generator instead of native PoissonInput; redundant reset w=0 omitted as in earlier registered equation replay. Not published behavioral replication.', 'decision':'All spike IDs/times/counts and inputs exact; selected v/g within prior absolute1e-10mV tolerance,rtol0 at1ms. Persistence is descriptive: report last .5s rates and tail activity; no invented navigation gate.'}
 OUT.mkdir(parents=True,exist_ok=True);path=OUT/'protocol.json'
 if path.exists():assert json.loads(path.read_text())==spec
 else:
  atomic_json(path,spec)
  for n in SOURCES:
   if n.startswith(('data/','vendor/fly-brain/data/')):continue
   dest=OUT/'sources'/n;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/n,dest)
 return spec

def worker(backend,seed,support):
 import brian2 as b,pandas as pd
 from types import SimpleNamespace
 from flygarden.continuous_candidate import ContinuousCandidate
 from flygarden.candidate_inputs import CandidateInputs,INPUT_KEYS
 from flygarden.neural_probe import NeuralProbe
 from scripts.audit_brain_reference import params
 spec=prepare();folder=OUT/f'{backend}-{seed}-support{support}';folder.mkdir(exist_ok=False);started=time.monotonic()
 if backend=='candidate':
  cand=ContinuousCandidate(spec['mapping'],seed,CandidateInputs(support));brain=cand.brain;targets=cand.targets;channels=cand.channels;gen=cand.generator
 else:
  b.prefs.codegen.target='cython';p=params();clock=b.Clock(dt=.1*b.ms)
  ids=pd.read_csv(ROOT/'vendor/fly-brain/data/2025_Completeness_783.csv',index_col=0).index.to_numpy(np.int64)
  ann=pd.read_csv(ROOT/'data/annotations.tsv',sep='\t',low_memory=False).fillna('');index={int(n):i for i,n in enumerate(ids)};targets=[];channels=[]
  for ch,key in enumerate(INPUT_KEYS):
   cell,side=key.rsplit('_',1);entries=ann.loc[ann.cell_type.eq(cell)&ann.side.eq(side)&ann.root_id.isin(ids),'root_id']
   actual=[index[int(n)] for n in entries];assert set(actual)=={n['index'] for n in spec['mapping'][key]}
   # Shared event column order is registered and independently checked against annotation sets.
   actual=[n['index'] for n in spec['mapping'][key]];targets.extend(actual);channels.extend([ch]*len(actual))
  targets=np.array(targets);channels=np.array(channels)
  neu=b.NeuronGroup(len(ids),p['eqs'],threshold=p['eq_th'],reset='v = v_rst; g = 0 * mV',refractory='rfc',method='linear',namespace=p,clock=clock)
  neu.v=p['v_0'];neu.g=0*b.mV;neu.rfc=p['t_rfc'];neu.rfc[targets]=0*b.ms
  con=pd.read_parquet(ROOT/'vendor/fly-brain/data/2025_Connectivity_783.parquet',columns=['Presynaptic_Index','Postsynaptic_Index','Excitatory x Connectivity'])
  syn=b.Synapses(neu,neu,'w:volt',on_pre='g += w',delay=p['t_dly'],clock=clock);syn.connect(i=con.Presynaptic_Index.to_numpy(np.int32),j=con.Postsynaptic_Index.to_numpy(np.int32));syn.w=con['Excitatory x Connectivity'].to_numpy()*p['w_syn'];del con
  gen=b.SpikeGeneratorGroup(len(targets),[],[]*b.second,clock=clock);ext=b.Synapses(gen,neu,on_pre='v_post += amplitude',namespace={'amplitude':p['w_syn']*p['f_poi']},clock=clock);ext.connect(i=np.arange(len(targets)),j=targets)
  monitor=b.SpikeMonitor(neu);net=b.Network(neu,syn,gen,ext,monitor);brain=SimpleNamespace(network=net,neurons=neu,n=len(ids),ids=ids,monitor=monitor)
 probe=NeuralProbe(brain,spec['selected'],gen);rng=np.random.default_rng(seed);previous=np.zeros(brain.n,dtype=np.int64)
 m={'status':'running','sources':spec['sources'],'backend':backend,'seed':seed,'support':support,'chunks':[],'n':brain.n,'input_indices':targets.tolist(),'input_channels':channels.tolist(),'learning':False};atomic_json(folder/'manifest.json',m)
 try:
  for k in range(160):
   odors=np.zeros((2,2));odors[0,0]=.1 if 12<=k<32 else 0
   rates=np.zeros(8);rates[4:6]=support;rates[0]=5 if 12<=k<32 else 0
   ticks,ii=np.nonzero(rng.random((250,len(targets)))<rates[channels]*.0001);events=(k*250+ticks)*.0001
   if backend=='candidate':
    cand.advance(.025,odors,[0,0]);si,st=cand.last_spikes;counts=cand.last['spike_counts'];assert np.array_equal(ii,cand.last['external_indices']);assert np.allclose(events,cand.last['external_times'],atol=1e-12,rtol=0)
   else:
    gen.set_spikes(ii,events*b.second,sorted=True);brain.network.run(.025*b.second,namespace={});total=np.asarray(brain.monitor.count[:],dtype=np.int64);counts=total-previous;previous=total.copy();si=np.asarray(brain.monitor.i[:],dtype=np.int32).copy();st=np.asarray(brain.monitor.t[:]/b.second).copy();brain.monitor.resize(0);brain.monitor.variables['N'].set_value(0)
   data=probe.drain();assert np.isfinite(data['voltage_mV']).all() and np.isfinite(data['net_synaptic_conductance_equivalent_mV']).all()
   path=folder/f'window-{k:04d}.npz';space_check(folder,8*1024**2)
   with path.with_suffix('.tmp').open('wb') as stream:np.savez_compressed(stream,**{n:v for n,v in data.items() if not isinstance(v,str)},spike_i=si,spike_t=st,counts=counts,external_i=ii,external_t=events)
   path.with_suffix('.tmp').replace(path);m['chunks'].append({'file':path.name,'sha256':file_sha(path)});atomic_json(folder/'manifest.json',m)
  m.update(status='complete',wall_seconds=time.monotonic()-started,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss);atomic_json(folder/'manifest.json',m);print(folder.name,'complete',flush=True)
 except BaseException as e:m.update(status='interrupted',error=repr(e));atomic_json(folder/'manifest.json',m);raise
 finally:probe.close()

def run():
 with (ROOT/'.runtime/experiment.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);s=prepare()
  for seed in s['seeds']:
   for support in s['supports']:
    for backend in s['backends']:
     p=OUT/f'{backend}-{seed}-support{support}'/'manifest.json'
     if p.exists():assert json.loads(p.read_text())['status']=='complete';continue
     subprocess.run([sys.executable,__file__,'--backend',backend,'--seed',str(seed),'--support',str(support)],check=True,cwd=ROOT)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--backend');p.add_argument('--seed',type=int);p.add_argument('--support',type=int);a=p.parse_args()
 if a.backend:worker(a.backend,a.seed,a.support)
 else:run()
