"""Step4: targeted model causal interventions and DNa02 delivery/state readout.

All imported neurons/connections remain present. Lesioned weights are explicit,
private diagnostic interventions, never a production controller or training.
"""
import argparse,json,sys,time,subprocess,resource
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from flygarden.recording import atomic_json,space_check
from scripts.audit_brain_reference import sha
from scripts.diagnose_sensorimotor_pathway import mapping
SEEDS=(4101,4102,4199)
PROFILES=('intact_narrow','intact_broad','right_negative_off','left_negative_off','PS049_negative_off','steering_feedback_off')
SOURCE_FILES=('flygarden/brain.py','data/annotations.tsv','vendor/fly-brain/data/2025_Completeness_783.csv','vendor/fly-brain/data/2025_Connectivity_783.parquet','scripts/diagnose_steering_bias.py','scripts/diagnose_sensorimotor_pathway.py')

class SteeringObserver:
 def __init__(self,brain,targets,enabled=True):
  import brian2 as b
  self.targets=np.asarray(targets,dtype=np.int32);group=brain.neurons
  for name in ('bias_exc','bias_inh'):group.variables.add_array(name,size=len(group),dimensions=b.volt.dim)
  post=np.asarray(brain.synapses.j[:],dtype=np.int32);self.edges=np.flatnonzero(np.isin(post,targets));pre=np.asarray(brain.synapses.i[self.edges],dtype=np.int32)
  self.synapses=b.Synapses(group,group,'weight:volt',on_pre='bias_exc_post += clip(weight,0*mV,inf*mV)*int(not_refractory_post)\nbias_inh_post += clip(-weight,0*mV,inf*mV)*int(not_refractory_post)',delay=1.8*b.ms,clock=brain.clock)
  self.synapses.connect(i=pre,j=post[self.edges]);self.synapses.weight=brain.synapses.w[self.edges];self.synapses.active=enabled
  self.state=b.StateMonitor(group,['v','g','not_refractory'],record=self.targets,dt=.1*b.ms,when='end');self.state.active=enabled
  brain.network.add(self.synapses,self.state);self.group=group
 def counters(self):
  import brian2 as b
  return np.stack([np.asarray(getattr(self.group,name)[self.targets]/b.mV) for name in ('bias_exc','bias_inh')])

def weights_mask(profile,pre,post,signed,targets,ps049):
 if profile=='right_negative_off':return (post==targets[1])&(signed<0)
 if profile=='left_negative_off':return (post==targets[0])&(signed<0)
 if profile=='PS049_negative_off':return np.isin(post,targets)&np.isin(pre,ps049)&(signed<0)
 if profile=='steering_feedback_off':return np.isin(pre,targets)
 return np.zeros(len(pre),dtype=bool)

def setup(root):
 import pandas as pd
 ids,pops,ann=mapping();targets=[pops['DNa02_left'][0]['index'],pops['DNa02_right'][0]['index']];order={int(v):i for i,v in enumerate(ids)};annot=ann.set_index('root_id')
 con=pd.read_parquet(ROOT/'vendor/fly-brain/data/2025_Connectivity_783.parquet',columns=['Presynaptic_Index','Postsynaptic_Index','Excitatory x Connectivity','Connectivity']);pre=con.Presynaptic_Index.to_numpy(dtype=np.int32);post=con.Postsynaptic_Index.to_numpy(dtype=np.int32);signed=con['Excitatory x Connectivity'].to_numpy()
 ps049=np.array([order[int(v)] for v in ann.loc[ann.cell_type.eq('PS049'),'root_id']],dtype=np.int32);assert len(ps049)>0
 broad={side:[{'root_id':str(int(v)),'index':order[int(v)]} for v in ann.loc[ann.cell_type.str.startswith('ORN')&ann.side.eq(side),'root_id']] for side in ('left','right')};assert all(broad.values())
 incoming=[]
 for i in np.flatnonzero(np.isin(post,targets)):
  label=annot.loc[int(ids[pre[i]])] if int(ids[pre[i]]) in annot.index else {}
  incoming.append({'edge_index':int(i),'pre_index':int(pre[i]),'post_index':int(post[i]),'pre_root_id':str(ids[pre[i]]),'post_root_id':str(ids[post[i]]),'signed_count':float(signed[i]),'anatomical_count':int(con.Connectivity.iloc[i]),'pre_cell_type':label.get('cell_type',''),'pre_side':label.get('side','')})
 lesions={}
 for profile in PROFILES:
  mask=weights_mask(profile,pre,post,signed,targets,ps049);rows=np.flatnonzero(mask)
  lesions[profile]={'edge_indices':rows.tolist(),'records_zeroed':int(mask.sum()),'anatomical_synapses_affected':int(con.Connectivity.to_numpy()[mask].sum()),'source_indices':np.unique(pre[mask]).tolist(),'target_indices':np.unique(post[mask]).tolist(),'intervention':'Diagnostic zeroing, all cells/edge records retained; not a biological knockout or usable full-function controller.' if mask.any() else 'No weight changes'}
 result={'schema_version':1,'seeds':SEEDS,'held_out_seed':4199,'profiles':PROFILES,'cues':['left','right'],'neurons':len(ids),'connections':len(pre),'sources':{name:sha(ROOT/name) for name in SOURCE_FILES},'DNa02_targets':targets,'DNa02_root_ids':[str(ids[i]) for i in targets],'narrow_populations':{side:pops['ORN_DM1_'+side] for side in ('left','right')},'broad_populations':broad,'excluded_broad_ORNs_without_left_right_annotation':int((ann.cell_type.str.startswith('ORN')&~ann.side.isin(['left','right'])).sum()),'PS049_root_ids':[str(ids[i]) for i in ps049],'incoming_edges':incoming,'lesions':lesions,'protocol':'2 simulated seconds, rest0-0.3,50Hz unilateral odor pulse0.3-0.8,recovery0.8-2;100us neural clock;0.1ms state samples;100ms chunks;only actual stimulated ORNs zero refractory;rest all2.2ms;no P9,loom,reward,body or learning.','hypotheses':['Broader annotated ORN recruitment restores lateralized DNa02 responses.','Right DNa02 silence depends on its negative modeled synaptic input.','PS049 negative delivery contributes to the bias.','DNa02 outgoing feedback is required for the bias.'],'decision':'Broad intact input must yield left-minus-right DNa02 >5Hz for left cue and <-5Hz for right cue in early0.3-0.5s on all seeds, with both outputs returning below5Hz in final0.5s. Lesions are diagnostic only. No automatic promotion.'}
 atomic_json(root/'protocol.json',result);return result

def worker(root,profile,cue,seed,observer_enabled=True):
 import brian2 as b
 from flygarden.brain import FullBrain
 start=time.perf_counter();protocol=json.loads((root/'protocol.json').read_text());brain=FullBrain(seed=seed,learning=False,record_spikes=True)
 assert brain.n==protocol['neurons'] and brain.edges==protocol['connections']
 assert all(sha(ROOT/name)==digest for name,digest in protocol['sources'].items())
 zero=np.array(protocol['lesions'][profile]['edge_indices'],dtype=np.int64);original=np.asarray(brain.synapses.w[zero]/b.mV).copy() if len(zero) else np.array([])
 if len(zero):brain.synapses.w[zero]=0*b.mV
 groups=protocol['broad_populations'] if profile=='intact_broad' else protocol['narrow_populations']
 population=groups[cue];targets=np.array([r['index'] for r in population],dtype=np.int32)
 external=b.SpikeGeneratorGroup(len(targets),np.array([],dtype=np.int32),np.array([])*b.second,clock=brain.clock)
 incoming=b.Synapses(external,brain.neurons,on_pre='v_post += 68.75*mV',clock=brain.clock);incoming.connect(i=np.arange(len(targets)),j=targets);im=b.SpikeMonitor(external,record=False)
 brain.network.add(external,incoming,im);brain.inputs.active=False;brain.neurons.rfc=2.2*b.ms;brain.neurons.rfc[targets]=0*b.ms
 observer=SteeringObserver(brain,protocol['DNa02_targets'],enabled=observer_enabled);rng=np.random.default_rng(seed)
 folder=root/('trials' if observer_enabled else 'instrumentation-control')/f'{profile}-{cue}-{seed}';folder.mkdir(parents=True,exist_ok=False)
 manifest={'schema_version':1,'status':'running','profile':profile,'cue':cue,'seed':seed,'neurons':brain.n,'connections':brain.edges,'observer_enabled':observer_enabled,'input_root_ids':[r['root_id'] for r in population],'input_indices':targets.tolist(),'lesion_records':len(zero),'sources':protocol['sources'],'chunks':[]}
 atomic_json(folder/'manifest.json',manifest);previous=np.zeros(brain.n,dtype=np.int64);previous_in=np.zeros(len(targets),dtype=np.int64);previous_delivered=np.zeros((2,2));rows=[];base_weights=brain.learned_state()['weights'].copy()
 for tick in range(20):
  t=tick*.1;rate=50 if 3<=tick<8 else 0;sample=rng.random((1000,len(targets)))<rate*.0001;ticks,neurons=np.nonzero(sample);external.set_spikes(neurons,(t+ticks*.0001)*b.second,sorted=True)
  brain.network.run(.1*b.second,namespace={});counts=np.asarray(brain.monitor.count[:],dtype=np.int64);delta=counts-previous;previous=counts.copy();ii=np.asarray(brain.monitor.i[:],dtype=np.int32).copy();tt=np.asarray(brain.monitor.t[:]/b.second).copy();brain.clear_recorded_spikes()
  assert len(ii)==int(delta.sum())
  total_in=np.asarray(im.count[:],dtype=np.int64);input_counts=total_in-previous_in;previous_in=total_in.copy();delivered=observer.counters();delivery=delivered-previous_delivered;previous_delivered=delivered.copy()
  if observer_enabled:
   vv=np.asarray(observer.state.v[:]/b.mV).copy();gg=np.asarray(observer.state.g[:]/b.mV).copy();gate=np.asarray(observer.state.not_refractory[:],dtype=bool).copy();st=np.asarray(observer.state.t[:]/b.second).copy();observer.state.resize(0)
  else:vv=np.empty((2,0));gg=np.empty((2,0));gate=np.empty((2,0),bool);st=np.array([])
  assert np.isfinite(brain.neurons.v[:]/b.mV).all() and np.isfinite(brain.neurons.g[:]/b.mV).all()
  output=delta[protocol['DNa02_targets']]/.1;row={'time':(tick+1)*.1,'input_rate_hz':rate,'input_events':int(input_counts.sum()),'DNa02_hz':output.tolist(),'delivered_exc_mV':delivery[0].tolist(),'delivered_inh_mV':delivery[1].tolist(),'mean_v_mV':vv.mean(axis=1).tolist() if vv.size else None,'mean_g_mV':gg.mean(axis=1).tolist() if gg.size else None,'refractory_fraction':(1-gate.mean(axis=1)).tolist() if gate.size else None};rows.append(row)
  file=folder/f'window-{tick:04d}.npz';space_check(folder,ii.nbytes+tt.nbytes+delta.nbytes+vv.nbytes+gg.nbytes+2*1024**2)
  with file.with_suffix('.tmp').open('wb') as f:np.savez_compressed(f,i=ii,t=tt,counts=delta,input_counts=input_counts,v_mV=vv,g_mV=gg,gate=gate,state_t=st,delivered_mV=delivery)
  file.with_suffix('.tmp').replace(file);manifest['chunks'].append({'file':file.name,'sha256':sha(file),'spikes':len(ii),'time':row['time']});atomic_json(folder/'manifest.json',manifest);atomic_json(folder/'bins.json',rows)
 assert np.array_equal(brain.learned_state()['weights'],base_weights) and brain.plasticity.updates==0
 if len(zero):assert np.all(np.asarray(brain.synapses.w[zero]/b.mV)==0)
 atomic_json(folder/'final-state.json',{'DNa02_v_mV':np.asarray(brain.neurons.v[protocol['DNa02_targets']]/b.mV).tolist(),'DNa02_g_mV':np.asarray(brain.neurons.g[protocol['DNa02_targets']]/b.mV).tolist()})
 manifest.update(status='complete',wall_seconds=time.perf_counter()-start,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,plasticity_updates=0);atomic_json(folder/'manifest.json',manifest)
 print(profile,cue,seed,'complete',flush=True)

def launch(root,profile,cue,seed,observer=True):
 command=[sys.executable,str(Path(__file__).resolve()),str(root),'--profile',profile,'--cue',cue,'--seed',str(seed)]
 if not observer:command.append('--observer-off')
 log=root/'logs'/f'{profile}-{cue}-{seed}-{observer}.log';log.parent.mkdir(exist_ok=True)
 with log.open('w') as output:subprocess.run(command,cwd=ROOT,stdout=output,stderr=subprocess.STDOUT,check=True)

def run(root):
 root.mkdir(parents=True,exist_ok=True);protocol=json.loads((root/'protocol.json').read_text()) if (root/'protocol.json').exists() else setup(root)
 assert all(sha(ROOT/name)==digest for name,digest in protocol['sources'].items())
 progress={'status':'running','planned':36,'completed':[]};start=time.perf_counter();atomic_json(root/'progress.json',progress)
 try:
  for profile in PROFILES:
   for cue in ('left','right'):
    for seed in SEEDS:
     name=f'{profile}-{cue}-{seed}';folder=root/'trials'/name
     if (folder/'manifest.json').exists() and json.loads((folder/'manifest.json').read_text())['status']=='complete':progress['completed'].append(name);continue
     if folder.exists():raise RuntimeError('Preserve interrupted attempt before retry: '+str(folder))
     space_check(root,32*1024**2);progress['current']=name;atomic_json(root/'progress.json',progress);launch(root,profile,cue,seed);progress['completed'].append(name);progress['wall_seconds']=time.perf_counter()-start;atomic_json(root/'progress.json',progress);print(len(progress['completed']),'/36',name,flush=True)
     if profile=='intact_narrow' and cue=='left' and seed==4101:
      launch(root,profile,cue,seed,observer=False);a=folder;other=root/'instrumentation-control'/name
      for file in a.glob('window-*.npz'):
       with np.load(file) as x,np.load(other/file.name) as y:
        for key in ('i','t','counts','input_counts'):assert np.array_equal(x[key],y[key]),('Observer changed outcome',key)
      assert json.loads((a/'final-state.json').read_text())==json.loads((other/'final-state.json').read_text())
      atomic_json(root/'instrumentation-check.json',{'passed':True,'scope':'All-neuron exact spikes/times/counts/input counts and matching final DNa02 state, observer on/off.'})
  progress['status']='complete';atomic_json(root/'progress.json',progress)
 except Exception as e:progress.update(status='interrupted',error=repr(e));atomic_json(root/'progress.json',progress);raise
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('--profile',choices=PROFILES);p.add_argument('--cue',choices=['left','right']);p.add_argument('--seed',type=int);p.add_argument('--observer-off',action='store_true');a=p.parse_args()
 if a.profile:worker(a.root,a.profile,a.cue,a.seed,not a.observer_off)
 else:run(a.root)
