"""Frozen-recording attribution and complete local-state replay; no brain rerun."""
import json,time,resource
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from flygarden.feedback_trace import delayed_impulses,filtered_impulses,refractory_gate
from flygarden.local_inhibition_adapter import LocalInhibitionAdapter
from flygarden.continuous_candidate import file_sha
from flygarden.recording import atomic_json,space_check
ROOT=Path(__file__).resolve().parents[1];REC=ROOT/'reports/brain-integration/recovery'
PRIOR=REC/'local-inhibition-full-network-v1';OUT=REC/'feedback-trace-v1';ROUTES=REC/'local-inhibition-adapter-v1/routes.json'
PHASES={'pulse_1':(3000,8000),'recovery_1':(25000,30000),'pulse_2':(33000,38000),'recovery_2':(55000,60000)}

def load_events(folder):
 m=json.loads((folder/'manifest.json').read_text());assert m['status']=='complete';ii=[];tt=[];stored=[];states=[]
 for chunk in m['chunks']:
  file=folder/chunk['file'];assert file_sha(file)==chunk['sha256']
  with np.load(file) as z:
   ii.append(z['spike_i'].copy());tt.append(np.rint(z['spike_t']/.0001).astype(int))
   if 'local_activity' in z:stored.append(z['local_filtered_drive'].copy());states.append(z['local_activity'].copy())
 return np.concatenate(ii),np.concatenate(tt),np.array(stored),np.array(states)

def filter_phase_counts(indices,arrivals,lo,hi,n):
 eligible=arrivals<hi;arr=arrivals[eligible];start=np.maximum(arr,lo);length=hi-start
 a=.0001/.005;mass=np.exp(-(start-arr)*a)*(-np.expm1(-length*a))/(-np.expm1(-a))/(hi-lo)
 return np.bincount(indices[eligible],weights=mass,minlength=n)

def main():
 started=time.monotonic();OUT.mkdir(exist_ok=True);space_check(OUT,300*1024**2)
 paths=[ROOT/'flygarden/feedback_trace.py',Path(__file__).resolve(),ROOT/'flygarden/local_inhibition_adapter.py',ROUTES,PRIOR/'protocol.json',PRIOR/'delivery-receipt.json',ROOT/'data/annotations.tsv',ROOT/'vendor/fly-brain/data/2025_Connectivity_783.parquet',ROOT/'vendor/fly-brain/data/2025_Completeness_783.csv']
 protocol={'version':1,'source_hashes':{str(p.relative_to(ROOT)):file_sha(p) for p in paths},'phases_ticks':PHASES,'clock_s':.0001,'delay_ticks':18,'filter_tau_s':.005,
 'gates':{'native_refractory_gate_exact':True,'recorded_filter_max_error':1e-10,'recorded_local_activity_max_error':1e-10},
 'scope':'Descriptive input/output attribution and replay of the failed frozen candidate. Accepted residual spike increments use recorded-spike-derived refractory gates verified against native saved gates. Continuous terms are commanded increments. Neither is a biological current or a causal intervention.',
 'no_parameter_or_controller_change':True,'native_brain_trials':0}
 atomic_json(OUT/'protocol.json',protocol)
 p=json.loads((PRIOR/'protocol.json').read_text());route=json.loads(ROUTES.read_text());ids=pd.read_csv(ROOT/'vendor/fly-brain/data/2025_Completeness_783.csv',index_col=0).index.to_numpy(dtype=np.int64);index={str(v):i for i,v in enumerate(ids)};n=len(ids)
 ann=pd.read_csv(ROOT/'data/annotations.tsv',sep='\t',dtype={'root_id':str},low_memory=False).fillna('').set_index('root_id').reindex([str(x) for x in ids]).fillna('')
 pn=[x['index'] for key in ['DM1_lPN_left','DM1_lPN_right'] for x in p['mapping'][key]];assert len(pn)==2
 # Check helper gate timing against already recorded native before-synapse gates.
 old=REC/'projection-cut-factorial-v1';oldp=json.loads((old/'protocol.json').read_text());oldf=old/'trials/intact-11601';oi,ot,_,_=load_events(oldf);om=json.loads((oldf/'manifest.json').read_text())
 for target in pn:
  gate=refractory_gate(ot[oi==target],40000);col=oldp['selected'].index(target)
  actual=[]
  for chunk in om['chunks']:
   with np.load(oldf/chunk['file']) as z:actual.append(z['gate'][col].copy())
  assert np.array_equal(gate,np.concatenate(actual)),('Native gate mismatch',target)
 atomic_json(OUT/'native-gate-validation.json',{'targets':[{ 'root_id':str(ids[t]),'index':t} for t in pn],'ticks_checked_per_target':40000,'exact':True})
 graph=pd.read_parquet(ROOT/'vendor/fly-brain/data/2025_Connectivity_783.parquet',columns=['Presynaptic_Index','Postsynaptic_Index','Excitatory x Connectivity','Connectivity'])
 pg=graph[graph.Postsynaptic_Index.isin(pn)].copy();del graph
 changes={}
 for row in route['routes']:
  if row['pre_node']>=0 or row['post_node']>=0:
   pair=(index[row['pre_root_id']],index[row['post_root_id']]);changes[pair]=changes.get(pair,0)+row['site_count']
 pg['residual_signed_sites']=[np.sign(s)*(count-changes.get((pre,post),0)) for pre,post,s,count in zip(pg.Presynaptic_Index,pg.Postsynaptic_Index,pg['Excitatory x Connectivity'],pg.Connectivity)]
 inputs=[row for row in route['routes'] if row['pre_node']<0 and row['post_node']>=0];tot=np.array(route['input_site_totals']);weights=csr_matrix(([row['original_sign']*row['site_count']/tot[row['post_node']]/100/.005 for row in inputs],([row['post_node'] for row in inputs],[index[row['pre_root_id']] for row in inputs])),shape=(len(tot),n))
 dm1_nodes=[i for i,key in enumerate(route['nodes']) if key.endswith('@left:DM1') or key.endswith('@right:DM1')];dm1_coeff=np.asarray(weights[dm1_nodes].mean(axis=0)).ravel()
 coupling=np.array(route['coupling']);outweights=np.zeros((2,len(tot)))
 for row in route['routes']:
  if row['pre_node']>=0 and row['post_node']<0 and index[row['post_root_id']] in pn:outweights[pn.index(index[row['post_root_id']]),row['pre_node']]+=row['site_count']
 jobs=[f'{m}-{cue}-{seed}' for seed in [11801,11802] for cue in ['none','a_left','a_right','a_both'] for m in ['original','local']]
 summaries=[];errors=[]
 for number,name in enumerate(jobs):
  ii,ticks,stored,stored_state=load_events(PRIOR/'trials'/name);arrivals=ticks+18;model=name.split('-')[0];release=None
  if model=='local':
   impulses=delayed_impulses(ii,ticks,weights,60000,18);filtered=filtered_impulses(impulses);del impulses
   err=float(np.max(np.abs(filtered[249::250]-stored)));assert err<=1e-10,('Filter replay mismatch',name,err)
   adapter=LocalInhibitionAdapter(route['nodes'],np.zeros_like(coupling));buffer=np.zeros((18,len(tot)));cursor=0;release=np.zeros((60000,len(tot)));activity_error=0.
   for tick in range(60000):
    delayed=buffer[cursor].copy();adapter.step(filtered[tick]+coupling@delayed);release[tick]=delayed*100;buffer[cursor]=adapter.activity;cursor=(cursor+1)%18
    if tick%250==249:activity_error=max(activity_error,float(np.max(np.abs(adapter.activity-stored_state[tick//250]))))
   assert activity_error<=1e-10,('Activity replay mismatch',name,activity_error);errors.append({'trial':name,'filter_max_error':err,'activity_max_error':activity_error})
   # Coarse state curves for visual review, reconstructed at the native clock.
   np.savez_compressed(OUT/(name+'-trace.npz'),time_s=np.arange(1,241)*.025,dm1_filter=filtered[249::250][:,dm1_nodes].mean(axis=1),dm1_activity=stored_state[:,dm1_nodes].mean(axis=1)*100)
  gates=[refractory_gate(ticks[ii==target],60000) for target in pn]
  for phase,(lo,hi) in PHASES.items():
   target_contributions=[]
   for j,target in enumerate(pn):
    rows=pg[pg.Postsynaptic_Index==target];signed=rows['residual_signed_sites' if model=='local' else 'Excitatory x Connectivity'].to_numpy()*.275
    eligible=(arrivals>=lo)&(arrivals<hi);sources=ii[eligible];arr=arrivals[eligible];accepted=gates[j][arr];counts=np.bincount(sources[accepted],minlength=n)
    values=counts[rows.Presynaptic_Index.to_numpy()]*signed/((hi-lo)*.0001);groups={};ranking=[]
    for pre,value in zip(rows.Presynaptic_Index,values):
     if value==0:continue
     label=str(ann.iloc[pre].cell_class) or 'unavailable';group=groups.setdefault(label,{'positive_mV_per_s':0.,'negative_mV_per_s':0.});group['positive_mV_per_s' if value>0 else 'negative_mV_per_s']+=abs(float(value))
     ranking.append({'root_id':str(ids[pre]),'index':int(pre),'cell_type':str(ann.iloc[pre].cell_type),'cell_class':label,'signed_accepted_mV_per_s':float(value)})
    target_contributions.append({'target_root_id':str(ids[target]),'target_index':target,'positive_mV_per_s':float(values[values>0].sum()),'negative_mV_per_s':float(-values[values<0].sum()),'by_class':groups,'top_positive_sources':sorted([x for x in ranking if x['signed_accepted_mV_per_s']>0],key=lambda x:-x['signed_accepted_mV_per_s'])[:15],
     'commanded_local_negative_mV_per_s':float((.275*(release[lo:hi]@outweights[j])).mean()) if release is not None else 0.})
   local_groups={}
   if model=='local':
    mass=filter_phase_counts(ii,arrivals,lo,hi,n);contributions=mass*dm1_coeff
    for source in np.flatnonzero(contributions):
     label=str(ann.iloc[source].cell_class) or 'unavailable';group=local_groups.setdefault(label,{'positive_filtered_drive':0.,'negative_filtered_drive':0.});v=contributions[source];group['positive_filtered_drive' if v>0 else 'negative_filtered_drive']+=abs(float(v))
    reconstructed=sum(x['positive_filtered_drive']-x['negative_filtered_drive'] for x in local_groups.values());assert abs(reconstructed-filtered[lo:hi][:,dm1_nodes].mean())<1e-10
   summaries.append({'trial':name,'phase':phase,'PN_inputs':target_contributions,'DM1_local_filter_sources_by_class':local_groups})
  if model=='local':del filtered,release
  atomic_json(OUT/'progress.json',{'status':'running','completed':number+1,'planned':len(jobs),'current':name});print(name,'traced',flush=True)
 result={'status':'complete','trials':len(jobs),'summaries':summaries,'replay_checks':errors,'native_gate_exact':True,'native_brain_runs':0,'parameter_changes':0,'controller_changed':False,'wall_seconds':time.monotonic()-started,'peak_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'sources_reverified':all(file_sha(ROOT/k)==v for k,v in protocol['source_hashes'].items())}
 atomic_json(OUT/'results.json',result);atomic_json(OUT/'progress.json',{'status':'complete','completed':len(jobs),'planned':len(jobs)});print('Trace complete',round(result['wall_seconds'],1),'seconds',flush=True)
if __name__=='__main__':main()
