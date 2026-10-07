"""Compile a count-preserving hybrid routing proposal, never mutate brain."""
import hashlib,json,time
from pathlib import Path
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
P=ROOT/'reports/brain-integration/recovery'
OUT=P/'local-inhibition-adapter-v1'
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
 return h.hexdigest()
def main():
 start=time.monotonic(); OUT.mkdir(exist_ok=True)
 bindings=P/'glomerular-affine-registration-v1/partial-anatomical-bindings.json'
 endpoints=P/'glomerular-affine-registration-v1/partial-bilateral-endpoints.parquet'
 graph=ROOT/'vendor/fly-brain/data/2025_Connectivity_783.parquet'
 comp=ROOT/'vendor/fly-brain/data/2025_Completeness_783.csv'
 sources=[bindings,endpoints,graph,comp,ROOT/'flygarden/local_inhibition_adapter.py',Path(__file__).resolve()]
 protocol={'version':1,'source_hashes':{str(p.relative_to(ROOT)):sha(p) for p in sources},
 'subtype':'lLN2P_b only: qualified GABA/MIP annotation bridge',
 'parameter_status':'Engineering hypothesis, not a fit to these cells',
 'tau_s':.015,'tau_provenance':'Borrowed as an explicit numerical hypothesis from LN2P_c/DC3 reference; no transfer claim',
 'rate_scale_hz':100.,'rate_scale_provenance':'Engineering normalization; not measured release',
 'dt_s':.0001,'local_activity':'Dimensionless, clipped 0..1',
 'incoming_signs':'Frozen graph for spiking sources; inhibitory for upgraded local sources',
 'output':'-.275 mV/site/spike times normalized release Hz, average mV/s proxy only',
 'coupling':'Only actual synapses joining eligible local nodes; no inferred cable coupling',
 'fallback':'Unassigned sites retain original spiking routing and original signs',
 'limitations':['Partial hybrid neuron rather than validated cable model','MIP, receptor dynamics and electrical propagation excluded','No learning or body integration','Full-network delay-aware injection not implemented'],
 'application_changed':False}
 (OUT/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
 b=json.loads(bindings.read_text())['bindings']; chosen={x['root_id']:x for x in b if x['cell_type']=='lLN2P_b'}
 assert len(chosen)==12
 e=pd.read_parquet(endpoints); e=e[e.root_id.isin(chosen)].copy()
 # One node per exact root and eligible region, requiring both input and output.
 counts=e[e.region_evidence_eligible].groupby(['root_id','bilateral_mesh_candidate','direction']).size().unstack(fill_value=0)
 viable=counts[(counts['input']>0)&(counts['output']>0)]
 nodes=[str(root)+'@'+str(region) for root,region in viable.index]; index={key:i for i,key in enumerate(nodes)}
 e['node']=[index.get(str(r)+'@'+str(g),-1) if ok else -1 for r,g,ok in zip(e.root_id,e.bilateral_mesh_candidate,e.region_evidence_eligible)]
 inp=e[e.direction=='input'][['synapse_id','node']].rename(columns={'node':'post_node'})
 out=e[e.direction=='output'][['synapse_id','node']].rename(columns={'node':'pre_node'})
 sites=e[['synapse_id','pre_root_id','post_root_id']].drop_duplicates('synapse_id').merge(inp,how='left',on='synapse_id',validate='one_to_one').merge(out,how='left',on='synapse_id',validate='one_to_one')
 sites[['pre_node','post_node']]=sites[['pre_node','post_node']].fillna(-1).astype(int)
 grouped=sites.groupby(['pre_root_id','post_root_id','pre_node','post_node']).size().rename('site_count').reset_index()
 ids=pd.read_csv(comp,index_col=0).index.to_numpy(dtype=np.int64)
 selected_indices=np.array([x['model_index'] for x in chosen.values()]);con=pd.read_parquet(graph,columns=['Presynaptic_Index','Postsynaptic_Index','Connectivity','Excitatory x Connectivity'])
 con=con[con.Presynaptic_Index.isin(selected_indices)|con.Postsynaptic_Index.isin(selected_indices)].copy()
 con['pre_root_id']=[str(x) for x in ids[con.Presynaptic_Index.to_numpy()]];con['post_root_id']=[str(x) for x in ids[con.Postsynaptic_Index.to_numpy()]]
 sums=grouped.groupby(['pre_root_id','post_root_id']).site_count.sum().to_dict()
 expected=con.set_index(['pre_root_id','post_root_id']).Connectivity.to_dict()
 assert sums==expected,'Routing changes original pair counts'
 con['original_sign']=np.sign(con['Excitatory x Connectivity']).astype(int)
 assert (con['Excitatory x Connectivity'].abs()==con.Connectivity).all(),'Unexpected non-sign graph weights'
 grouped=grouped.merge(con[['pre_root_id','post_root_id','original_sign']],on=['pre_root_id','post_root_id'],validate='many_to_one')
 routes=grouped.to_dict('records');totals=np.zeros(len(nodes),dtype=int);coupling=np.zeros((len(nodes),len(nodes)))
 for row in routes:
  if row['post_node']>=0:totals[row['post_node']]+=row['site_count']
 assert np.all(totals>0)
 for row in routes:
  if row['post_node']>=0 and row['pre_node']>=0:
   coupling[row['post_node'],row['pre_node']]-=row['site_count']/totals[row['post_node']]
 payload={'version':1,'nodes':nodes,'root_model_indices':{r:v['model_index'] for r,v in chosen.items()},'input_site_totals':totals.tolist(),'coupling':coupling.tolist(),'routes':routes,'all_pair_counts_preserved':True,'model_ready':False}
 (OUT/'routes.json').write_text(json.dumps(payload,separators=(',',':'))+'\n')
 result={'nodes':len(nodes),'selected_roots':12,'original_pair_count':len(expected),'original_sites':int(sum(expected.values())),'routing_groups':len(routes),'upgraded_output_sites':sum(r['site_count'] for r in routes if r['pre_node']>=0),'upgraded_input_sites':sum(r['site_count'] for r in routes if r['post_node']>=0),'local_to_local_sites':sum(r['site_count'] for r in routes if r['post_node']>=0 and r['pre_node']>=0),'unchanged_spiking_sites':sum(r['site_count'] for r in routes if r['post_node']<0 and r['pre_node']<0),'all_pair_counts_preserved':True,'application_changed':False,'wall_seconds':time.monotonic()-start,'sources_reverified':all(sha(ROOT/k)==v for k,v in protocol['source_hashes'].items())}
 (OUT/'routing-results.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
