"""Anatomical reachability; not a claim of functional or excitatory transmission."""
import sys,json
from pathlib import Path
import numpy as np,pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.diagnose_sensorimotor_pathway import mapping
from scripts.audit_brain_reference import sha
from flygarden.recording import atomic_json

def run(root):
 ids,pops,ann=mapping();n=len(ids);lookup=ann.set_index('root_id').to_dict('index')
 path=ROOT/'vendor/fly-brain/data/2025_Connectivity_783.parquet';con=pd.read_parquet(path,columns=['Presynaptic_Index','Postsynaptic_Index','Connectivity','Excitatory x Connectivity'])
 pre=con.Presynaptic_Index.to_numpy(dtype=np.int32);post=con.Postsynaptic_Index.to_numpy(dtype=np.int32);signed=con['Excitatory x Connectivity'].to_numpy();anatomical=con.Connectivity.to_numpy()
 graph=csr_matrix((np.ones(len(pre),dtype=np.int8),(pre,post)),shape=(n,n));graph.sort_indices()
 def detail(index):
  a=lookup.get(int(ids[index]),{});return {'root_id':str(ids[index]),'index':int(index),'cell_type':a.get('cell_type',''),'cell_class':a.get('cell_class',''),'side':a.get('side',''),'annotation_available':bool(a)}
 routes=[]
 for cue in ('ORN_DM1_left','ORN_DM1_right','ORN_DM2_left','ORN_DM2_right'):
  source=np.array([r['index'] for r in pops[cue]],dtype=np.int32)
  distance,pred,origins=dijkstra(graph,directed=True,indices=source,min_only=True,unweighted=True,return_predecessors=True)
  for cell in ('DNa02','DNp09','DNg13'):
   for side in ('left','right'):
    for target in pops[f'{cell}_{side}']:
     idx=target['index'];chain=[]
     if np.isfinite(distance[idx]):
      at=idx
      while at>=0:
       chain.append(at);at=int(pred[at])
      chain.reverse()
     edges=[]
     for a,b in zip(chain,chain[1:]):
      match=(pre==a)&(post==b);edges.append({'from_root_id':str(ids[a]),'to_root_id':str(ids[b]),'records':int(match.sum()),'anatomical_synapses':int(anatomical[match].sum()),'signed_model_count':float(signed[match].sum())})
     routes.append({'source_population':cue,'target':target,'reachable':bool(chain),'minimum_hops':int(distance[idx]) if chain else None,'representative_shortest_path':[detail(i) for i in chain],'edges':edges})
 result={'schema_version':1,'neurons':n,'connection_records':len(pre),'annotation_sha256':sha(ROOT/'data/annotations.tsv'),'connectivity_sha256':sha(path),'populations':pops,'routes':routes,'limitations':['All imported records are used for directed topology, including inhibitory/zero signed weights.','One representative shortest path is shown, not a unique pathway.','Anatomical reachability does not establish functional transmission or the sign of a multihop response.','No edges in the brain are altered or thresholded.']}
 atomic_json(root/'anatomical-routes.json',result);print('Mapped',len(routes),'routes')
if __name__=='__main__':run(Path(sys.argv[1]))
