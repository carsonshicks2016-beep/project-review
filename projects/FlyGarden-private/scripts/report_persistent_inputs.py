"""Descriptive late-tail presynaptic hypotheses; not causal intervention."""
import sys,json,numpy as np,pandas as pd
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from flygarden.recording import atomic_json
from flygarden.continuous_candidate import file_sha
p=ROOT/'reports/brain-integration/recovery/persistence-reference-v1';s=json.loads((p/'protocol.json').read_text());folder=p/'candidate-10101-support0';m=json.loads((folder/'manifest.json').read_text());assert m['status']=='complete';count=np.zeros(138639,dtype=np.int64)
for chunk in m['chunks'][140:]:
 path=folder/chunk['file'];assert file_sha(path)==chunk['sha256']
 with np.load(path) as z:count+=z['counts']
ids=pd.read_csv(ROOT/'vendor/fly-brain/data/2025_Completeness_783.csv',index_col=0).index.to_numpy();ann=pd.read_csv(ROOT/'data/annotations.tsv',sep='\t',low_memory=False).fillna('').set_index('root_id');con=pd.read_parquet(ROOT/'vendor/fly-brain/data/2025_Connectivity_783.parquet',columns=['Presynaptic_Index','Postsynaptic_Index','Excitatory x Connectivity']);records=[]
for pop in ('DM1_lPN_left','DM1_lPN_right','DNa02_left','DNa02_right'):
 targets=[n['index'] for n in s['mapping'][pop]];edges=con[con.Postsynaptic_Index.isin(targets)].copy();edges['nominal_signed_arrival_mV_per_second']=count[edges.Presynaptic_Index.to_numpy()]*edges['Excitatory x Connectivity']*.275/.5
 edges=edges.sort_values('nominal_signed_arrival_mV_per_second',key=lambda v:v.abs(),ascending=False);entries=[]
 for pre,post,signed,arrival in edges.head(12).itertuples(index=False,name=None):
  root=int(ids[int(pre)]);entries.append({'pre_root_id':str(root),'post_root_id':str(ids[int(post)]),'pre_cell_type':str(ann.loc[root,'cell_type']) if root in ann.index else 'unavailable','pre_side':str(ann.loc[root,'side']) if root in ann.index else '', 'signed_synapses':float(signed),'presynaptic_hz':float(count[int(pre)]/.5),'nominal_signed_arrival_mV_per_second':float(arrival)})
 records.append({'target':pop,'top_by_absolute_nominal_arrival':entries})
atomic_json(p/'tail-input-hypotheses.json',{'seed':10101,'support':0,'phase':'3.5–4.0s; odor ended.8s','scope':'Presynaptic spike-count weighted anatomical arrivals. Boundary delays and refractory delivery not corrected; hypotheses only, not actual currents or lesion proof.','source_manifest_sha256':file_sha(folder/'manifest.json'),'auditor_sha256':file_sha(Path(__file__)),'records':records})
print(json.dumps([{'target':r['target'],'top':r['top_by_absolute_nominal_arrival'][:2]} for r in records],indent=2))
