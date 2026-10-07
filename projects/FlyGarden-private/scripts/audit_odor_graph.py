"""Anatomical lateralization hypotheses, never evidence of effective transmission."""
import sys,json
from pathlib import Path
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from flygarden.recording import atomic_json
from flygarden.continuous_candidate import file_sha
OUT=ROOT/'reports/brain-integration/recovery/odor-bias-diagnosis'
mapping=json.loads((OUT/'diagnostic-protocol.json').read_text())['mapping']
# Read only selected postsynaptic connectivity; all connections remain unchanged.
table=pd.read_parquet(ROOT/'vendor/fly-brain/data/2025_Connectivity_783.parquet',columns=['Presynaptic_Index','Postsynaptic_Index','Excitatory x Connectivity','Connectivity'])
keys=('DM1_lPN_left','DM1_lPN_right','DM2_lPN_left','DM2_lPN_right','DNa02_left','DNa02_right','DNg13_left','DNg13_right')
selected={n['index']:key for key in keys for n in mapping[key]}
local=table[table.Postsynaptic_Index.isin(selected)].copy();del table
local['target_population']=local.Postsynaptic_Index.map(selected)
records=[]
for source in ('ORN_DM1_left','ORN_DM1_right','ORN_DM2_left','ORN_DM2_right','DM1_lPN_left','DM1_lPN_right','DM2_lPN_left','DM2_lPN_right','DNp09_left','DNp09_right'):
 pre={n['index'] for n in mapping[source]}
 for target in keys:
  edges=local[local.Presynaptic_Index.isin(pre)&local.target_population.eq(target)]
  records.append({'source':source,'target':target,'aggregated_records':len(edges),'anatomical_synapses':int(edges.Connectivity.sum()),'positive_signed_count':float(edges['Excitatory x Connectivity'].clip(lower=0).sum()),'negative_signed_count':float(edges['Excitatory x Connectivity'].clip(upper=0).sum())})
ann=pd.read_csv(ROOT/'data/annotations.tsv',sep='\t',low_memory=False).fillna('');ids=pd.read_csv(ROOT/'vendor/fly-brain/data/2025_Completeness_783.csv',index_col=0).index.to_numpy(np.int64);lookup=ann.set_index('root_id')
top={}
for target in ('DNa02_left','DNa02_right'):
 edges=local[local.target_population.eq(target)].copy();edges['magnitude']=edges['Excitatory x Connectivity'].abs();edges=edges.sort_values('magnitude',ascending=False).head(12)
 rows=[]
 for r in edges.itertuples(index=False):
  root=int(ids[r.Presynaptic_Index]);cell=lookup.loc[root].cell_type if root in lookup.index else ''
  rows.append({'root_id':str(root),'cell_type':cell,'signed_anatomical_count':float(getattr(r,'_2')),'anatomical_synapses':int(r.Connectivity)})
 top[target]=rows
atomic_json(OUT/'graph-hypotheses.json',{'status':'complete','records':records,'largest_incoming_magnitude':top,'data_sha256':file_sha(ROOT/'vendor/fly-brain/data/2025_Connectivity_783.parquet'),'auditor_sha256':file_sha(Path(__file__)),'scope':'Anatomical synapse counts and aggregated signed records. Graph links, strongest weights and zero direct links are hypotheses, not functional or unique-pathway evidence.'})
for r in records:
 if r['source'].startswith('ORN_DM1') and r['target'].startswith('DM1'):print(r)
