"""Record annotated population time courses without altering the full model."""
import sys,json,hashlib,time
from pathlib import Path
import numpy as np,pandas as pd
root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root))
from flygarden.brain import FullBrain
brain=FullBrain(learning=False);brain.mark_initial()
ann=pd.read_csv(root/'data/annotations.tsv',sep='\t',low_memory=False).fillna('')
pops={name:np.array([brain.index[int(i)] for i in ann.loc[ann.cell_type.eq(name),'root_id'] if int(i) in brain.index],int) for name in ('ORN_DM1','ORN_DM2','DM1_lPN','DM2_lPN','APL')}
pops['KCg']=brain.populations['kcg']
r={'status':'running','seeds':[301,302,303],'bin_seconds':.02,'window_seconds':.2,'populations':{name:[str(brain.ids[i]) for i in ids] for name,ids in pops.items()},'runs':[],'weights_changed':False,'scope':'Temporal annotated-population diagnostic; no pathway ablation or anatomical causality claim'}
start=time.perf_counter()
for seed in r['seeds']:
 traces=[]
 for cue in ((1,0),(0,1)):
  brain.reset_transient(seed=seed);bins=[]
  for step in range(10):
   brain.advance(.02,odor=cue);bins.append({'time':(step+1)*.02,'populations':{name:{'mean_hz':float(brain.last_delta[ids].mean()/.02) if len(ids) else None,'active_fraction':float(np.mean(brain.last_delta[ids]>0)) if len(ids) else None,'counts':brain.last_delta[ids].tolist()} for name,ids in pops.items()}})
  traces.append({'odor':list(cue),'bins':bins})
 r['runs'].append({'seed':seed,'traces':traces});r['wall_seconds']=time.perf_counter()-start;(root/'reports/olfactory-pathway.json').write_text(json.dumps(r,indent=2));print('Completed seed',seed,flush=True)
r['status']='completed';r['source_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest();(root/'reports/olfactory-pathway.json').write_text(json.dumps(r,indent=2))
