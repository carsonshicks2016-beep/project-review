"""Bounded independent gain sweep; records evidence, never promotes settings."""
import sys,json,time,hashlib
from pathlib import Path
import numpy as np
root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root))
from flygarden.brain import FullBrain
brain=FullBrain(learning=False);brain.mark_initial()
r={'status':'running','seeds':[201,202,203],'gains':[.05,.1,.25,.5,1.],'window_seconds':.2,'runs':[],'promoted':False,'protocol':'Rested paired cue probes with matched RNG; all imported neurons and weights retained'}
start=time.perf_counter()
for gain in r['gains']:
 for seed in r['seeds']:
  codes=[];rates=[]
  for odor in ((gain,0),(0,gain)):
   brain.reset_transient(seed=seed);brain.advance(.2,odor=odor)
   codes.append(brain.last_delta[brain.populations['kcg']].copy());rates.append(brain.last_rates.copy())
  a,b=codes;active_a=a>0;active_b=b>0;union=np.count_nonzero(active_a|active_b)
  cosine=float(np.dot(a.astype(float),b)/(np.linalg.norm(a)*np.linalg.norm(b))) if np.linalg.norm(a)*np.linalg.norm(b)>0 else None
  run={'gain':gain,'seed':seed,'active_a':int(active_a.sum()),'active_b':int(active_b.sum()),'population_size':len(a),'jaccard':float(np.count_nonzero(active_a&active_b)/union) if union else None,'rate_cosine':cosine,'rates':rates}
  r['runs'].append(run);r['wall_seconds']=time.perf_counter()-start;(root/'reports/odor-calibration.json').write_text(json.dumps(r,indent=2));print({k:v for k,v in run.items() if k!='rates'},flush=True)
r['status']='completed';r['source_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest();r['scope']='Sensory calibration diagnostic, not acquisition, choice, or biological validation';(root/'reports/odor-calibration.json').write_text(json.dumps(r,indent=2))
