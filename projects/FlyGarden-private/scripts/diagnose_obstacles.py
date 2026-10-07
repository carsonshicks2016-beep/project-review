"""Independent physical exposure diagnostic; no steering or pose correction."""
import sys,json,time
from pathlib import Path
import numpy as np
root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root))
from flygarden.body import Body
report={'status':'running','protocol':'Fixed straight supplied gait, physical wall versus shallow step; independent seeds 101/102; 10 seconds each','runs':[],'controller_changed':False}
for height in (.15,3.):
 for seed in (101,102):
  body=Body(seed=seed,blocks=[dict(x=8.,y=0.,w=2.,h=12.,z=height)])
  run={'height_mm':height,'seed':seed,'first_contact_seconds':None,'first_flip_seconds':None,'frames':[]}
  try:
   for i in range(100):
    obs=body.advance(.1,[1.,1.]);d=body.sim.mj_data;m=body.sim.mj_model
    contacts=0
    for contact in d.contact:
     names=[m.geom(int(g)).name for g in contact.geom]
     contacts+=int(any('obstacle_0' in name for name in names))
    if contacts and run['first_contact_seconds'] is None:run['first_contact_seconds']=(i+1)*.1
    if obs['flipped'] and run['first_flip_seconds'] is None:run['first_flip_seconds']=(i+1)*.1
    run['frames'].append({'time':(i+1)*.1,'position':obs['position'],'flipped':obs['flipped'],'obstacle_contacts':contacts})
   run['finite']=True;run['passed_obstacle']=max(x['position'][0] for x in run['frames'])>10.
  finally:body.close()
  report['runs'].append(run);(root/'reports/obstacle-diagnostic.json').write_text(json.dumps(report,indent=2));print({k:v for k,v in run.items() if k!='frames'},flush=True)
report['status']='completed';report['scope']='Physical exposure diagnostic only; does not establish navigation or arbitrary terrain stability'
(root/'reports/obstacle-diagnostic.json').write_text(json.dumps(report,indent=2))
