import sys,json,time
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from flygarden.body import Body
from flygarden.world import World
root=Path(__file__).resolve().parents[1];runs=[]
for seed in range(5):
 w=World(seed=seed,level=5);w.arena['predator'].update(x=-12.,y=-4.,heading=2.8,speed=5.);body=Body(seed=seed,blocks=w.arena['blocks'],spawn=w.arena['spawn']);seen=False
 for i in range(200):
  s=w.sensory(body.observation());seen|=s['threat']>0;gradient=float(np.sum(s['odor_left'])-np.sum(s['odor_right']));turn=float(np.clip(gradient*8+np.sin(w.time*.7)*.15,-.65,.65));motor=np.array([.85-turn,.85+turn])
  if s['threat']>.25:
   turn=.6 if s['threat_bearing']>0 else -.6;motor=np.array([1+turn,1-turn])
  obs=body.advance(.1,np.clip(motor,0,1.2));w.advance(.1,obs)
  if w.status!='running':break
 runs.append(dict(seed=seed,status=w.status,time=w.time,food=w.collected,escape_events=w.escape_count,threat_exposed=seen,travel=w.travel));body.close();print(runs[-1],flush=True)
r={'status':'completed','runs':runs,'scope':'Five embodied baseline trials with fixed engineered speed5; provisional calibration only, not biological avoidance','captures':sum(x['status']=='captured' for x in runs),'falls':sum(x['status']=='fallen' for x in runs),'survived_20s':sum(x['status']=='running' for x in runs)};(root/'reports/predator-calibration.json').write_text(json.dumps(r,indent=2))
