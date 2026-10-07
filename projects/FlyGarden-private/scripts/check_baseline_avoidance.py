"""Independent physical baseline wall-avoidance trials; no brain claim."""
import sys,json
from pathlib import Path
import numpy as np
root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root))
from flygarden.body import Body
from flygarden.world import World
from flygarden.baseline import motor_command
r={'status':'running','controller':'Supplied local-sensory baseline v3','runs':[],'full_brain_navigation_validated':False}
for seed in (501,502,503):
 w=World(seed=seed);w.arena['blocks']=[dict(x=8.,y=0.,w=2.,h=12.,z=3.)];w.arena['foods']=[dict(id='goal',x=20.,y=0.,odor=0,units=3)]
 body=Body(seed=seed,blocks=w.arena['blocks']);frames=[]
 try:
  for i in range(100):
   obs=body.observation();s=w.sensory(obs);motor=motor_command(s,i*.1);obs=body.advance(.1,motor);frames.append({'time':(i+1)*.1,'position':obs['position'],'flipped':obs['flipped'],'ranges':s['obstacle_ranges'],'motor':motor.tolist()})
  run={'seed':seed,'finite':True,'fall_frames':sum(x['flipped'] for x in frames),'passed_wall':any(x['position'][0]>10. for x in frames),'frames':frames}
 finally:body.close()
 r['runs'].append(run);(root/'reports/baseline-avoidance.json').write_text(json.dumps(r,indent=2));print({k:v for k,v in run.items() if k!='frames'},flush=True)
r['status']='completed';r['passed']=all(x['finite'] and x['fall_frames']==0 and x['passed_wall'] for x in r['runs']);r['scope']='Three short supplied-baseline trials only; no full-brain navigation or biological sensory validation';(root/'reports/baseline-avoidance.json').write_text(json.dumps(r,indent=2))
