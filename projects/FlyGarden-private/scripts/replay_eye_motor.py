"""Open-loop FlyGym replay of measured visual-run descending commands."""
import json,sys,time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from flygarden.body import Body
from flygarden.recording import atomic_json,space_check
from scripts.audit_brain_reference import sha
OUT=ROOT/'reports/brain-integration/stage6-20261006'
def replay():
 assert json.loads((OUT/'progress.json').read_text())['status']=='complete'
 out=OUT/'body-replay';out.mkdir(exist_ok=False);results=[]
 for cue in ('blank','loom_left','loom_right'):
  rows=json.loads((OUT/'trials'/f'{cue}-6199'/'bins.json').read_text());body=Body(seed=6199);initial=body.observation();trace=[];frames=[];prev=np.array(initial['position']);travel=0.;start=time.perf_counter()
  try:
   for row in rows:
    obs=body.advance(.1,row['candidate_motor'],capture=lambda t,p:frames.append({'time':t,**p}));pos=np.array(obs['position']);travel+=float(np.linalg.norm(pos[:2]-prev[:2]));prev=pos.copy();trace.append({'time':row['time'],'motor':row['candidate_motor'],'body':obs})
  finally:body.close()
  heading=np.unwrap([initial['heading']]+[t['body']['heading'] for t in trace]);r={'cue':cue,'seed':6199,'heading_change_radians':float(heading[-1]-heading[0]),'travel_mm':travel,'flipped_samples':sum(t['body']['flipped'] for t in trace),'finite':all(np.isfinite(t['body']['position']).all() for t in trace),'frames':len(frames),'wall_seconds':time.perf_counter()-start}
  space_check(out,4*1024**2);atomic_json(out/f'{cue}.json',{'result':r,'initial':initial,'trace':trace,'frames':frames});results.append(r);print(r,flush=True)
 atomic_json(out/'summary.json',{'status':'complete','scope':'Open-loop replay of measured neural motors. Eye inputs were recorded in a stationary pose; replay is not closed-loop navigation or biological escape. Small passive body motion is reported as motion, not success.','results':results,'sources':{s:sha(ROOT/s) for s in ('flygarden/body.py','scripts/replay_eye_motor.py')}})
if __name__=='__main__':replay()
