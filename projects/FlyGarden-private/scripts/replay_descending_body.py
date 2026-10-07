"""Replay measured candidate neural motor output into actual FlyGym body.

Run only after the isolated brain worker completes. This is open-loop motor
replay, not closed-loop sensory navigation. Direct controls are labelled.
"""
import json,sys,time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from flygarden.body import Body
from flygarden.recording import atomic_json,space_check
from scripts.audit_brain_reference import sha

def run(root):
 if json.loads((root/'progress.json').read_text())['status']!='complete':raise RuntimeError('Finish neural simulations first')
 out=root/'body-replay';out.mkdir(exist_ok=False);results=[];start=time.perf_counter()
 for cue in ('walking','walking_a_left','walking_a_right','direct_left','direct_right'):
  rows=json.loads((root/'trials'/f'{cue}-3199'/'bins.json').read_text());body=Body(seed=3199);initial=body.observation();frames=[];trace=[];distance=0;prev=np.array(initial['position'])
  def capture(t,pose):frames.append({'time':t,**pose})
  try:
   for row in rows:
    obs=body.advance(.1,row['candidate_motor'],capture=capture);pos=np.array(obs['position']);distance+=np.linalg.norm(pos[:2]-prev[:2]);prev=pos
    trace.append({'time':row['time'],'motor':row['candidate_motor'],'body':obs})
  finally:body.close()
  headings=np.unwrap([initial['heading']]+[r['body']['heading'] for r in trace]);result={'cue':cue,'seed':3199,'artificial_direct_output_control':cue.startswith('direct'),'initial':initial,'final':trace[-1]['body'],'heading_change_radians':float(headings[-1]-headings[0]),'travel_distance':float(distance),'flipped_samples':sum(r['body']['flipped'] for r in trace),'finite':all(np.isfinite(r['body']['position']).all() for r in trace),'frames':len(frames)}
  space_check(out,4*1024**2);atomic_json(out/f'{cue}.json',{'result':result,'trace':trace,'frames':frames});results.append(result);print(cue,result['heading_change_radians'],flush=True)
 atomic_json(out/'summary.json',{'status':'complete','results':results,'wall_seconds':time.perf_counter()-start,'source_sha256':sha(Path(__file__)),'body_source_sha256':sha(ROOT/'flygarden/body.py'),'scope':'Recorded neural motors replayed into supplied gait and FlyGym body; open-loop, no sensory feedback or navigation claim.'})
if __name__=='__main__':run(Path(sys.argv[1]))
