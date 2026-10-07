"""Ten physical arena trials of the explicitly supplied baseline, with playback."""
import sys,json,uuid,time,hashlib
from pathlib import Path
import numpy as np
root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root))
from flygarden.body import Body
from flygarden.world import World
from flygarden.baseline import motor_command
from flygarden.trial_recording import EmbodiedRecording
r={'status':'running','protocol':'10 seeded 60-second bounded physical arenas; baseline v3; odd trials include an obstacle','controller_sha256':hashlib.sha256((root/'flygarden/baseline.py').read_bytes()).hexdigest(),'runs':[],'full_brain_navigation_validated':False}
for seed in range(601,611):
 w=World(seed=seed,level=2 if seed%2 else 1);body=Body(seed=seed,blocks=w.arena['blocks'],spawn=w.arena['spawn'],foods=w.arena['foods'],predator=w.arena['predator']);started=time.perf_counter();frames=0;finite=True;error=None
 recording=EmbodiedRecording(body,w,seed=seed,controller='Supplied local-sensory baseline v3',diagnostic=True,controller_source_sha256=r['controller_sha256']);identity=recording.recorder.id;folder=recording.recorder.folder
 try:
  if True:
   for step in range(600):
    before=body.observation();sense=w.sensory(before);motor=motor_command(sense,w.time);obs=body.advance(.1,motor,capture=recording.capture)
    if max(motor)>.1 and np.linalg.norm(np.array(obs['position'])[:2]-np.array(before['position'])[:2])<.005:w.stalls+=1
    reinforcement=w.advance(.1,obs);body.update_visual_world(w.arena['foods'],w.arena['predator'])
    recording.step(sense,motor,.1,reinforcement=reinforcement);frames=recording.recorder.meta['frames']
    if step%100==99:print(seed,round(w.time),'/60s',flush=True)
    if w.status!='running':break
 except Exception as exc:finite=False;error=str(exc)
 finally:
  recording.finish(dict(status=w.status,food=w.collected,time=w.time,error=error),status='complete' if finite else 'interrupted');body.close()
 run={'seed':seed,'simulated_seconds':w.time,'finite':finite,'error':error,'status':w.status,'food':w.collected,'travel_mm':w.travel,'stalls':w.stalls,'obstacle_layout':seed%2==1,'recording':identity,'samples':frames,'wall_seconds':time.perf_counter()-started};r['runs'].append(run);(root/'reports/baseline-acceptance.json').write_text(json.dumps(r,indent=2));print(run,flush=True)
r['status']='completed';r['finite_60_second_gate']=all(x['finite'] and x['simulated_seconds']>=59.9 for x in r['runs']);r['foraging_gate']=all(x['food']>=1 for x in r['runs']);r['scope']='Supplied baseline only; finite continuation, foraging and full-brain learning are separate gates';(root/'reports/baseline-acceptance.json').write_text(json.dumps(r,indent=2))
