import json,time,resource,sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from flygarden.body import Body
from flygarden.trial_recording import EmbodiedRecording
root=Path(__file__).resolve().parents[1]
report=dict(status='running',protocol='10 independent seeds, 60 simulated seconds; walking, turning, stopping, collision exposure',runs=[],physics_dt=Body.dt,walking_control_dt=Body.control_dt)
start=time.perf_counter()
for seed in range(10):
 print(f'Building seed {seed}',flush=True)
 blocks=[dict(x=32.,y=0.,w=2.,h=10.,z=3.)] if seed%2 else []
 body=Body(seed=seed,blocks=blocks);initial=np.array(body.observation()['position']);falls=0;stalls=0;path=[];finite=True;t=time.perf_counter()
 try:
  recording=EmbodiedRecording(body,seed=seed,arena=dict(blocks=blocks,foods=[],predator=dict(enabled=False),spawn=[0,0]),controller='Supplied gait body acceptance',diagnostic=True)
  for frame in range(600):
   now=frame*.1;phase=int(now//10)%6
   motor=([1,1],[1,.45],[0,0],[.45,1],[1,1],[.6,.6])[phase]
   obs=body.advance(.1,motor,capture=recording.capture);recording.step({},motor,.1);path.append(obs['position']);falls+=int(obs['flipped']);stalls+=int(frame>0 and np.linalg.norm(np.asarray(path[-1])-path[-2])<.005)
   if frame%100==99:print(f'Seed {seed}: {(frame+1)/10:.0f}/60s',flush=True)
  run=dict(seed=seed,status='completed',simulated_seconds=60,wall_seconds=time.perf_counter()-t,finite=True,fall_frames=falls,stall_frames=stalls,displacement=float(np.linalg.norm(np.asarray(path[-1])-initial)),collision_arena=bool(blocks))
 except Exception as exc:run=dict(seed=seed,status='failed',error=str(exc),finite=False,simulated_seconds=len(path)*.1)
 finally:
  if 'recording' in locals():recording.finish(run,status='complete' if run['finite'] else 'interrupted')
  body.close()
 report['runs'].append(run);report['wall_seconds']=time.perf_counter()-start
 (root/'reports/body-acceptance.json').write_text(json.dumps(report,indent=2));print(json.dumps(run),flush=True)
report['status']='completed';report['passed']=len(report['runs'])==10 and all(x['finite'] and x['simulated_seconds']==60 for x in report['runs']);report['peak_rss_bytes']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
(root/'reports/body-acceptance.json').write_text(json.dumps(report,indent=2));print('Body acceptance finished',report['passed'],flush=True)
