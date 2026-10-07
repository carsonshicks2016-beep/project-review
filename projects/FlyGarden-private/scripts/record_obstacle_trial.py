"""30-second full-network recording diagnostic; fixed controller and frozen plasticity."""
import sys,json,time,hashlib,resource
from pathlib import Path
root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root))
import numpy as np
import brian2 as b
from flygarden.brain import FullBrain
from flygarden.body import Body
from flygarden.world import World
from flygarden.trial_recording import EmbodiedRecording
from flygarden.recording import atomic_json,space_check
seed=801;started=time.perf_counter();brain=FullBrain(seed=seed,learning=False);brain.mark_initial();world=World(seed=seed,level=2)
body=Body(seed=seed,blocks=world.arena['blocks'],spawn=world.arena['spawn'],foods=world.arena['foods'],predator=world.arena['predator'])
rec=EmbodiedRecording(body,world,brain,seed=seed,controller='Full connectome + supplied gait · 30-second obstacle diagnostic',diagnostic=True,protocol='recording30',planned_duration=30.,controller_source_sha256=hashlib.sha256((root/'flygarden/brain.py').read_bytes()).hexdigest())
report=root/'reports/recording-delivery/obstacle-trial.json';reward=0.;total_spikes=0;finite=True
try:
 for step in range(300):
  space_check(rec.recorder.folder,16*1024**2);before=body.observation();sense=world.sensory(before);brain_start=float(brain.network.t/b.second)
  motor=brain.advance(.1,odor=sense['odor'],p9=(.65,.65),loom=[0.,0.],reinforcement=reward);motor=np.clip(motor,0,1.2)
  body.update_visual_world(world.arena['foods'],world.arena['predator']);obs=body.advance(.1,motor,capture=rec.capture)
  if max(motor)>.1 and np.linalg.norm(np.asarray(obs['position'])[:2]-np.asarray(before['position'])[:2])<.005:world.stalls+=1
  finite=finite and bool(np.isfinite(body.sim.mj_data.qpos).all()) and bool(np.isfinite(body.sim.mj_data.qvel).all())
  reward=world.advance(.1,obs);brain.reinforce(reward);rec.step(sense,motor,.1,brain_start,reinforcement=reward);total_spikes+=len(brain.last_spikes[0])
  if (step+1)%10==0:
   progress=dict(status='running',run=rec.recorder.id,time=world.time,planned_duration=30.,food=world.collected,world_status=world.status,frames=rec.recorder.meta['frames'],spikes=total_spikes,wall_seconds=time.perf_counter()-started)
   atomic_json(report,progress);print(json.dumps(progress),flush=True)
  if world.status!='running' or not finite:break
 outcome=dict(status=world.status,time=world.time,food=world.collected,travel_mm=world.travel,stalls=world.stalls,termination='scheduled 30-second diagnostic boundary' if step==299 else 'physical/game termination',diagnostic=True)
 rec.finish(outcome)
 result=dict(status='completed',run=rec.recorder.id,seed=seed,full_network_neurons=brain.n,planned_duration=30.,outcome=outcome,frames=rec.recorder.meta['frames'],spikes=total_spikes,monitor_spikes=int(brain.monitor.count[:].sum()),finite_physics=finite,recording_counts_match=total_spikes==int(brain.monitor.count[:].sum()),wall_seconds=time.perf_counter()-started,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,recording_bytes=sum(p.stat().st_size for p in rec.recorder.folder.rglob('*') if p.is_file()),scope='One frozen full-brain diagnostic; no learning or navigation-success claim')
 atomic_json(report,result);print(json.dumps(result),flush=True)
except BaseException as exc:
 rec.finish(dict(status='interrupted',time=world.time,error=str(exc)),status='interrupted');raise
finally:body.close()
