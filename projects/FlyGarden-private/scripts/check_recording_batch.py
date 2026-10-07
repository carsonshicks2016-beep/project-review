"""Short embodied recording acceptance; a normal trial and real physical capture."""
import sys,json,time,fcntl
from pathlib import Path
root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root))
import brian2 as b
from flygarden.brain import FullBrain
from flygarden.body import Body
from flygarden.world import World
from flygarden.trial_recording import EmbodiedRecording
lock=open(root/'.runtime/experiment.lock','a');fcntl.flock(lock,fcntl.LOCK_EX)
brain=FullBrain(seed=801,learning=False);brain.mark_initial();results=[]
for seed,capture in ((801,False),(802,True)):
 brain.reset_transient(seed);w=World(seed=seed)
 if capture:w.arena['predator'].update(enabled=True,x=-22.,y=0.,spawn=[-22.,0.])
 body=Body(seed=seed,blocks=w.arena['blocks'],spawn=w.arena['spawn'],foods=w.arena['foods'],predator=w.arena['predator']);recording=EmbodiedRecording(body,w,brain,seed=seed,controller='Full connectome + supplied gait · short recording acceptance',diagnostic=True);started=time.perf_counter()
 for step in range(30 if not capture else 10):
  before=body.observation();sense=w.sensory(before);brain_start=float(brain.network.t/b.second);motor=brain.advance(.1,odor=sense['odor'],p9=(.65,.65));obs=body.advance(.1,motor,capture=recording.capture);reward=w.advance(.1,obs)
  if w.status=='captured':reward=0.
  brain.reinforce(reward);recording.step(sense,motor,.1,brain_start,reinforcement=reward)
  if w.status!='running':break
 outcome=dict(seed=seed,status=w.status,time=w.time,food=w.collected,diagnostic=True,schedule='3-second recording diagnostic' if not capture else 'Predator starts within capture radius');recording.finish(outcome);results.append(dict(**outcome,run=recording.recorder.id,frames=recording.recorder.meta['frames'],spikes=recording.recorder.meta['spike_count'],wall_seconds=time.perf_counter()-started));body.close();print(results[-1],flush=True)
assert results[0]['frames']==91
assert results[1]['status']=='captured'
(root/'reports/recording-delivery/batch-acceptance.json').write_text(json.dumps(dict(status='passed',full_network_neurons=brain.n,runs=results,scope='Recording/export acceptance only; no navigation or learning validation'),indent=2))
