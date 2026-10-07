"""Leave a genuine short body recording unfinished to verify cold-start recovery."""
import sys,json
from pathlib import Path
root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root))
from flygarden.body import Body
from flygarden.world import World
from flygarden.trial_recording import EmbodiedRecording
body=Body(seed=890);world=World(seed=890);recording=EmbodiedRecording(body,world,seed=890,controller='Supplied gait · interruption recovery diagnostic',diagnostic_fixture=True)
sense=world.sensory(body.observation());motor=[.7,.9];obs=body.advance(.1,motor,capture=recording.capture);world.advance(.1,obs);recording.step(sense,motor,.1);body.close()
(root/'reports/recording-delivery/recovery-fixture.json').write_text(json.dumps(dict(run=recording.recorder.id,frames=recording.recorder.meta['frames'],writer_pid=recording.recorder.manifest['writer']['pid']),indent=2))
# Deliberately do not finalize: the dead writer is detected at application startup.
