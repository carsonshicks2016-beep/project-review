"""Recording adapters for existing experiment schedules, with no control changes."""
import copy,hashlib,json,os
from .recording import Recorder,atomic_json
from .storage import ROOT

def manifest(seed,mode,arena,controller,learning=False,**extra):
    return dict(seed=seed,mode=mode,arena=copy.deepcopy(arena),controller=dict(name=controller),learning=learning,lineage='recorded-experiment',parent=None,checkpoint_links=[],provenance=json.loads((ROOT/'reports/provenance.json').read_text()),**extra)

class EmbodiedRecording:
    def __init__(self,body,world=None,brain=None,seed=0,arena=None,controller='Supplied gait diagnostic',**extra):
        self.body=body;self.world=world;self.brain=brain;self.base_time=body.steps*body.dt;self.previous_time=world.time if world else 0.;self.poses=[]
        arena=arena or (world.arena if world else dict(blocks=[],foods=[],predator=dict(enabled=False),spawn=[0,0]))
        self.recorder=Recorder(manifest(seed,'full' if brain else 'baseline',arena,controller,brain.plasticity.enabled if brain else False,body_recorded=True,**extra),body.geometry(),brain.ids if brain else None)
        self.last_state=self.state();self.recorder.append_window([self.last_state])
    def state(self,sense=None,motor=None):
        w=self.world
        values={k:getattr(w,k) for k in ('time','energy','collected','status','mode','level','rewarded_odor','valid','travel','escape_count','stalls')} if w else dict(time=self.previous_time,energy=None,collected=None,status='running',mode='body-acceptance',level=1)
        return dict(ready=True,world=values,arena=copy.deepcopy(w.arena if w else self.recorder.manifest['arena']),body=self.body.observation(),visuals=self.body.visuals(),sensory=sense or {},motor=list(motor) if motor is not None else [],rates=copy.deepcopy(self.brain.last_rates) if self.brain else {},learning=self.brain.plasticity.enabled if self.brain else False,events=copy.deepcopy(w.events[-30:]) if w else [],run_id=self.recorder.id)
    def capture(self,stamp,pose):self.poses.append((stamp-self.base_time,pose))
    def step(self,sense,motor,dt,brain_start=None,reinforcement=0.):
        end=self.world.time if self.world else self.previous_time+dt;final=self.state(sense,motor);final['world']['time']=end;final['reinforcement']=float(reinforcement);frames=[]
        for stamp,pose in self.poses:
            state=copy.deepcopy(final if stamp>=end-1e-6 else self.last_state);state.update(pose);state['world']['time']=stamp;state.update(sensory=sense,sensory_time=self.previous_time,motor=list(motor));frames.append(state)
        indices,times=(self.brain.last_spikes if self.brain else ([],[]))
        if self.brain is not None:times=times-brain_start+self.previous_time
        self.recorder.append_window(frames,indices,times)
        if reinforcement:
            with (self.recorder.folder/'events.jsonl').open('a') as f:
                f.write(json.dumps(dict(time=end,kind='reinforcement',value=float(reinforcement),message='Positive modeled reinforcement' if reinforcement>0 else 'Aversive modeled reinforcement'))+'\n');f.flush();os.fsync(f.fileno())
        self.poses=[];self.previous_time=end;self.last_state=final
    def finish(self,outcome,status='complete'):self.recorder.finish(outcome,status)

class NeuralRecording:
    def __init__(self,brain,seed,condition):
        self.brain=brain;self.elapsed=0.;self.events=[];arena=dict(blocks=[],foods=[],predator=dict(enabled=False),spawn=[0,0])
        self.recorder=Recorder(manifest(seed,'full',arena,'Full connectome conditioning screen · no physical body',brain.plasticity.enabled,body_recorded=False,condition=condition),dict(meshes={},geoms=[]),brain.ids)
        self.recorder.manifest.update(pose_fps=0,pose_scope='Neural conditioning only; no physical body simulated',controller_source_sha256=hashlib.sha256((ROOT/'flygarden/brain.py').read_bytes()).hexdigest());atomic_json(self.recorder.folder/'manifest.json',self.recorder.manifest)
    def advance(self,dt,**kwargs):
        import brian2 as b
        begin=float(self.brain.network.t/b.second);result=self.brain.advance(dt,**kwargs);indices,times=self.brain.last_spikes;times=times-begin+self.elapsed
        frames=[]
        for k in range(round(dt*30)+1):
            stamp=self.elapsed+k/30
            if k==0 and self.elapsed>0:continue
            frames.append(dict(ready=True,world=dict(time=stamp,energy=None,collected=None,status='neural conditioning',mode='experiment'),arena=self.recorder.manifest['arena'],body={},visuals=[],sensory=kwargs,motor=result.tolist(),rates=self.brain.last_rates,learning=self.brain.plasticity.enabled,events=list(self.events),neural_inputs=kwargs,run_id=self.recorder.id))
        self.recorder.append_window(frames,indices,times);self.elapsed+=dt;return result
    def reinforce(self,event):
        from .recording import space_check
        space_check(self.recorder.folder,4096)
        self.brain.reinforce(event)
        self.events.append(dict(time=self.elapsed,kind='reinforcement',value=float(event),message='Modeled plasticity reinforcement'))
        with (self.recorder.folder/'events.jsonl').open('a') as f:
            f.write(json.dumps(self.events[-1])+'\n');f.flush();__import__('os').fsync(f.fileno())
    def finish(self,outcome):self.recorder.finish(outcome)
