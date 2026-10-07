from pathlib import Path
import copy,json,time,uuid,threading,queue,traceback,hashlib
import numpy as np
from .world import World,default_arena,shelter
from .storage import Store,ROOT
from .controller_contract import controller_contract,attribute_motor,TONIC_P9

class Engine:
    window=.1
    def __init__(self,shared=None,commands=None):
        self.shared=shared;self.commands=commands if commands is not None else queue.Queue();self.lock=threading.Lock();self.cached={'ready':False,'phase':'Starting physical fly'};self.geometry_cache={};self.running=False;self.stop=False;self.body=None;self.brain=None;self.mode='full';self.world=World();self.store=Store();self.parent=None;self.lineage=uuid.uuid4().hex;self.history=[];self.frames=0;self.sim_seconds=0.;self.wall_seconds=0.;self.error=None;self.last_sensory={};self.last_motor=[0,0];self.reinforcement=0.;self.unlocks=1;self.current_run=None;self.moderate=True;self.target_speed=1.;self.next_tick=0.
        if shared is None:
            self.thread=threading.Thread(target=self.loop,daemon=True,name='FlyGarden-simulation');self.thread.start()
    def submit(self,action,arguments=None,timeout=180):
        event=threading.Event();result={};self.commands.put((action,arguments or {},event,result))
        if not event.wait(timeout):raise TimeoutError('Simulation is busy; command remains queued')
        if 'error' in result:raise ValueError(result['error'])
        return result.get('value')
    def state(self):
        with self.lock:return copy.deepcopy(self.cached)
    def finish_run(self,status='interrupted'):
        if getattr(self,'recorder',None) and not self.recorder.closed:
            world=self.recorder.meta.get('last_world',{})
            self.recorder.finish(dict(status=world.get('status',self.world.status),food=world.get('collected',self.world.collected),time=world.get('time',self.world.time),valid=world.get('valid',self.world.valid),mode=self.mode),status=status)
    def new_run(self):
        self.finish_run()
        from .recording import Recorder
        self.current_run=uuid.uuid4().hex;self.frames=0
        controller_identity=controller_contract(self.mode,self.brain.plasticity.enabled if self.brain else False)
        self.recorder=Recorder(dict(controller=controller_identity,id=self.current_run,lineage=self.lineage,parent=self.parent,checkpoint_links=[self.parent] if self.parent else [],seed=self.world.seed,mode=self.mode,arena=copy.deepcopy(self.world.arena),learning=controller_identity['learning_active'],provenance=json.loads((ROOT/'reports/provenance.json').read_text())),self.geometry_cache,self.brain.ids if self.mode=='full' and self.brain else None)
        if not hasattr(self,'last_motor_decision'):self.last_motor_decision={}
        self.run_path=self.recorder.folder;self.pending_recording=None
    def load_brain(self):
        if self.brain is None:
            with self.lock:self.cached['phase']='Loading all 138,639 neurons; first run compiles the model'
            if self.shared is not None:self.shared['state']=copy.deepcopy(self.cached)
            from .brain import FullBrain
            self.brain=FullBrain(seed=self.world.seed,learning=True);self.brain.mark_initial()
            if getattr(self,'recorder',None) and self.mode=='full':self.recorder.attach_neurons(self.brain.ids);self.recorder.manifest['learning']=self.brain.plasticity.enabled;__import__('flygarden.recording',fromlist=['atomic_json']).atomic_json(self.recorder.folder/'manifest.json',self.recorder.manifest)
            if getattr(self,'recorder',None) and self.mode=='full':
                self.recorder.manifest['controller']=controller_contract(self.mode,self.brain.plasticity.enabled)
                __import__('flygarden.recording',fromlist=['atomic_json']).atomic_json(self.recorder.folder/'manifest.json',self.recorder.manifest)
    def rebuild_body(self):
        from .body import Body
        if self.body:self.body.close()
        self.body=Body(seed=self.world.seed,blocks=self.world.arena['blocks'],spawn=self.world.arena['spawn'],foods=self.world.arena['foods'],predator=self.world.arena['predator'])
        self.geometry_cache=self.body.geometry()
        if self.shared is not None:self.shared['geometry']=self.geometry_cache
    def publish(self,record=False):
        body=self.body.observation() if self.body else {}
        throughput=self.sim_seconds/self.wall_seconds if self.wall_seconds else 0
        state=dict(target_speed=self.target_speed,ready=self.body is not None,phase='Paused' if not self.running else 'Running',running=self.running,error=self.error,mode=self.mode,world={k:getattr(self.world,k) for k in ('time','energy','collected','status','mode','level','rewarded_odor','valid','travel','escape_count','stalls')},arena=self.world.arena,body=body,reinforcement=self.reinforcement,visuals=self.body.visuals() if self.body else [],sensory=self.last_sensory,motor=self.last_motor,rates=self.brain.last_rates if self.brain else {},learning=self.brain.plasticity.enabled if self.brain else False,learning_updates=self.brain.plasticity.updates if self.brain else 0,plastic_edges=len(self.brain.plastic_indices) if self.brain else 0,events=self.world.events[-30:],history=self.history[-100:],checkpoints=self.store.list(),unlocks=self.unlocks,lineage=self.lineage,run_id=self.current_run,throughput=throughput,coverage={'neurons':self.brain.n if self.brain else 0,'connections':self.brain.edges if self.brain else 0,'anatomical_synapses':self.brain.anatomical_synapses if self.brain else 0},claims=dict(brain='Full imported network' if self.brain else 'Not loaded',body='NeuroMechFly 2.1 + supplied HybridTurningController',vision='Engineered egocentric proxy; retina mapping unvalidated',contact='Contact and motion feedback feed the supplied gait controller; ascending brain mapping remains unvalidated',odor='Synthetic ORN_DM1/DM2 channels',learning='Experimental compartment plasticity; behavioral learning not established',reflex='Engineered escape reflex; identical in plasticity comparisons',real_time=throughput>=1))
        state['controller_contract']=controller_contract(self.mode,self.brain.plasticity.enabled if self.brain else False)
        state['motor_decision']=copy.deepcopy(getattr(self,'last_motor_decision',{}))
        state['learning']=state['controller_contract']['learning_active']
        if self.mode=='baseline':
            state.update(rates={},coverage={'neurons':0,'connections':0,'anatomical_synapses':0},learning_updates=0,plastic_edges=0)
            state['claims']['brain']='Inactive in supplied baseline mode'
        with self.lock:self.cached=copy.deepcopy(state)
        if self.shared is not None:self.shared['state']=state
        return state
    def flush_recording(self):
        if self.pending_recording:
            frames,indices,times=self.pending_recording
            self.recorder.append_window(frames,indices,times);self.frames+=len(frames);self.pending_recording=None
        if self.world.status!='running':self.finish_run('complete')
    def tick(self):
        self.flush_recording()
        if self.world.status!='running':self.running=False;return
        from .recording import space_check
        space_check(self.run_path,16*1024**2)
        before_state=copy.deepcopy(self.publish());input_reinforcement=self.reinforcement;world_start=self.world.time;body_start=self.body.steps*self.body.dt;poses=[]
        if not self.frames:poses.append((world_start,self.body.recording_pose()))
        t=time.perf_counter();self.body.update_visual_world(self.world.arena['foods'],self.world.arena['predator']);obs=self.body.observation();sensory=self.world.sensory(obs);self.last_sensory=sensory
        if self.mode=='full':
            self.load_brain()
            # Tonic descending stimulation is an engineered exploratory drive, not learned steering.
            loom=[sensory['threat'] if sensory['threat_bearing']>0 else 0,sensory['threat'] if sensory['threat_bearing']<=0 else 0]
            brain_start=float(self.brain.network.t/__import__('brian2').second)
            motor=self.brain.advance(self.window,odor=sensory['odor'],p9=TONIC_P9,loom=loom,reinforcement=self.reinforcement)
        else:
            from .baseline import motor_command
            motor=motor_command(sensory,self.world.time)
        self.last_motor_decision=attribute_motor(self.mode,sensory,motor)
        self.last_motor_decision.update(observation_time=world_start,applied_start=world_start,applied_end=world_start+self.window)
        self.last_motor=self.last_motor_decision['applied_motor'];next_obs=self.body.advance(self.window,self.last_motor,capture=lambda stamp,pose:poses.append((world_start+stamp-body_start,pose)))
        if max(self.last_motor)>.1 and np.linalg.norm(np.asarray(next_obs['position'])[:2]-np.asarray(obs['position'])[:2])<.005:self.world.stalls+=1
        self.reinforcement=self.world.advance(self.window,next_obs)
        if self.world.status=='captured':
            self.reinforcement=0.;self.world.event('conditioning','Capture conditioning disabled pending controlled aversive validation')
        if self.mode=='full':self.brain.reinforce(self.reinforcement)
        self.sim_seconds+=self.window;compute_elapsed=time.perf_counter()-t;self.wall_seconds+=compute_elapsed
        if self.world.status!='running':
            self.running=False;self.history.append(dict(run=self.current_run,level=self.world.level,status=self.world.status,food=self.world.collected,time=self.world.time,valid=self.world.valid,mode=self.mode,travel=self.world.travel,stalls=self.world.stalls,captured=self.world.status=='captured',escape_events=self.world.escape_count))
            if self.world.status=='success' and self.world.valid and self.world.mode=='challenge':self.unlocks=min(6,max(self.unlocks,self.world.level+1))
        final=copy.deepcopy(self.publish());frames=[]
        if self.reinforcement:
            final['events'].append(dict(time=self.world.time,kind='reinforcement',value=float(self.reinforcement),message='Positive modeled reinforcement' if self.reinforcement>0 else 'Aversive modeled reinforcement'))
        for stamp,pose in poses:
            snapshot=copy.deepcopy(final if stamp>=self.world.time-1e-6 else before_state)
            snapshot.update(pose);snapshot['world']['time']=stamp;snapshot.update(sensory=self.last_sensory,motor=self.last_motor,motor_decision=copy.deepcopy(self.last_motor_decision),running=False,checkpoints=[],history=[],neural_inputs=dict(odor=sensory['odor'],p9=list(TONIC_P9),loom=loom,reinforcement=input_reinforcement) if self.mode=='full' else {})
            frames.append(snapshot)
        indices,times=(self.brain.last_spikes if self.mode=='full' else ([],[]))
        if self.mode=='full':times=np.asarray(times)-brain_start+world_start
        self.pending_recording=(frames,indices,times)
        try:self.flush_recording()
        finally:
            self.wall_seconds+=time.perf_counter()-t-compute_elapsed
            throughput=self.sim_seconds/self.wall_seconds
            with self.lock:self.cached['throughput']=throughput;self.cached['claims']['real_time']=throughput>=1
            if self.shared is not None:self.shared['state']=copy.deepcopy(self.cached)
    def command(self,action,arg):
        if action=='play':
            self.flush_recording()
            if self.world.status!='running':raise ValueError('Trial ended; reset before continuing')
            if self.mode=='full':self.load_brain()
            self.running=True
        elif action=='speed':
            speed=float(arg['speed'])
            if not np.isfinite(speed) or not .1<=speed<=4:raise ValueError('Requested speed must be 0.1–4 simulated seconds per wall second')
            self.target_speed=speed;self.next_tick=0.
        elif action=='pause':self.running=False
        elif action=='shutdown':self.running=False;self.flush_recording();self.finish_run('interrupted')
        elif action=='step':self.running=False;self.tick()
        elif action=='mode':
            self.running=False
            if arg['mode'] not in ('full','baseline'):raise ValueError('Unknown model')
            self.mode=arg['mode'];self.new_run();self.world.valid=False;self.world.event('mode','Controller changed; reset required for a scored trial')
        elif action=='learning':
            if self.mode!='full':raise ValueError('Choose the engineered full-brain hybrid to change neural plasticity')
            self.running=False;self.load_brain();self.brain.plasticity.enabled=bool(arg['enabled']);self.world.valid=False;self.world.event('learning','Plasticity toggled; current trial invalid for scientific scoring');self.new_run()
        elif action=='reset':
            self.running=False;level=int(arg.get('level',self.world.level));mode=arg.get('activity',self.world.mode)
            if not 1<=level<=6:raise ValueError('Invalid challenge level')
            if mode not in ('sandbox','challenge'):raise ValueError('Invalid activity')
            if mode=='challenge' and level>self.unlocks:raise ValueError('Complete preceding challenges first; sandbox can load any layout')
            seed=int(arg.get('seed',self.world.seed+1));rewarded=int(arg.get('rewarded_odor',self.world.rewarded_odor))
            if rewarded not in (0,1):raise ValueError('Invalid odor identity')
            arena=None if 'level' in arg else self.world.trial_arena()
            self.world=World(seed=seed,level=level,mode=mode,arena=arena,valid=not (arena or {}).get('custom',False));self.world.rewarded_odor=rewarded
            if self.brain:self.brain.reset_transient(seed=self.world.seed)
            self.reinforcement=0.;self.last_sensory={};self.last_motor=[0,0];self.last_motor_decision={};self.rebuild_body();self.new_run();self.world.event('reset','Body and transient activity reset; learned weights retained')
        elif action=='edit':
            self.running=False;kind=arg['kind'];x=float(arg['x']);y=float(arg['y'])
            if not np.isfinite([x,y]).all() or abs(x)>36 or abs(y)>26:raise ValueError('Place items within arena bounds')
            if kind not in ('food_a','food_b','block','terrain','shelter','predator','spawn'):raise ValueError('Unknown arena tool')
            arena=copy.deepcopy(self.world.arena);arena['custom']=True
            if kind.startswith('food_'):arena['foods'].append(dict(id=uuid.uuid4().hex[:8],x=x,y=y,odor=0 if kind=='food_a' else 1,units=3))
            elif kind=='block':arena['blocks'].append(dict(id=uuid.uuid4().hex[:8],x=x,y=y,w=4.,h=4.,z=3.))
            elif kind=='terrain':arena['blocks'].append(dict(id=uuid.uuid4().hex[:8],x=x,y=y,w=2.,h=12.,z=.15,kind='terrain'))
            elif kind=='shelter':arena['blocks']+=shelter(x,y)
            elif kind=='predator':arena['predator'].update(enabled=True,x=x,y=y,spawn=[x,y],spawn_heading=2.5,state='patrol',last_seen=None,velocity=0.)
            else:arena['spawn']=[x,y]
            spawn=arena['spawn']
            from .world import intersects
            if any(intersects(spawn,spawn,b,1.) for b in arena['blocks']):raise ValueError('Obstacle overlaps fly spawn; move the spawn or obstacle')
            self.world=World(seed=self.world.seed,level=self.world.level,mode=self.world.mode,arena=arena,valid=False,rewarded_odor=self.world.rewarded_odor)
            if self.brain:self.brain.reset_transient(seed=self.world.seed)
            self.reinforcement=0.;self.last_sensory={};self.last_motor=[0,0];self.last_motor_decision={};self.rebuild_body();self.new_run();self.world.event('edit','Arena edited: paused trial reset, retained learning, evaluation invalidated')
        elif action=='difficulty':
            self.running=False;speed=float(arg['speed'])
            if not np.isfinite(speed) or not 2<=speed<=12:raise ValueError('Predator speed must be between 2 and 12 mm/s')
            self.world.arena['predator']['speed']=speed;self.world.arena['predator']['velocity']=min(speed,self.world.arena['predator']['velocity']);self.world.valid=False;self.new_run();self.world.event('difficulty','Predator speed changed; evaluation invalidated')
        elif action=='retina':
            self.running=False
            import io,base64
            from PIL import Image
            frames=self.body.eye_frames();images=[]
            for frame in frames:
                buffer=io.BytesIO();Image.fromarray(frame).save(buffer,format='PNG');images.append('data:image/png;base64,'+base64.b64encode(buffer.getvalue()).decode())
            self.publish();return dict(images=images,readout_shape=list(self.body.retinal_readouts().shape),scope='Actual compound-eye images; controller still uses an unvalidated geometric threat proxy')
        elif action=='save':
            self.running=False;kind=arg.get('kind','scene')
            if kind not in ('scene','learned'):raise ValueError('Unknown checkpoint kind')
            if kind=='learned' and self.brain is None:
                if self.mode!='full':raise ValueError('Choose the full connectome before saving learned parameters')
                self.load_brain()
            payload=dict(version=2,display=dict(sensory=self.last_sensory,motor=self.last_motor),mode=self.mode,lineage=self.lineage,world=self.world.snapshot(),history=self.history,unlocks=self.unlocks,reinforcement=self.reinforcement,learned=self.brain.learned_state() if self.brain else None)
            payload['controller_contract']=controller_contract(self.mode,self.brain.plasticity.enabled if self.brain else False)
            payload['display']['motor_decision']=copy.deepcopy(getattr(self,'last_motor_decision',{}))
            if kind=='scene':payload['body']=self.body.snapshot()
            else:
                for key in ('world','display','reinforcement'):payload.pop(key,None)
            metadata=self.store.create(kind,payload,self.brain if kind=='scene' else None,parent=self.parent);self.parent=metadata['id'];self.world.event('save',f'Immutable {kind} checkpoint saved');self.publish();return metadata
        elif action in ('load','branch'):
            self.running=False;payload,metadata,folder=self.store.load(arg['id'])
            if metadata['kind']=='scene' and not payload.get('body',{}).get('signature'):raise ValueError('Legacy scene predates portable body state; use a learned-fly checkpoint')
            if metadata['brain']:
                import pickle
                with open(folder/'brain.bin.adapter','rb') as f:adapter=pickle.load(f)
                if adapter.get('version')!=2:raise ValueError('Legacy scene predates portable brain state; use a learned-fly checkpoint')
            self.mode=payload['mode'];self.history=payload['history'];self.unlocks=payload['unlocks']
            self.last_motor_decision=copy.deepcopy(payload.get('display',{}).get('motor_decision',{}))
            if metadata['kind']=='scene':
                self.last_sensory=payload.get('display',{}).get('sensory',{});self.last_motor=payload.get('display',{}).get('motor',[0,0]);self.reinforcement=payload['reinforcement'];self.world.restore(payload['world']);self.rebuild_body()
                if metadata['brain']:self.load_brain();self.brain.load(folder/'brain.bin')
                elif self.brain is not None:
                    self.brain.restore_learned(dict(weights=self.brain.plasticity.base.copy(),updates=0,enabled=True));self.brain.reset_transient(seed=self.world.seed)
                self.body.restore(payload['body'])
            else:
                arena=self.world.trial_arena();self.world=World(seed=self.world.seed,level=self.world.level,mode=self.world.mode,arena=arena,rewarded_odor=self.world.rewarded_odor,valid=not arena.get('custom',False));self.reinforcement=0.;self.last_sensory={};self.last_motor=[0,0]
                if payload['learned'] is not None:self.load_brain();self.brain.reset_transient(seed=self.world.seed);self.brain.restore_learned(payload['learned'])
                self.rebuild_body()
            self.lineage=uuid.uuid4().hex if action=='branch' else payload['lineage'];self.parent=metadata['id'];self.new_run();self.world.event(action,'Checkpoint restored' if action=='load' else 'New lineage branched from immutable checkpoint')
        else:raise ValueError('Unknown command')
        self.publish();return dict(ok=True)
    def loop(self):
        try:self.rebuild_body();self.new_run();self.publish()
        except Exception:self.error=traceback.format_exc();self.publish();return
        while not self.stop:
            try:
                try:item=self.commands.get(timeout=.01 if self.running else .1)
                except queue.Empty:item=None
                if item:
                    action,arg,event,result=item
                    try:result['value']=self.command(action,arg)
                    except Exception as exc:result['error']=str(exc)
                    finally:
                        if self.shared is not None:self.shared['reply_'+event]=result
                        else:event.set()
                elif self.running and time.monotonic()>=self.next_tick:
                    begun=time.monotonic();self.tick();self.next_tick=begun+self.window/self.target_speed
            except Exception:
                self.running=False;self.error=traceback.format_exc();self.world.event('error',self.error.splitlines()[-1]);self.publish()


def _simulation_main(shared,commands):
    import faulthandler
    faulthandler.enable()
    engine=Engine(shared=shared,commands=commands)
    engine.loop()

class LocalSimulation:
    """Run native simulation on its own process main thread, outside HTTP threads."""
    def __init__(self):
        import multiprocessing as mp
        self.ctx=mp.get_context('spawn');self.manager=self.ctx.Manager();self.shared=self.manager.dict();self.shared['state']={'ready':False,'phase':'Starting physical fly'}
        self.commands=self.ctx.Queue();self.process=self.ctx.Process(target=_simulation_main,args=(self.shared,self.commands),name='FlyGardenSimulation');self.process.start()
    def state(self):
        state=self.shared.get('state',{})
        if not self.process.is_alive():state={**state,'running':False,'error':'Simulation process exited; checkpoints are preserved. Restart Fly Garden.'}
        return state
    @property
    def geometry_cache(self):return self.shared.get('geometry',{})
    def submit(self,action,arguments=None,timeout=180):
        if not self.process.is_alive():raise ValueError('Simulation process exited; restart Fly Garden')
        identity=uuid.uuid4().hex;self.commands.put((action,arguments or {},identity,{}));deadline=time.monotonic()+timeout;key='reply_'+identity
        while time.monotonic()<deadline:
            if key in self.shared:
                result=self.shared.pop(key)
                if 'error' in result:raise ValueError(result['error'])
                return result.get('value')
            if not self.process.is_alive():raise ValueError('Native simulation exited; checkpoint files remain intact')
            time.sleep(.02)
        raise TimeoutError('Simulation is busy; command remains queued')
    def close(self):
        if self.process.is_alive():
            try:self.submit('shutdown',timeout=8)
            except (ValueError,TimeoutError):pass
            self.process.terminate();self.process.join(10)
        self.manager.shutdown()
