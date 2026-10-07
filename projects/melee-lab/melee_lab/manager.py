import fcntl
import os
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path
from .config import ROOT, Config
from .storage import write_json, validate_checkpoint
from .catalog import Catalog, read_json, signature
from .analytics import MatchIndex

ACTIVE = {'starting', 'running', 'paused'}


class RunBusy(ValueError):
    """The caller may retry after the active run or emulator teardown finishes."""


def alive(pid):
    try:
        p = int(pid)
        os.kill(p, 0)
        try:
            cmd = subprocess.check_output(['ps', '-p', str(p), '-o', 'command='], text=True, stderr=subprocess.DEVNULL)
            return any(k in cmd for k in ('python', 'melee_lab', 'slippi', 'dolphin'))
        except Exception:
            return True
    except (ProcessLookupError, ValueError, TypeError, PermissionError):
        return False


class Manager:
    def __init__(self, root=ROOT):
        self.root=Path(root).resolve()
        self.runs=self.root/'runs'; self.runs.mkdir(exist_ok=True)
        self.lock=threading.RLock(); self.children={}
        self.catalog=Catalog(self.root); self.matches=MatchIndex(); self.json_cache={}

    def cached_json(self, path):
        path=Path(path)
        try: stamp=signature(path)
        except FileNotFoundError: return {}
        cached=self.json_cache.get(str(path))
        if cached and cached[0]==stamp: return dict(cached[1])
        data=read_json(path)
        self.json_cache[str(path)]=(stamp,data)
        return dict(data)

    def run_path(self, run_id):
        path=(self.runs/run_id).resolve()
        if path.parent!=self.runs or not path.is_dir():
            raise ValueError('Run was not found.')
        return path

    def slots(self, directory, state):
        indices=[int(p.name[3:]) for p in directory.glob('env*') if p.is_dir() and p.name[3:].isdigit()]
        count=max(int(state.get('envs') or 1), max(indices, default=-1)+1)
        result=[]
        for i in range(count):
            path=directory/f'env{i}'/'status.json'
            single=count==1 and not path.exists()
            data=dict(state) if single else self.cached_json(path)
            updated=(directory/'status.json' if single else path)
            age=max(0,time.time()-updated.stat().st_mtime) if updated.exists() else None
            live={k:data[k] for k in ('connection','frame','action','players','cpu_level','result','episode',
                                     'starting_stocks','TimeLimit_truncated','failed','traceback','human_connected','opponent_type','recoveries','recovery_reason',
                                     'stalled_observations','reset_seconds','reset_seconds_mean') if k in data}
            live.update(index=i, age_seconds=round(age,1) if age is not None else None,
                        waiting=not bool(data), stale=bool(age is not None and age>15 and state['status']=='running'))
            result.append(live)
        return result

    def list_runs(self, include_slots=False, selected_run=None):
        with self.lock:
            for key,child in list(self.children.items()):
                if child.poll() is not None: del self.children[key]
            result=[]
            paths=sorted(self.runs.glob('*/status.json'),key=lambda p:p.stat().st_mtime,reverse=True)
            for path in paths:
                state=self.cached_json(path)
                if not state: continue
                request=self.cached_json(path.parent/'request.json')
                state['id']=path.parent.name
                state['mode']=state.get('mode') or request.get('mode') or ('evaluate' if 'evaluat' in state.get('phase','') else 'train')
                state['envs']=state.get('envs') or request.get('envs') or max(1,len(list(path.parent.glob('env[0-9]*'))))
                state['requested_steps']=request.get('steps')
                state['opponent_checkpoint']=request.get('opponent_checkpoint')
                state['benchmark']=request.get('benchmark',False)
                state['source_checkpoint']=request.get('checkpoint')
                state['config']=self.cached_json(path.parent/'config.json')
                state['pid']=state.get('pid') or self.cached_json(path.parent/'process.json').get('pid')
                meta=self.cached_json(path.parent/'latest.json')
                input_meta=self.cached_json(path.parent/'input-policy.json')
                state['architecture']=state.get('architecture') or meta.get('architecture') or input_meta.get('architecture') or ('recurrent' if request.get('recurrent') else 'feedforward')
                if state.get('status') in ACTIVE and state['pid'] and not alive(state['pid']):
                    state.update(status='failed',error='Worker exited. Its last checkpoint and logs are preserved.')
                state['control']=self.cached_json(path.parent/'control.json')
                state['recovery']=self.cached_json(path.parent/'recovery.json')
                if include_slots and (state.get('status') in ACTIVE or state['id']==selected_run or (not selected_run and not result)):
                    state['slots']=self.slots(path.parent,state)
                state.pop('history',None)  # Full match data is incremental and served on demand.
                match_path=path.parent/'matches.jsonl'
                state['match_revision']=list(signature(match_path)) if match_path.exists() else None
                result.append(state)
            return result

    def active(self):
        return next((r for r in self.list_runs() if r.get('status') in ACTIVE),None)

    def availability(self):
        active=self.active()
        if active:
            stopping=active.get('control',{}).get('stop',False)
            return dict(can_start=False, reason='stopping' if stopping else 'active', run_id=active['id'],
                        message='Saving and closing emulators…' if stopping else 'A run is active.')
        lock_path=self.root/'.runtime/emulator.lock'
        if lock_path.exists():
            with lock_path.open('a') as handle:
                try: fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
                except BlockingIOError:
                    return dict(can_start=False,reason='teardown',message='Emulators are still closing. Start will be available shortly.')
                else: fcntl.flock(handle,fcntl.LOCK_UN)
        return dict(can_start=True,reason='ready',message='Ready for a new run.')

    def checkpoints(self):
        with self.lock: return self.catalog.checkpoints()

    def dataset(self,name):
        path=(self.root/name).resolve()
        if not path.is_relative_to((self.root/'datasets').resolve()) or path.suffix!='.npz' or not path.is_file():
            raise ValueError('Choose an existing demonstration dataset in datasets/.')
        return path

    def datasets(self):
        """Parsed tournament footage available to anchor training to. Only the .npy
        headers are read; decompressing ten megabytes to render a dropdown is not worth
        it, and the shape is the only thing the caller has to check."""
        import zipfile
        import numpy as np
        directory=self.root/'datasets'
        result=[]
        for path in sorted(directory.glob('*.npz')) if directory.is_dir() else []:
            entry=dict(path=str(path.relative_to(self.root)),name=path.stem,
                       bytes=path.stat().st_size,updated=path.stat().st_mtime,
                       samples=None,features=None,packed=False,returns=False,groups=False)
            try:
                with zipfile.ZipFile(path) as archive:
                    names=set(archive.namelist())
                    # Packed datasets carry ids/continuous; pre-packing ones carry
                    # observations. Either way the row count is in the .npy header.
                    member='ids.npy' if 'ids.npy' in names else 'observations.npy'
                    with archive.open(member) as handle:
                        version=np.lib.format.read_magic(handle)
                        shape,_,_=(np.lib.format.read_array_header_1_0(handle) if version==(1,0)
                                   else np.lib.format.read_array_header_2_0(handle))
                    from .state import OBS_SIZE
                    entry.update(samples=int(shape[0]),packed=member=='ids.npy',
                                 features=OBS_SIZE if member=='ids.npy' else (int(shape[1]) if len(shape)>1 else 0),
                                 returns='returns.npy' in names,groups='groups.npy' in names)
            except (OSError,ValueError,KeyError):
                pass
            result.append(entry)
        return result

    def checkpoint(self,name,config):
        path=self.catalog.resolve(name)
        validate_checkpoint(path,config)
        return path

    def start(self,mode='train',steps=100000,bootstrap=4096,checkpoint=None,episodes=10,curriculum=True,envs=1,level=None,human_character=None,opponent_checkpoint=None,benchmark=False,fast_inputs=False,anchor=None,anchor_coef=.5,anchor_floor=.12,win_threshold=.60,finetune=False,recurrent=False,character=None,execution_mode='assisted',league_checkpoints=None):
        with self.lock:
            if mode not in ('train','evaluate','play'): raise ValueError('Unknown run mode.')
            if self.active(): raise RunBusy('A run is already active. Stop it before starting another.')
            if not 1<=envs<=12: raise ValueError('Choose between 1 and 12 emulators.')
            if level is not None and not 1<=level<=9: raise ValueError('CPU level must be 1-9.')
            if envs>1:
                if mode!='train': raise ValueError('Parallel emulators support training only.')
                if not checkpoint and not anchor and not recurrent:
                    raise ValueError('Parallel emulators need a saved checkpoint or tournament anchor.')
            availability=self.availability()
            if not availability['can_start']: raise RunBusy(availability['message'])
            config=Config.load(self.root/'config.local.json')
            config.execution_mode=execution_mode
            config.strict_evaluation=mode=='evaluate'
            config.league_manifest=None
            if league_checkpoints is not None:
                if mode!='train' or not checkpoint or opponent_checkpoint:
                    raise ValueError('League training requires a saved controller policy and no fixed opponent or transfer.')
                if len(league_checkpoints)>12: raise ValueError('Choose at most twelve historical opponents.')
                for source in league_checkpoints: self.catalog.resolve(source)
                curriculum=False
                config.randomize_opponent=True
            if character:
                config.character=str(character).upper()
            source_config={}
            if checkpoint:
                source_config=read_json(self.catalog.resolve(checkpoint).with_suffix('.json'))['config']
                config.action_set=source_config.get('action_set','legacy')
                config.action_frames=source_config['action_frames']
                if not character and 'character' in source_config:
                    config.character=source_config['character']
            # Asking for the full controller on a policy that already has it is a plain
            # resume. Transferring would rebuild an action head that needs no rebuilding
            # and randomise every button in the process.
            if fast_inputs and source_config.get('action_set')=='controller': fast_inputs=False
            if config.port+envs-1>65535: raise ValueError('Emulator ports exceed 65535.')
            if mode=='play':
                config.speed=1.0
                config.opponent=human_character or config.opponent
                config.randomize_opponent=True  # permit any human matchup without changing the policy contract
            if benchmark:
                if mode!='evaluate': raise ValueError('The roster benchmark requires evaluation mode.')
                config.randomize_opponent=True
                config.cpu_level=9
                level=9
            opponent=None
            if opponent_checkpoint:
                if mode!='train': raise ValueError('Frozen opponents are for training only.')
                opponent=self.catalog.resolve(opponent_checkpoint)
                meta=read_json(opponent.with_suffix('.json'))
                config.opponent=meta['config']['character']
                config.randomize_opponent=True
                from dataclasses import replace
                opponent_config=Config(**meta['config'])
                validate_checkpoint(opponent,replace(config,character=config.opponent,action_set=opponent_config.action_set,action_frames=opponent_config.action_frames))
                curriculum=False
            if fast_inputs:
                if mode!='train': raise ValueError('Controller transfer must be trained before evaluation or challenge.')
                config.action_set='controller'; config.action_frames=1
            if anchor:
                if mode!='train': raise ValueError('Demonstration anchoring is for training only.')
                if getattr(config,'action_set','legacy')!='controller':
                    raise ValueError('Anchoring needs a full-controller policy. Resume a cloned checkpoint, or tick full controller.')
                if not 0<=anchor_coef<=10: raise ValueError('Anchor strength must be between 0 and 10.')
                anchor=self.dataset(anchor)
            config.validate()
            if checkpoint and fast_inputs:
                checkpoint=self.catalog.resolve(checkpoint)
                from dataclasses import replace
                old=Config(**read_json(checkpoint.with_suffix('.json'))['config'])
                validate_checkpoint(checkpoint,replace(config,action_set=old.action_set,action_frames=old.action_frames))
            else: checkpoint=self.checkpoint(checkpoint,config) if checkpoint else None
            if mode in ('evaluate','play') and not checkpoint: raise ValueError('Choose a checkpoint to evaluate or challenge.')
            run_id=time.strftime('%Y%m%d-%H%M%S')+'-'+uuid.uuid4().hex[:5]
            directory=self.runs/run_id; directory.mkdir()
            if league_checkpoints is not None:
                if config.action_set!='controller': raise ValueError('League training requires a full-controller checkpoint.')
                from .league import snapshot_league
                manifest=snapshot_league(self.catalog,directory/'league',checkpoint,league_checkpoints,config)
                config.league_manifest=str(manifest.resolve())
            config.save(directory/'config.json')
            request=dict(mode=mode,steps=steps,bootstrap=bootstrap,episodes=episodes,curriculum=curriculum,
                         envs=envs,level=level,benchmark=benchmark,fast_inputs=fast_inputs,opponent_checkpoint=opponent_checkpoint,human_character=human_character,checkpoint=str(checkpoint.relative_to(self.root)) if checkpoint else None,
                         anchor=str(anchor.relative_to(self.root)) if anchor else None,anchor_coef=anchor_coef if anchor else None,
                         anchor_floor=anchor_floor if anchor else None,win_threshold=win_threshold,finetune=finetune,recurrent=recurrent,
                         config=vars(config),created=time.time(),execution_mode=execution_mode,league_checkpoints=league_checkpoints)
            # Datasets are large and shared between runs, so the run records which bytes
            # it read rather than keeping a copy of them.
            if anchor: request['anchor_sha256']=self.catalog.digest(anchor)
            # Freeze every resumed/evaluated input. latest.zip may be replaced later.
            if checkpoint:
                request['checkpoint_identity']=self.catalog.snapshot(checkpoint,directory/'input-policy.zip')
                checkpoint=directory/'input-policy.zip'
            if opponent:
                request['opponent_identity']=self.catalog.snapshot(opponent,directory/'opponent-policy.zip')
                opponent=directory/'opponent-policy.zip'
            write_json(directory/'request.json',request)
            command=[sys.executable,'-u','-m','melee_lab.supervisor' if mode=='train' else 'melee_lab.worker','--mode',mode,
                     '--config',str(directory/'config.json'),'--run-dir',str(directory),
                     '--steps',str(steps),'--bootstrap',str(bootstrap),'--episodes',str(episodes),'--envs',str(envs)]
            if checkpoint: command+=['--checkpoint',str(checkpoint)]
            if fast_inputs: command+=['--fast-inputs']
            if recurrent: command+=['--recurrent']
            if anchor: command+=['--anchor',str(anchor),'--anchor-coef',str(anchor_coef),'--anchor-floor',str(anchor_floor)]
            if win_threshold is not None: command+=['--win-threshold',str(win_threshold)]
            if finetune: command+=['--finetune']
            if opponent: command+=['--opponent-checkpoint',str(opponent)]
            if benchmark: command+=['--benchmark']
            if not curriculum: command+=['--no-curriculum']
            if level is not None: command+=['--level',str(level)]
            write_json(directory/'status.json',dict(status='starting',phase='booting',mode=mode,envs=envs,started=time.time()))
            try:
                with (directory/'worker.log').open('w') as log:
                    child=subprocess.Popen(command,cwd=self.root,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL)
                self.children[run_id]=child
                write_json(directory/'process.json',{'pid':child.pid})
                write_json(directory/'status.json',dict(status='starting',phase='booting',mode=mode,envs=envs,started=time.time(),pid=child.pid))
            except Exception as exc:
                write_json(directory/'status.json',dict(status='failed',error=str(exc)))
                raise
            return run_id

    def compare(self,candidates,episodes=10,roster=True,modes=('assisted','raw'),seed=17001):
        """Replay frozen candidates against a fixed panel. This never trains or promotes.

        The comparison owns the emulator for its whole length, so it takes the same
        availability lock as a run and appears in the run list as mode 'compare'.
        """
        import shutil
        from .gauntlet import prepare
        with self.lock:
            if self.active(): raise RunBusy('A run is already active. Stop it before starting another.')
            availability=self.availability()
            if not availability['can_start']: raise RunBusy(availability['message'])
            modes=list(dict.fromkeys(modes))
            config=Config.load(self.root/'config.local.json')
            config.validate()
            run_id=time.strftime('%Y%m%d-%H%M%S')+'-'+uuid.uuid4().hex[:5]
            directory=self.runs/run_id; directory.mkdir()
            child=None
            try:
                request=prepare(self.catalog,directory,candidates,config,episodes=episodes,
                                roster=roster,modes=modes,seed=seed)
                jobs=len(request['jobs']); games=sum(j['episodes'] for j in request['jobs'])
                write_json(directory/'control.json',{})
                write_json(directory/'request.json',dict(mode='compare',candidates=list(candidates),
                           episodes=episodes,roster=roster,modes=modes,seed=seed,envs=1,jobs=jobs,games=games,
                           snapshots=request['snapshots'],config=request['config'],created=time.time()))
                status=dict(status='starting',phase='freezing candidates',mode='compare',envs=1,
                            started=time.time(),jobs_total=jobs,jobs_completed=0)
                write_json(directory/'status.json',status)
                command=[sys.executable,'-u','-m','melee_lab.gauntlet','--run-dir',str(directory)]
                with (directory/'worker.log').open('w') as log:
                    child=subprocess.Popen(command,cwd=self.root,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL)
                self.children[run_id]=child
                write_json(directory/'process.json',{'pid':child.pid})
                write_json(directory/'status.json',dict(status,pid=child.pid))
            except Exception as exc:
                # Nothing has played yet, so a failed launch leaves no evidence to keep.
                if child is None: shutil.rmtree(directory,ignore_errors=True)
                else: write_json(directory/'status.json',dict(status='failed',mode='compare',error=str(exc)))
                raise
            return run_id

    def evaluation_pace(self):
        """Median wall-clock seconds per evaluation game, from this project's own runs.

        A comparison is hundreds of games long, so the cost of starting one should be
        stated in hours before it is started, and measured rather than assumed.
        """
        samples=[]
        for path in self.runs.glob('*/evaluation.json'):
            episodes=self.cached_json(path).get('episodes') or 0
            elapsed=self.cached_json(path.parent/'status.json').get('elapsed')
            if episodes>0 and isinstance(elapsed,(int,float)) and elapsed>0:
                samples.append(elapsed/episodes)
        samples.sort()
        return dict(seconds_per_game=samples[len(samples)//2] if samples else None,trials=len(samples))

    def comparison(self,run_id):
        """The comparison's evidence, with per-job execution counters folded to totals."""
        directory=self.run_path(run_id)
        request=self.cached_json(directory/'comparison.json')
        if not request: raise ValueError('That run is not a policy comparison.')
        results=self.cached_json(directory/'results.json')
        rows=[]
        for row in results.get('rows',[]):
            counters=[c for case in row.get('execution',[]) for c in case.values()]
            decisions=sum(c.get('decisions',0) for c in counters)
            changed=sum(c.get('changed_decisions',0) for c in counters)
            rows.append(dict({k:v for k,v in row.items() if k!='execution'},
                             override_rate=changed/decisions if decisions else None))
        return dict(id=run_id,status=self.cached_json(directory/'status.json'),
                    snapshots=request.get('snapshots',[]),modes=request.get('modes',[]),
                    jobs=len(request.get('jobs',[])),seed=request.get('seed'),
                    episodes=request['jobs'][0]['episodes'] if request.get('jobs') else 0,
                    rows=rows,gate=results.get('gate'))

    def control(self,action):
        with self.lock:
            active=self.active()
            if not active: raise ValueError('No active run.')
            path=self.runs/active['id']/'control.json'
            command=read_json(path)
            if action=='pause': command['pause']=True
            elif action=='resume': command['pause']=False
            elif action=='stop': command.update(stop=True,pause=False)
            elif action=='save':
                if active['mode']!='train': raise ValueError('Only training runs can save a policy.')
                command['save']=str(time.time_ns())
            else: raise ValueError('Unknown control action.')
            write_json(path,command)
            return active['id']

    def promote(self,checkpoint,name,note):
        with self.lock: return self.catalog.promote(checkpoint,name,note)
