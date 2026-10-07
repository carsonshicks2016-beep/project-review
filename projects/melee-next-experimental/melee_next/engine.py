import fcntl
import multiprocessing as mp
import os
import queue
import signal
import subprocess
import time
import traceback
from pathlib import Path
import torch
from .actor import actor_main
from .curriculum import Curriculum
from .model import Policy,seed_all,load,save,publish
from .learning import update
from .storage import ROOT,read,atomic_json,append,rows,digest

TERMINAL={'completed','stopped','failed'}

def lock_ports(config):
    handles=[]
    try:
        for port in range(config.base_port,config.base_port+config.workers):
            path=ROOT/'.runtime'/f'port-{port}.lock';path.parent.mkdir(exist_ok=True)
            handle=path.open('a');handles.append(handle)
            fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
        return handles
    except Exception:
        for handle in handles:handle.close()
        raise RuntimeError('Another Melee Next job owns these emulator ports')

def reap_owned_dolphin(slot):
    pid=read(slot/'dolphin.json').get('pid')
    if not pid:return
    command=subprocess.run(['ps','-p',str(pid),'-o','command='],capture_output=True,text=True).stdout
    if str(slot/'dolphin-user') in command:
        try:os.kill(pid,signal.SIGKILL)
        except ProcessLookupError:pass

def train(config,directory,checkpoint=None,seconds=None,mode='train'):
    config.validate();directory=Path(directory).resolve();directory.mkdir(parents=True,exist_ok=True)
    state=dict(status='starting',mode=mode,backend=config.backend,pid=os.getpid(),started=time.time(),steps=0,updates=0,
               accepted=0,stale=0,phase='initializing',workers=config.workers,history=[])
    atomic_json(directory/'config.json',config.asdict());atomic_json(directory/'status.json',state)
    ctx=mp.get_context('spawn');stop=ctx.Event();output=ctx.Queue(maxsize=config.workers*3);processes=[];locks=[]
    policy=None;optimizer=None;progress={};last_saved=-1;first_sample=None;first_accepted=0;finished='completed'
    def request_stop(*_):stop.set()
    previous_handlers={s:signal.signal(s,request_stop) for s in (signal.SIGTERM,signal.SIGINT)}
    def status(**kw):
        state.update(kw);state['updated']=time.time();state['elapsed']=time.time()-state['started']
        atomic_json(directory/'status.json',state)
    def snapshot():
        nonlocal last_saved
        if policy is None or last_saved==state['steps']:return
        identity=save(directory/'checkpoints'/f"step-{state['steps']:012d}.pt",policy,optimizer,config,progress)
        atomic_json(directory/'latest.json',identity);last_saved=state['steps'];state['checkpoint']=identity
    try:
        if not config.disk_ok():raise RuntimeError('Free disk space is below the configured reserve')
        locks=lock_ports(config);seed_all(config.seed);policy=Policy();optimizer=torch.optim.Adam(policy.parameters(),lr=config.learning_rate,eps=1e-5)
        progress={'steps':0,'updates':0,'curriculum':Curriculum(config).state()}
        if checkpoint:
            checkpoint=Path(checkpoint).resolve()
            progress=load(checkpoint,policy,config,optimizer)
            state['parent']={'path':str(checkpoint),'sha256':digest(checkpoint)}
            atomic_json(directory/'parent.json',state['parent'])
        initial=int(progress['steps']);target=initial+config.total_steps;version=int(progress['updates'])
        state.update(steps=initial,updates=version,target_steps=target)
        curriculum=Curriculum(config,progress.get('curriculum'))
        atomic_json(directory/'curriculum.json',curriculum.state());publish(directory/'broadcast.pt',policy,version)
        snapshot()
        for index in range(config.workers):
            process=ctx.Process(target=actor_main,args=(config,str(directory),index,output,stop),name=f'melee-next-{index}')
            process.start();processes.append(process)
        pending=[];pending_count=0;match_counts={};last_merge=0.;last_report=0.;last_data=time.monotonic();clock=time.monotonic()
        while state['steps']<target and not stop.is_set():
            command=read(directory/'control.json')
            if mode=='benchmark' and read(directory.parent/'control.json').get('stop'):command={'stop':True}
            if command.get('stop'):finished='stopped';break
            if not config.disk_ok():raise RuntimeError('Disk reserve reached; stopping and preserving checkpoint')
            if seconds and first_sample is not None and time.monotonic()-first_sample>=seconds:break
            now=time.monotonic()
            if now-last_merge>.5:
                for index in range(config.workers):
                    source=directory/f'worker-{index:02d}'/'matches.jsonl';records=rows(source)
                    for row in records[match_counts.get(index,0):]:
                        append(directory/'matches.jsonl',row)
                        if row['result']!='setup_failed':curriculum.record(row)
                    match_counts[index]=len(records)
                atomic_json(directory/'curriculum.json',curriculum.state());last_merge=now
            if command.get('pause'):
                pending=[];pending_count=0
                while True:
                    try:state['stale']+=len(output.get_nowait()['obs'])
                    except queue.Empty:break
                status(status='paused',phase='emulators closed; checkpoint preserved');snapshot();time.sleep(.2);last_data=now;continue
            if not any(p.is_alive() for p in processes):raise RuntimeError('All game workers stopped; inspect worker errors')
            if now-last_data>180:raise TimeoutError('No usable rollout arrived for 180 seconds')
            try:fragment=output.get(timeout=.2)
            except queue.Empty:fragment=None
            if fragment is not None:
                last_data=now;length=len(fragment['obs'])
                if fragment['version']!=version:state['stale']+=length
                else:
                    if first_sample is None:first_sample=time.monotonic();state['startup_seconds']=first_sample-clock
                    remaining=target-state['steps']-pending_count
                    if length>remaining:
                        # Keep precomputed fragment GAE but limit updates to the exact budget.
                        for key in ('obs','actions','logp','advantages','returns'):fragment[key]=fragment[key][:remaining]
                        length=remaining
                    pending.append(fragment);pending_count+=length
            needed=min(config.batch_steps,target-state['steps'])
            if pending_count>=needed and pending_count:
                status(status='running',phase='learning')
                started=time.monotonic();metrics=update(policy,optimizer,pending,config);learn_seconds=time.monotonic()-started
                state['steps']+=pending_count;state['accepted']+=pending_count;version+=1;state['updates']=version
                metrics.update(steps=state['steps'],version=version,learner_seconds=learn_seconds,time=time.time())
                append(directory/'metrics.jsonl',metrics);state['history']=(state['history']+[metrics])[-100:]
                progress={'steps':state['steps'],'updates':version,'curriculum':curriculum.state()}
                publish(directory/'broadcast.pt',policy,version);pending=[];pending_count=0
                if version%config.checkpoint_every==0:snapshot()
            if now-last_report>.5:
                slots=[read(directory/f'worker-{i:02d}'/'status.json') for i in range(config.workers)]
                elapsed=max(.001,now-first_sample) if first_sample else 0
                status(status='running',phase='collecting',pending=pending_count,slots=slots,
                       accepted_per_second=state['accepted']/elapsed if elapsed else 0,
                       collected=sum(s.get('collected',0) for s in slots),cpu_level=curriculum.level,
                       active_workers=sum(p.is_alive() for p in processes))
                last_report=now
        if stop.is_set():finished='stopped'
        # An incomplete PPO batch is not claimed as trained progress.
        progress.update(steps=state['steps'],updates=version,curriculum=curriculum.state())
        snapshot()
        measurement=time.monotonic()-first_sample if first_sample else 0
        status(status=finished,phase='checkpoint saved',measurement_seconds=measurement,
               accepted_per_second=state['accepted']/measurement if measurement else 0,
               pending_not_trained=pending_count,pending=0)
    except BaseException as exc:
        finished='failed'
        status(status='failed',phase='needs attention',error=str(exc),traceback=traceback.format_exc())
        if policy is not None and progress:
            try:snapshot()
            except Exception as save_error:status(checkpoint_error=str(save_error))
    finally:
        stop.set()
        deadline=time.monotonic()+12
        for process in processes:process.join(max(0,deadline-time.monotonic()))
        for index,process in enumerate(processes):
            if process.is_alive():process.terminate();process.join(3)
            if process.is_alive():process.kill();process.join(3)
            reap_owned_dolphin(directory/f'worker-{index:02d}')
        # Include shutdown interruption records in evidence.
        merged=[]
        for index in range(config.workers):merged.extend(rows(directory/f'worker-{index:02d}'/'matches.jsonl'))
        with (directory/'matches.jsonl').open('w') as f:
            import json
            for row in sorted(merged,key=lambda r:r['time']):f.write(json.dumps(row)+'\n')
        for handle in locks:handle.close()
        output.close();output.cancel_join_thread()
        for s,handler in previous_handlers.items():signal.signal(s,handler)
        status(status=finished,cleanup_complete=True,finished=time.time(),matches=len(merged),
               wins=sum(r['result']=='win' for r in merged))
    return state
