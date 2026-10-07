import os
import signal
import subprocess
import sys
import threading
import time
import uuid
from dataclasses import replace
from pathlib import Path
from .config import Config
from .storage import ROOT,atomic_json,read,rows

class Manager:
    def __init__(self):self.lock=threading.Lock();self.children={}
    def listing(self):
        result=[]
        for directory in sorted((ROOT/'runs').glob('*'),reverse=True):
            state=read(directory/'status.json')
            if not state:continue
            if state.get('status') not in ('completed','failed','stopped') and state.get('pid'):
                pid=state['pid']
                command=subprocess.run(['ps','-p',str(pid),'-o','command='],capture_output=True,text=True).stdout
                if str(directory) not in command:
                    state.update(status='failed',phase='runner exited unexpectedly; saved checkpoints retained')
                    atomic_json(directory/'status.json',state)
            state.update(id=directory.name,config=read(directory/'config.json'),latest=read(directory/'latest.json'))
            state['slots']=[read(p) for p in sorted(directory.glob('worker-*/status.json'))]
            state['matchups']=self.matchups(directory)
            result.append(state)
        return result
    def matchups(self,directory):
        grouped={}
        for row in rows(directory/'matches.jsonl'):
            key=(row.get('opponent','unknown'),row.get('cpu_level',0))
            group=grouped.setdefault(key,dict(opponent=key[0],cpu=key[1],attempts=0,wins=0,timeouts=0,interruptions=0))
            group['attempts']+=1;group['wins']+=row['result']=='win';group['timeouts']+=row['result']=='timeout'
            group['interruptions']+=row['result'] in ('interrupted','setup_failed')
        return list(grouped.values())
    def start(self,request):
        with self.lock:
            mode=request.get('mode','train')
            if mode not in ('train','evaluate','watch','benchmark'):raise ValueError('Unknown job type')
            active=[s for s in self.listing() if s['status'] not in ('completed','failed','stopped')]
            if active and (mode=='benchmark' or any(s['mode']=='benchmark' for s in active)):raise ValueError('Finish or stop the active job before a benchmark')
            group='training' if mode=='train' else 'evaluation'
            if any(('training' if s['mode']=='train' else 'evaluation')==group for s in active):raise ValueError(f'A {group} job is already active')
            config=Config.load()
            allowed=('character','stage','workers','cpu_level','action_frames','total_steps','opponents','max_frames','curriculum')
            for key in allowed:
                if key in request:setattr(config,key,request[key])
            checkpoint=request.get('checkpoint') or None
            if checkpoint:
                checkpoint=Path(checkpoint).resolve()
                if not checkpoint.is_relative_to(ROOT/'runs') or not checkpoint.is_file() or checkpoint.suffix!='.pt':raise ValueError('Choose a saved V2 checkpoint')
            if mode in ('evaluate','watch'):
                if not checkpoint:raise ValueError('Choose a checkpoint first')
                config=replace(config,workers=1,base_port=config.base_port+32,speed=1. if mode=='watch' else 0.,curriculum=False)
            config.validate()
            if not config.disk_ok():raise ValueError('Free disk space is below the reserve')
            episodes=int(request.get('episodes',12))
            if not 1<=episodes<=500:raise ValueError('Evaluation episodes must be 1–500')
            ident=time.strftime('%Y%m%d-%H%M%S')+'-'+uuid.uuid4().hex[:6]
            directory=ROOT/'runs'/ident;directory.mkdir(parents=True)
            payload=dict(mode=mode,config=config.asdict(),checkpoint=str(checkpoint) if checkpoint else None,episodes=episodes)
            atomic_json(directory/'request.json',payload);atomic_json(directory/'config.json',config.asdict())
            with (directory/'runner.log').open('w') as log:
                process=subprocess.Popen([sys.executable,'-u','-m','melee_next.cli','_run',str(directory)],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            self.children[ident]=process
            atomic_json(directory/'status.json',dict(status='starting',phase='launching',mode=mode,pid=process.pid,started=time.time()))
            return {'id':ident}
    def control(self,ident,action):
        if action not in ('pause','resume','stop'):raise ValueError('Unknown control')
        if '/' in ident or '..' in ident:raise ValueError('Invalid run')
        directory=ROOT/'runs'/ident
        state=read(directory/'status.json')
        if not state:raise ValueError('Run not found')
        if action=='pause' and state['mode']!='train':raise ValueError('Only training can be paused')
        command=read(directory/'control.json')
        command.update(pause=action=='pause',stop=action=='stop')
        atomic_json(directory/'control.json',command)
        if action=='stop' and state.get('mode')=='benchmark' and state.get('pid'):
            os.kill(state['pid'],signal.SIGTERM)
        return command
    def checkpoints(self):
        result=[]
        for path in sorted((ROOT/'runs').glob('*/checkpoints/*.json'),reverse=True):
            row=read(path)
            if row:result.append(row)
        return result
