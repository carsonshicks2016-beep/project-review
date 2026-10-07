import argparse
import json
import os
import sys
import time
import uuid
from dataclasses import replace
from pathlib import Path
from .config import Config
from .storage import ROOT,read,atomic_json

def run_directory(prefix):
    return ROOT/'runs'/(time.strftime('%Y%m%d-%H%M%S')+'-'+prefix+'-'+uuid.uuid4().hex[:5])

def main():
    p=argparse.ArgumentParser(description='Melee Next — independent Jigglypuff actor/learner')
    sub=p.add_subparsers(dest='command',required=True)
    sub.add_parser('serve');sub.add_parser('doctor')
    for name in ('train','evaluate','watch','benchmark'):
        q=sub.add_parser(name);q.add_argument('--workers',type=int);q.add_argument('--checkpoint');q.add_argument('--steps',type=int)
        q.add_argument('--seconds',type=float);q.add_argument('--directory');q.add_argument('--synthetic',action='store_true')
        q.add_argument('--episodes',type=int,default=12);q.add_argument('--max-frames',type=int);q.add_argument('--cpu-level',type=int)
        q.add_argument('--counts',default='1,2,4');q.add_argument('--opponents');q.add_argument('--batch-steps',type=int);q.add_argument('--fragment',type=int)
    q=sub.add_parser('_run');q.add_argument('directory')
    args=p.parse_args()
    if args.command=='serve':
        import uvicorn
        uvicorn.run('melee_next.server:app',host='127.0.0.1',port=8776);return
    config=Config.load()
    if args.command=='doctor':
        import shutil
        config.validate();print(json.dumps(dict(configuration='valid',python=sys.executable,cpus=os.cpu_count(),
            free_gb=shutil.disk_usage(ROOT).free/1024**3,character=config.character,runtime_root=str(ROOT)),indent=2));return
    if args.command=='_run':
        directory=Path(args.directory);request=read(directory/'request.json');config=Config(**request['config']);mode=request['mode']
        checkpoint=request.get('checkpoint');episodes=request.get('episodes',12);seconds=None
    else:
        directory=Path(args.directory) if args.directory else run_directory(args.command);mode=args.command
        checkpoint=args.checkpoint;episodes=args.episodes;seconds=args.seconds
        for argument,field in (('workers','workers'),('steps','total_steps'),('max_frames','max_frames'),('cpu_level','cpu_level'),('batch_steps','batch_steps'),('fragment','fragment')):
            value=getattr(args,argument,None)
            if value is not None:setattr(config,field,value)
        if args.synthetic:config.backend='synthetic';config.min_disk_mb=64
        if args.opponents:config.opponents=args.opponents.split(',')
    print(f'Run: {directory}',flush=True)
    if mode=='train':
        from .engine import train
        result=train(config,directory,checkpoint,seconds)
    elif mode=='benchmark':
        from .benchmark import benchmark
        directory.mkdir(parents=True,exist_ok=True)
        atomic_json(directory/'status.json',dict(status='running',mode='benchmark',pid=os.getpid(),started=time.time()))
        counts=tuple(int(x) for x in getattr(args,'counts','1,2,4').split(','))
        result=benchmark(config,directory,counts,seconds or 45)
        atomic_json(directory/'status.json',dict(result,mode='benchmark',status='completed' if result['status']=='completed' else 'failed',cleanup_complete=True))
    else:
        if not checkpoint:p.error('Evaluation and watch require --checkpoint')
        from .evaluation import evaluate
        config=replace(config,workers=1,base_port=config.base_port+32,speed=1. if mode=='watch' else 0.,curriculum=False)
        result=evaluate(config,directory,checkpoint,episodes,mode=='watch')
    print(json.dumps({k:v for k,v in result.items() if k not in ('history','slots','results','traceback')},indent=2))
    if result['status']=='failed':raise SystemExit(1)

if __name__=='__main__':main()
