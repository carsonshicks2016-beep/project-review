import math
import signal
import time
from pathlib import Path
import torch
from .game import game_class,Cancelled
from .engine import lock_ports
from .model import Policy,seed_all,load
from .storage import atomic_json,append,read,digest

def wilson(wins,n):
    if not n:return [0.,1.]
    z=1.96;p=wins/n;den=1+z*z/n
    mid=(p+z*z/(2*n))/den;half=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
    return [max(0,mid-half),min(1,mid+half)]

def evaluate(config,directory,checkpoint,episodes=12,watch=False):
    config.validate();seed_all(config.seed);directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    identity={'path':str(Path(checkpoint).resolve()),'sha256':digest(checkpoint)}
    atomic_json(directory/'config.json',config.asdict());atomic_json(directory/'checkpoint.json',identity)
    state=dict(status='starting',mode='watch' if watch else 'evaluate',backend=config.backend,started=time.time(),checkpoint=identity,
               requested_episodes=episodes,deterministic=True,execution='raw',results=[])
    def publish(**kw):state.update(kw);atomic_json(directory/'status.json',state)
    stopped=False;last_telemetry=0.
    def stop(*_):
        nonlocal stopped
        stopped=True
    old={s:signal.signal(s,stop) for s in (signal.SIGINT,signal.SIGTERM)}
    def control():
        if stopped or read(directory/'control.json').get('stop'):raise Cancelled()
    def telemetry(**kw):
        nonlocal last_telemetry
        if time.monotonic()-last_telemetry>.5:publish(**kw);last_telemetry=time.monotonic()
    game=None;locks=[]
    try:
        locks=lock_ports(config);policy=Policy().eval();progress=load(checkpoint,policy,config)
        publish(status='running',steps=progress['steps'])
        game=game_class(config)(config,directory/'worker-00',0,control,telemetry)
        for index in range(episodes):
            opponent=config.opponents[index%len(config.opponents)];begun=False
            try:
                control();obs=game.reset(opponent,config.cpu_level);begun=True
                while True:
                    with torch.inference_mode():action,_,_=policy.act(torch.from_numpy(obs).unsqueeze(0),deterministic=True)
                    obs,reward,terminal,truncated,info=game.step(action[0].numpy())
                    if terminal or truncated:
                        row=dict(info,checkpoint_sha256=identity['sha256'],episode=index,opponent=opponent,cpu_level=config.cpu_level);break
            except Cancelled:
                row=dict(result='interrupted',opponent=opponent,cpu_level=config.cpu_level,episode=index,reason='stopped',checkpoint_sha256=identity['sha256'])
                stopped=True
            except Exception as exc:
                row=dict(result='interrupted' if begun else 'setup_failed',opponent=opponent,cpu_level=config.cpu_level,
                         episode=index,error=str(exc),checkpoint_sha256=identity['sha256'])
                game.close()
            append(directory/'matches.jsonl',row);state['results'].append(row)
            publish(completed_episodes=len(state['results']))
            if stopped:break
        wins=sum(r['result']=='win' for r in state['results']);n=len(state['results'])
        matchups=[]
        for opponent in config.opponents:
            subset=[r for r in state['results'] if r['opponent']==opponent];w=sum(r['result']=='win' for r in subset)
            matchups.append(dict(opponent=opponent,attempts=len(subset),wins=w,win_rate=w/len(subset) if subset else None,interval=wilson(w,len(subset))))
        played=sum(r['result'] in ('win','loss','draw','timeout') for r in state['results'])
        publish(status='stopped' if stopped else 'completed' if played else 'failed',phase='evaluation saved',wins=wins,attempts=n,
                win_rate=wins/n if n else None,interval=wilson(wins,n),matchups=matchups,
                valid_melee_evidence=config.backend=='dolphin' and played>0,
                outcomes={result:sum(r['result']==result for r in state['results']) for result in ('win','loss','draw','timeout','interrupted','setup_failed')})
    except Exception as exc:publish(status='failed',error=str(exc))
    finally:
        if game:game.close()
        for lock in locks:lock.close()
        for s,handler in old.items():signal.signal(s,handler)
        publish(cleanup_complete=True,finished=time.time())
    return state
