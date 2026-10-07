import os
import queue
import time
import traceback
from pathlib import Path
import numpy as np
import torch
from .codec import OBS
from .game import Cancelled,game_class
from .model import Policy,seed_all
from .learning import advantages
from .storage import atomic_json,read,append

def actor_main(config,directory,index,output,stop):
    seed_all(config.seed+index*1009)
    directory=Path(directory);slot=directory/f'worker-{index:02d}';slot.mkdir(exist_ok=True)
    state=dict(index=index,pid=os.getpid(),phase='starting',collected=0,discarded_queue=0,restarts=0,updated=time.time())
    last_publish=0.;last_control=0.;control_cache={};version=-1;policy=Policy().eval();rng=np.random.default_rng(config.seed+index)
    def telemetry(**values):
        nonlocal last_publish
        state.update(values)
        if time.monotonic()-last_publish>.5 or values.get('phase') in ('failed','stopped'):
            state['updated']=time.time();atomic_json(slot/'status.json',state);last_publish=time.monotonic()
    def control():
        nonlocal last_control,control_cache
        if stop.is_set():raise Cancelled()
        if time.monotonic()-last_control>.2:
            control_cache=read(directory/'control.json');last_control=time.monotonic()
        if control_cache.get('stop') or control_cache.get('pause'):raise Cancelled()
    game=game_class(config)(config,slot,index,control,telemetry)
    obs=None;episode_open=False;episode_version=None;errors=0
    def record(result,**extra):
        append(slot/'matches.jsonl',dict(time=time.time(),worker=index,result=result,opponent=state.get('opponent'),
            cpu_level=state.get('cpu_level'),policy_version_start=episode_version,policy_version_end=version,
            kind='training' if config.backend=='dolphin' else 'synthetic',**extra))
    try:
        while not stop.is_set():
            command=read(directory/'control.json')
            if command.get('stop'):break
            if command.get('pause'):
                if episode_open:record('interrupted',reason='pause');episode_open=False
                game.close();obs=None;telemetry(phase='paused');time.sleep(.2);continue
            try:
                control_cache={};last_control=0
                weights=torch.load(directory/'broadcast.pt',map_location='cpu',weights_only=True)
                if weights['version']!=version:policy.load_state_dict(weights['model']);version=weights['version']
                if obs is None:
                    curriculum=read(directory/'curriculum.json')
                    opponent=str(rng.choice(curriculum['opponents'],p=curriculum['weights']));level=curriculum['level']
                    telemetry(opponent=opponent,cpu_level=level)
                    obs=game.reset(opponent,level);episode_open=True;episode_version=version
                batch={k:[] for k in ('obs','actions','logp','values','next_values','rewards','terminated','boundaries','discounts')}
                for _ in range(config.fragment):
                    control()
                    with torch.inference_mode():
                        action,logp,value=policy.act(torch.from_numpy(obs).unsqueeze(0))
                    next_obs,reward,terminal,truncated,info=game.step(action[0].numpy())
                    with torch.inference_mode():next_value=policy(torch.from_numpy(next_obs).unsqueeze(0))[1].item()
                    for key,item in dict(obs=obs,actions=action[0].numpy(),logp=logp.item(),values=value.item(),next_values=next_value,
                        rewards=reward,terminated=terminal,boundaries=terminal or truncated,discounts=config.gamma_frame**info['frames']).items():batch[key].append(item)
                    obs=next_obs;state['collected']+=1
                    if terminal or truncated:
                        record(info['result'],players=info['players'],episode_frames=info['episode_frames'],episode_return=info['episode_return'])
                        episode_open=False;obs=None
                        break  # Send partial fragments before reset; reset never holds up the learner.
                batch={key:np.asarray(value,dtype=np.int64 if key=='actions' else np.float32) for key,value in batch.items()}
                adv,returns=advantages(batch['rewards'],batch['values'],batch['next_values'],batch['terminated'],batch['boundaries'],batch['discounts'],config.gae_lambda)
                batch.update(advantages=adv,returns=returns,version=version,worker=index)
                try:output.put_nowait(batch)
                except queue.Full:state['discarded_queue']+=len(adv)
                errors=0;telemetry(policy_version=version,collected=state['collected'])
            except Cancelled:
                if episode_open:record('interrupted',reason='control');episode_open=False
                game.close();obs=None
                if stop.is_set() or read(directory/'control.json').get('stop'):break
            except Exception as exc:
                record('interrupted' if episode_open else 'setup_failed',reason=str(exc));episode_open=False
                errors+=1;state['restarts']+=1;telemetry(phase='recovering',error=str(exc))
                append(slot/'errors.jsonl',dict(time=time.time(),error=str(exc),traceback=traceback.format_exc()))
                game.close();obs=None
                if errors>=3:
                    telemetry(phase='failed',error=f'Three consecutive failures: {exc}');return
                stop.wait(min(2*errors,6))
    finally:
        if episode_open:record('interrupted',reason='shutdown')
        game.close();telemetry(phase='stopped' if state['phase']!='failed' else 'failed')
        output.cancel_join_thread()
