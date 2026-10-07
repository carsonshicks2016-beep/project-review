"""Bounded recovery for unattended training. Checkpoints and the original budget survive."""
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from .catalog import read_json
from .storage import write_json


def supervise(arguments):
    directory=Path(arguments[arguments.index('--run-dir')+1])
    request=read_json(directory/'request.json')
    initial=request.get('checkpoint_identity',{}).get('steps',0)
    target=initial+int(arguments[arguments.index('--steps')+1])
    attempts=[]
    for attempt in range(5):
        if read_json(directory/'control.json').get('stop'):
            state=read_json(directory/'status.json');state.update(status='stopped',phase='stopped before recovery')
            write_json(directory/'status.json',state);return 0
        child=subprocess.Popen([sys.executable,'-u','-m','melee_lab.worker',*arguments])
        code=child.wait()
        state=read_json(directory/'status.json')
        if code==0 or state.get('status') in ('completed','stopped'): return code
        candidate=max((p for p in (directory/'interrupted.zip',directory/'latest.zip',directory/'controller-transfer.zip') if p.exists() and p.with_suffix('.json').exists()),key=lambda p:(read_json(p.with_suffix('.json')).get('steps',0),p.stat().st_mtime),default=None)
        # No usable checkpoint: retry startup with the original input. Never
        # recreate random weights halfway through a learned run.
        checkpoint_steps=read_json(candidate.with_suffix('.json')).get('steps',initial) if candidate else initial
        attempts.append(dict(attempt=attempt+1,time=time.time(),error=state.get('error'),checkpoint=str(candidate) if candidate else None,steps=checkpoint_steps))
        write_json(directory/'recovery.json',dict(attempts=attempts,max_retries=4))
        if attempt==4:
            state.update(status='failed',phase='recovery limit reached',error=f"Recovery limit reached. Last failure: {state.get('error','worker exited')}")
            write_json(directory/'status.json',state);return code or 1
        remaining=target-checkpoint_steps
        if remaining<=0:
            state.update(status='completed',phase='training budget reached; checkpoint preserved')
            write_json(directory/'status.json',state);return 0
        if candidate:
            if '--checkpoint' in arguments: arguments[arguments.index('--checkpoint')+1]=str(candidate)
            else: arguments+=['--checkpoint',str(candidate)]
            if '--fast-inputs' in arguments: arguments.remove('--fast-inputs')
        arguments[arguments.index('--steps')+1]=str(remaining)
        delay=min(30,5*(attempt+1))
        state.update(status='starting',phase=f'recovering worker · attempt {attempt+1}/4',pid=os.getpid(),recovery_attempt=attempt+1)
        write_json(directory/'status.json',state)
        deadline=time.monotonic()+delay
        while time.monotonic()<deadline:
            if read_json(directory/'control.json').get('stop'): break
            time.sleep(.2)
    return 1

if __name__=='__main__':
    raise SystemExit(supervise(sys.argv[1:]))
