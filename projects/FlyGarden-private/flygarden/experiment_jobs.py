"""Owned experiment identity survives restarting the HTTP application."""
import json,time
import psutil
from .storage import ROOT

def records():
    folder=ROOT/'.runtime/jobs';folder.mkdir(parents=True,exist_ok=True);out=[]
    for p in folder.glob('*.json'):
        try:out.append(json.loads(p.read_text()))
        except (ValueError,OSError):continue
    return out

def alive(r):
    try:
        p=psutil.Process(r['pid'])
        return p.is_running() and abs(p.create_time()-r['created'])<.1 and str(ROOT/'scripts/run_experiment.py') in p.cmdline() and p.status()!=psutil.STATUS_ZOMBIE
    except (psutil.Error,KeyError):return False

def active():return any(alive(r) for r in records())

def record_start(identity,protocol,pid):
    r=dict(id=identity,protocol=protocol,pid=pid,created=psutil.Process(pid).create_time(),started=time.time())
    folder=ROOT/'.runtime/jobs';folder.mkdir(parents=True,exist_ok=True);(folder/f'{identity}.json').write_text(json.dumps(r))

def statuses():
    out=[]
    for r in sorted(records(),key=lambda x:x['started'],reverse=True)[:20]:
        folder=ROOT/'data/experiments'/r['id'];completion=folder/'completion.json';result=json.loads(completion.read_text()) if completion.exists() else {};running=alive(r);status='running' if running else result.get('status','interrupted');log=ROOT/'reports'/f"job-{r['id']}.log"
        out.append(dict(id=r['id'],protocol=r['protocol'],status=status,exit_code=result.get('exit_code'),tail=log.read_text()[-2500:] if log.exists() else ''))
    return out
