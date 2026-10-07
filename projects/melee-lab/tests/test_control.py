import json
import os
import threading
import time
from fastapi.testclient import TestClient
import pytest
from melee_lab.manager import Manager
from melee_lab.storage import write_json
from melee_lab.config import Config
from melee_lab.worker import Runtime
from melee_lab.environment import Stopped
from melee_lab.server import app


def test_pause_save_resume_stop_lifecycle(tmp_path):
    runtime=Runtime(tmp_path,Config());runtime.publish(status='running')
    saves=[];runtime.save=lambda *a:saves.append(True)
    path=tmp_path/'control.json'
    write_json(path,{'pause':True,'save':'once'})
    thread=threading.Thread(target=runtime.control);thread.start()
    deadline=time.monotonic()+3
    while runtime.state['status']!='paused' and time.monotonic()<deadline: time.sleep(.02)
    assert runtime.state['status']=='paused' and saves==[True]
    write_json(path,{'pause':False,'save':'once'});thread.join(3)
    assert not thread.is_alive() and runtime.state['status']=='running' and saves==[True]
    runtime.last_poll=0;write_json(path,{'stop':True})
    with pytest.raises(Stopped): runtime.control()


def test_duplicate_start_rejected_and_dead_worker_detected(tmp_path):
    manager=Manager(tmp_path);run=tmp_path/'runs/one';run.mkdir()
    write_json(run/'status.json',{'status':'running','pid':os.getpid()})
    with pytest.raises(ValueError,match='already active'): manager.start()
    write_json(run/'status.json',{'status':'running','pid':99999999})
    assert manager.active() is None
    assert manager.list_runs()[0]['status']=='failed'


def test_http_rejects_cross_origin_changes_and_invalid_counts():
    client=TestClient(app)
    assert client.post('/api/control/stop',headers={'Origin':'https://evil.test'}).status_code==403
    assert client.post('/api/start',json={'steps':-2}).status_code==422
    assert client.get('/',headers={'Host':'evil.test'}).status_code==403
    assert client.get('/').status_code==200


def test_pause_keeps_servicing_the_emulator(tmp_path):
    """A pause that stops calling console.step() desynchronises the Slippi stream,
    which killed a run mid-training before the idle hook existed."""
    runtime=Runtime(tmp_path,Config());runtime.publish(status='running')
    runtime.save=lambda *a:None
    pumps=[];runtime.idle=lambda:pumps.append(time.monotonic())
    path=tmp_path/'control.json'
    write_json(path,{'pause':True})
    thread=threading.Thread(target=runtime.control,daemon=True);thread.start()
    deadline=time.monotonic()+3
    while len(pumps)<5 and time.monotonic()<deadline: time.sleep(.02)
    assert len(pumps)>=5,'emulator left unserviced while paused'
    write_json(path,{'pause':False});thread.join(3)
    assert not thread.is_alive() and runtime.state['status']=='running'
