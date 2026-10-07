"""Challenge transport, opponent observations, and benchmark contracts."""
import json
import os
import time
from types import SimpleNamespace
import numpy as np
import pytest
from starlette.websockets import WebSocketDisconnect
from melee_lab import server
from melee_lab.config import Config
from melee_lab.storage import write_json
from melee_lab.human import HumanInput
from melee_lab.opponent import FrozenOpponent
from melee_lab.worker import Runtime, evaluate
from test_dashboard import lab, checkpoint
from test_learning import game, Controller
from test_environment import env_with_frames


def fake_spawn(monkeypatch):
    calls=[]
    def spawn(command,**kw):
        calls.append(command)
        return SimpleNamespace(pid=os.getpid(),poll=lambda:None)
    monkeypatch.setattr('melee_lab.manager.subprocess.Popen',spawn)
    return calls


def test_challenge_forces_realtime_and_freezes_policy(lab,monkeypatch):
    manager,client=lab;checkpoint(manager.root)
    calls=fake_spawn(monkeypatch)
    response=client.post('/api/start',json={'mode':'play','checkpoint':'runs/source/latest.zip','human_character':'MARTH','episodes':3})
    assert response.status_code==200,response.text
    directory=manager.runs/response.json()['id']
    config=json.loads((directory/'config.json').read_text())
    assert config['speed']==1 and config['opponent']=='MARTH'
    assert '--mode' in calls[0] and 'play' in calls[0]
    assert (directory/'input-policy.zip').exists()
    assert client.post('/api/control/save').status_code==400


@pytest.mark.parametrize('payload',[
    {'mode':'play'}, {'mode':'play','checkpoint':'x','envs':2},
    {'mode':'train','benchmark':True}, {'mode':'evaluate','checkpoint':'x','opponent_checkpoint':'x'},
    {'mode':'play','checkpoint':'x','human_character':'NOT_REAL'},
])
def test_invalid_modes_rejected_before_spawn(lab,payload):
    _,client=lab
    assert client.post('/api/start',json=payload).status_code==422


def test_frozen_opponent_is_snapshotted_and_curriculum_disabled(lab,monkeypatch):
    manager,client=lab;checkpoint(manager.root);fake_spawn(monkeypatch)
    response=client.post('/api/start',json={'checkpoint':'runs/source/latest.zip','opponent_checkpoint':'runs/source/latest.zip','envs':2})
    assert response.status_code==200,response.text
    directory=manager.runs/response.json()['id']
    request=json.loads((directory/'request.json').read_text())
    assert request['curriculum'] is False
    assert request['config']['opponent']=='FOX'
    assert (directory/'opponent-policy.zip').read_bytes()==(directory/'input-policy.zip').read_bytes()
    assert request['opponent_identity']['sha256']==request['checkpoint_identity']['sha256']


def test_benchmark_overrides_level_and_allows_roster(lab,monkeypatch):
    manager,client=lab;checkpoint(manager.root);calls=fake_spawn(monkeypatch)
    result=client.post('/api/start',json={'mode':'evaluate','checkpoint':'runs/source/latest.zip','benchmark':True,'level':2})
    assert result.status_code==200
    command=calls[0]
    assert '--benchmark' in command and command[command.index('--level')+1]=='9'


def test_swapped_opponent_perspective_does_not_mutate_game():
    g=game();g.players[1].position.x=-20;g.players[2].position.x=40
    view=FrozenOpponent.perspective(g)
    assert view.players[1] is g.players[2]
    assert view.players[2] is g.players[1]
    assert g.players[1].position.x==-20
    from melee_lab.state import encode,PLAYER_SIZE
    obs=encode(view)
    assert obs[0]==pytest.approx(.2) and obs[PLAYER_SIZE]==pytest.approx(-.1)


def test_human_disconnect_releases_every_button(tmp_path):
    import melee
    class Pad(Controller):
        def press_shoulder(self,b,v): pass
    path=tmp_path/'input.json';pad=Pad();human=HumanInput(path)
    write_json(path,{'buttons':['A'],'main':[1,.5],'received_at':time.time()})
    human.apply(pad)
    assert human.connected and melee.Button.BUTTON_A in pad.buttons
    write_json(path,{'buttons':['A'],'received_at':time.time()-1})
    human.apply(pad)
    assert not human.connected and not pad.buttons


def test_human_frame_validation_rejects_nonfinite_and_out_of_range():
    from pydantic import ValidationError
    for axes in ([float('nan'),.5],[2,.5],[-1,0]):
        with pytest.raises(ValidationError): server.HumanFrame(main=axes)


def test_input_socket_ownership_origin_and_disconnect(lab):
    manager,client=lab;directory=manager.runs/'play';directory.mkdir()
    write_json(directory/'status.json',{'status':'running','mode':'play','pid':os.getpid()})
    with pytest.raises(WebSocketDisconnect):
        with client.websocket_connect('/api/play/play/input',headers={'Origin':'https://evil.test'}): pass
    with client.websocket_connect('/api/play/play/input',headers={'Origin':'http://testserver'}) as socket:
        socket.send_json({'buttons':['A'],'main':[1,.5]})
        deadline=time.monotonic()+2
        while not (directory/'human-input.json').exists() and time.monotonic()<deadline: time.sleep(.01)
        data=json.loads((directory/'human-input.json').read_text())
        assert data['buttons']==['A'] and data['received_at']>0
        with pytest.raises(WebSocketDisconnect):
            with client.websocket_connect('/api/play/play/input',headers={'Origin':'http://testserver'}): pass
    assert json.loads((directory/'human-input.json').read_text())['received_at']==0
    assert 'play' not in server._input_owners


def test_human_input_is_applied_during_game_instead_of_cleared(tmp_path):
    env=env_with_frames(tmp_path,[game((3,0),frame=1)])
    calls=[];env.human_input=SimpleNamespace(apply=lambda c:calls.append(c),connected=True)
    env.step(0)
    assert calls==[env.controllers[1]]


def test_balanced_roster_report_and_human_evidence_separation(tmp_path):
    class Env:
        config=Config(cpu_level=9)
        def reset(self,**kw): return np.zeros(1),{}
        def step(self,action):
            return np.zeros(1),0,True,False,dict(result='win',players={},cpu_level=9,frame=1,starting_stocks=[3,3],episode={'r':1})
    model=SimpleNamespace(predict=lambda *a,**k:(0,None))
    env=Env();runtime=Runtime(tmp_path/'benchmark',env.config)
    evaluate(model,env,runtime,2,benchmark=True)
    report=json.loads((runtime.directory/'evaluation.json').read_text())
    assert report['episodes']==12 and len(report['matchups'])==6
    assert all(m['episodes']==2 for m in report['matchups'])
    runtime=Runtime(tmp_path/'human',env.config)
    evaluate(model,env,runtime,1,human=True)
    assert (runtime.directory/'challenge.json').exists()
    assert not (runtime.directory/'evaluation.json').exists()
    assert json.loads((runtime.directory/'matches.jsonl').read_text())['kind']=='human challenge'


def test_diagnostics_artifacts_and_traversal(lab):
    manager,client=lab;directory=manager.runs/'sample';directory.mkdir()
    (directory/'worker.log').write_text('sample log')
    (directory/'private.txt').write_text('not an artifact')
    result=client.get('/api/runs/sample/diagnostics').json()
    assert result['log']=='sample log'
    assert [a['name'] for a in result['artifacts']]==['worker.log']
    assert client.get('/api/runs/sample/artifact/worker.log').text=='sample log'
    assert client.get('/api/runs/sample/artifact/private.txt').status_code==404
    (directory/'escape.slp').symlink_to(manager.root/'config.local.json')
    assert client.get('/api/runs/sample/artifact/escape.slp').status_code==404


def test_readiness_never_promises_superhuman_from_training(lab):
    manager,client=lab;checkpoint(manager.root)
    readiness=client.get('/api/state').json()['readiness']
    assert readiness['label']=='Superhuman performance unproven'
    assert readiness['evidence']['evaluation_games']==0
    assert readiness['architecture']['recurrent'] is False
