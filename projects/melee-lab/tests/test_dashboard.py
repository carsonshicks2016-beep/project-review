"""Dashboard contracts and evidence integrity. Fixtures are not game results."""
import fcntl
import json
import os
import shutil
import time
import zipfile
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
import pytest
from fastapi.testclient import TestClient
from melee_lab import server
from melee_lab.config import Config
from melee_lab.manager import Manager
from melee_lab.storage import write_json
from melee_lab.state import SCHEMA, OBS_SIZE
from melee_lab.actions import ACTIONS


@pytest.fixture
def lab(tmp_path, monkeypatch):
    manager=Manager(tmp_path)
    monkeypatch.setattr(server,'manager',manager)
    monkeypatch.setattr(Config,'validate',lambda self:None)
    Config().save(tmp_path/'config.local.json')
    return manager,TestClient(server.app)


def checkpoint(root,run='source',name='latest',steps=100,extra=''):
    directory=root/'runs'/run;directory.mkdir(parents=True,exist_ok=True)
    path=directory/(name+'.zip')
    with zipfile.ZipFile(path,'w') as out:
        out.writestr('data',json.dumps({'num_timesteps':steps,'test_fixture':extra}))
    write_json(path.with_suffix('.json'),dict(schema=SCHEMA,observation_size=OBS_SIZE,
        actions=[a.name for a in ACTIONS],config=asdict(Config()),steps=steps,
        bootstrap_samples=20,training_cpu_level=1))
    return path


def row(result='win',kind='ppo',cpu=1):
    return dict(result=result,kind=kind,cpu_level=cpu,return_=1)


def write_matches(path,rows):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w') as out:
        for record in rows:
            record=dict(record);record['return']=record.pop('return_',0)
            out.write(json.dumps(record)+'\n')


def finished_evaluation(manager,source,run='eval',results=None,cpu=1):
    results=results or ['win','loss','timeout']
    directory=manager.runs/run;directory.mkdir(exist_ok=True)
    identity=manager.catalog.snapshot(source,directory/'input-policy.zip')
    config=asdict(Config());config['cpu_level']=cpu
    write_json(directory/'request.json',{'mode':'evaluate','checkpoint_identity':identity,'config':config})
    write_json(directory/'evaluation.json',dict(episodes=len(results),wins=results.count('win'),
        results=results,config=config,policy='deterministic neural network',scripted_overrides=False,
        three_stock_start_verified=True,win_rate=999,win_rate_95_interval=[999,999]))
    return directory


def test_all_slots_are_exposed_and_missing_failed_stale_are_distinct(lab):
    manager,client=lab;directory=manager.runs/'parallel';directory.mkdir()
    write_json(directory/'status.json',dict(status='running',pid=os.getpid(),envs=4))
    for i in (0,1,2):
        write_json(directory/f'env{i}'/'status.json',dict(index=i,frame=i*100,players={'1':{'stocks':3}},
            **({'failed':'TimeoutError: no frames','traceback':'fixture traceback'} if i==1 else {})))
    old=time.time()-30;os.utime(directory/'env2/status.json',(old,old))
    slots=client.get('/api/state').json()['runs'][0]['slots']
    assert [s['index'] for s in slots]==[0,1,2,3]
    assert slots[1]['failed'].startswith('TimeoutError') and slots[1]['traceback']=='fixture traceback'
    assert slots[2]['stale'] is True
    assert slots[3]['waiting'] is True and slots[3]['age_seconds'] is None


def test_single_emulator_falls_back_to_run_telemetry(lab):
    manager,client=lab;directory=manager.runs/'single';directory.mkdir()
    write_json(directory/'status.json',dict(status='stopped',frame=31,players={'1':{'stocks':2}}))
    result=client.get('/api/state?run_id=single').json()['runs'][0]
    assert result['slots'][0]['frame']==31 and result['slots'][0]['players']['1']['stocks']==2


def test_match_filters_keep_demo_timeout_and_interruption_honest(lab):
    manager,client=lab
    rows=[row('win','scripted demonstration')]*100+[row('loss')]*2+[row('timeout'),row('interrupted'),row('win')]+[row('win',cpu=2)]
    write_matches(manager.runs/'history/matches.jsonl',rows)
    body=client.get('/api/runs/history/matches?kind=ppo&cpu=1&limit=2').json()
    stats=body['groups'][0]
    assert body['total']==5 and len(body['rows'])==2
    assert stats['wins']==1 and stats['outcomes']['loss']==2
    assert stats['outcomes']['timeout']==1 and stats['outcomes']['interrupted']==1
    assert stats['win_rate']==.2 and len(stats['win_rate_95_interval'])==2
    filtered=client.get('/api/runs/history/matches?kind=ppo&cpu=1&outcome=timeout').json()
    assert filtered['total']==1 and filtered['groups'][0]['episodes']==5
    all_groups=client.get('/api/runs/history/matches?kind=all').json()['groups']
    assert len(all_groups)==3


def test_match_reader_incremental_and_partial_lines(lab,monkeypatch):
    manager,client=lab;path=manager.runs/'history/matches.jsonl'
    write_matches(path,[row()])
    client.get('/api/runs/history/matches')
    original=Path.open
    def no_reread(self,*args,**kwargs):
        if self==path: raise AssertionError('Unchanged match log reread')
        return original(self,*args,**kwargs)
    with monkeypatch.context() as m:
        m.setattr(Path,'open',no_reread)
        assert client.get('/api/runs/history/matches').json()['total']==1
    with path.open('a') as out:out.write('{"result":"timeout","kind":"ppo"')
    assert client.get('/api/runs/history/matches').json()['total']==1
    with path.open('a') as out:out.write(',"cpu_level":1}\n')
    assert client.get('/api/runs/history/matches').json()['total']==2


@pytest.mark.parametrize('body',[
    {'envs':4,'curriculum':False,'bootstrap':0},
    {'envs':4,'mode':'evaluate','curriculum':False,'checkpoint':'runs/x/latest.zip'},
    {'envs':13}, {'envs':0}, {'mode':'evaluate'},
])
def test_parallel_combinations_fail_before_spawning(lab,body):
    _,client=lab
    assert client.post('/api/start',json=body).status_code==422


def test_lock_teardown_is_retryable_and_recovers_without_a_failed_run(lab):
    manager,client=lab;directory=manager.root/'.runtime';directory.mkdir()
    with (directory/'emulator.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        assert client.get('/api/state').json()['availability']['reason']=='teardown'
        response=client.post('/api/start',json={})
        assert response.status_code==409 and response.json()['detail']['retryable'] is True
        assert list(manager.runs.iterdir())==[]
    assert client.get('/api/state').json()['availability']['can_start'] is True


def test_start_snapshots_checkpoint_and_records_exact_source(lab,monkeypatch):
    manager,client=lab;source=checkpoint(manager.root)
    calls=[]
    def popen(command,**kwargs):
        calls.append(command)
        return SimpleNamespace(pid=os.getpid(),poll=lambda:None)
    monkeypatch.setattr('melee_lab.manager.subprocess.Popen',popen)
    response=client.post('/api/start',json={'mode':'evaluate','checkpoint':'runs/source/latest.zip','episodes':3})
    assert response.status_code==200
    directory=manager.runs/response.json()['id']
    request=json.loads((directory/'request.json').read_text())
    assert request['checkpoint_identity']['source']=='runs/source/latest.zip'
    assert str(directory/'input-policy.zip') in calls[0]
    original=(directory/'input-policy.zip').read_bytes()
    checkpoint(manager.root,steps=200)
    assert original==(directory/'input-policy.zip').read_bytes()!=source.read_bytes()


def test_evidence_attaches_to_exact_bytes_and_survives_run_cleanup(lab):
    manager,client=lab;source=checkpoint(manager.root)
    evaluation=finished_evaluation(manager,source)
    state=client.get('/api/state').json()
    original=next(c for c in state['checkpoints'] if c['path']=='runs/source/latest.zip')
    assert original['evaluations'][0]['win_rate']==pytest.approx(1/3)
    assert original['evaluations'][0]['outcomes']['timeout']==1
    assert original['evaluations'][0]['win_rate_95_interval'][1]<=1
    response=client.post('/api/champions',json={'checkpoint':original['path'],'name':'First candidate','note':'Fixture note'})
    assert response.status_code==200
    promoted=manager.root/'champions'/response.json()['id']/'policy.zip'
    assert promoted.read_bytes()==source.read_bytes()
    checkpoint(manager.root,steps=200)
    current=next(c for c in client.get('/api/state').json()['checkpoints'] if c['path']==original['path'])
    assert current['evaluations']==[], 'New latest.zip inherited an older model score'
    shutil.rmtree(source.parent);shutil.rmtree(evaluation)
    champion=client.get('/api/champions').json()['champions'][0]
    assert promoted.is_file() and champion['evaluations'][0]['episodes']==3
    assert manager.checkpoint(champion['path'],Config())==promoted


def test_unevaluated_champion_has_no_training_score(lab):
    manager,client=lab;source=checkpoint(manager.root)
    write_matches(source.parent/'matches.jsonl',[row()]*50)
    response=client.post('/api/champions',json={'checkpoint':'runs/source/latest.zip','name':'Unscored','note':''})
    assert response.status_code==200
    champion=client.get('/api/champions').json()['champions'][0]
    assert champion['evaluations']==[]


def test_champion_evaluation_can_be_launched_with_an_immutable_input(lab,monkeypatch):
    manager,client=lab;source=checkpoint(manager.root)
    entry=client.post('/api/champions',json={'checkpoint':'runs/source/latest.zip','name':'Candidate'}).json()
    champion_path=f"champions/{entry['id']}/policy.zip"
    monkeypatch.setattr('melee_lab.manager.subprocess.Popen',lambda *a,**kw:SimpleNamespace(pid=os.getpid(),poll=lambda:None))
    response=client.post('/api/start',json={'mode':'evaluate','checkpoint':champion_path,'episodes':10})
    assert response.status_code==200
    request=json.loads((manager.runs/response.json()['id']/'request.json').read_text())
    assert request['checkpoint_identity']['source']==champion_path


def test_unlinked_or_scripted_reports_never_become_evaluation_evidence(lab):
    manager,client=lab;source=checkpoint(manager.root)
    evaluation=finished_evaluation(manager,source)
    report=json.loads((evaluation/'evaluation.json').read_text());report['scripted_overrides']=True
    write_json(evaluation/'evaluation.json',report)
    assert not any(c['evaluations'] for c in client.get('/api/state').json()['checkpoints'])
    report['scripted_overrides']=False;write_json(evaluation/'evaluation.json',report)
    (evaluation/'request.json').unlink()
    assert not any(c['evaluations'] for c in client.get('/api/state').json()['checkpoints'])


def test_promotion_detects_mismatched_zip_and_metadata(lab):
    manager,client=lab;path=checkpoint(manager.root)
    metadata=json.loads(path.with_suffix('.json').read_text());metadata['steps']=200
    write_json(path.with_suffix('.json'),metadata)
    response=client.post('/api/champions',json={'checkpoint':'runs/source/latest.zip','name':'Do not copy'})
    assert response.status_code==400
    assert not list((manager.root/'champions').glob('*/policy.zip'))


def test_new_endpoints_preserve_local_access_and_reject_escape(lab,tmp_path):
    manager,client=lab;checkpoint(manager.root)
    body={'checkpoint':'runs/source/latest.zip','name':'Candidate'}
    assert client.post('/api/champions',json=body,headers={'Origin':'https://elsewhere.test'}).status_code==403
    assert client.get('/api/champions',headers={'Host':'elsewhere.test'}).status_code==403
    assert client.post('/api/champions',json={**body,'checkpoint':'../../elsewhere.zip'}).status_code==400
    assert client.post('/api/champions',json={**body,'name':'  '}).status_code==400
    assert client.get('/api/runs/missing/matches').status_code==404
    assert client.get('/api/runs/source/matches?limit=500').status_code==422


def test_settings_randomize_opponent(lab):
    manager,client=lab
    res=client.post('/api/config',json={
        'iso':'fake_iso',
        'dolphin':'fake_dolphin',
        'cpu_level':3,
        'speed':2.0,
        'randomize_opponent':True
    })
    assert res.status_code==200
    state=client.get('/api/state').json()
    assert state['config']['randomize_opponent'] is True


def test_validate_checkpoint_randomize_opponent(tmp_path):
    from melee_lab.storage import validate_checkpoint
    path=checkpoint(tmp_path)
    # Checkpoint was created with opponent='MARIO'
    # Strict check raises ValueError if opponent differs
    strict_cfg=Config(opponent='FOX',randomize_opponent=False)
    with pytest.raises(ValueError,match='opponent'):
        validate_checkpoint(path,strict_cfg)

    # With randomize_opponent=True, opponent mismatch is allowed
    random_cfg=Config(opponent='FOX',randomize_opponent=True)
    meta=validate_checkpoint(path,random_cfg)
    assert meta['config']['character']=='FOX'


def test_resume_jigglypuff_parallel(lab, monkeypatch):
    manager, client = lab
    run_dir = manager.root / 'runs' / 'puff_source'
    run_dir.mkdir(parents=True, exist_ok=True)
    zip_path = run_dir / 'latest.zip'
    with zipfile.ZipFile(zip_path, 'w') as out:
        out.writestr('data', json.dumps({'num_timesteps': 250}))
    cfg = asdict(Config())
    cfg['character'] = 'JIGGLYPUFF'
    write_json(zip_path.with_suffix('.json'), dict(
        schema=SCHEMA,
        observation_size=OBS_SIZE,
        actions=[a.name for a in ACTIONS],
        config=cfg,
        steps=250,
        bootstrap_samples=50,
        training_cpu_level=1,
        architecture='recurrent'
    ))

    calls = []
    def popen(command, **kwargs):
        calls.append(command)
        return SimpleNamespace(pid=os.getpid(), poll=lambda: None)
    monkeypatch.setattr('melee_lab.manager.subprocess.Popen', popen)

    state = client.get('/api/state').json()
    puff_cp = next(c for c in state['checkpoints'] if c['path'] == 'runs/puff_source/latest.zip')
    assert puff_cp['config']['character'] == 'JIGGLYPUFF'
    assert puff_cp['architecture'] == 'recurrent'

    res = client.post('/api/start', json={
        'mode': 'train',
        'checkpoint': 'runs/puff_source/latest.zip',
        'envs': 4,
        'recurrent': True,
    })
    assert res.status_code == 200, res.text
    run_id = res.json()['id']
    req = json.loads((manager.runs / run_id / 'request.json').read_text())
    assert req['config']['character'] == 'JIGGLYPUFF'
    assert req['envs'] == 4
    assert req['checkpoint'] == 'runs/puff_source/latest.zip'

