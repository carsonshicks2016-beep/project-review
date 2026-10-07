from dataclasses import replace
from types import SimpleNamespace as N
from pathlib import Path
import json
import time
import numpy as np
import pytest
import torch
from melee_next.codec import Reward,AXES,OBS,SCHEMA,polar,History
from melee_next.config import Config
from melee_next.curriculum import Curriculum
from melee_next.learning import advantages,update
from melee_next.model import Policy,save,load,seed_all
from melee_next.engine import train
from melee_next.evaluation import evaluate,wilson
from melee_next.storage import read,digest

@pytest.fixture
def config():return Config(backend='synthetic',base_port=54441,min_disk_mb=64,workers=1,fragment=32,batch_steps=64,total_steps=128,epochs=1,minibatch=32,opponents=['FOX'],mock_delay=.001)

def test_terminal_does_not_bootstrap():
    a,r=advantages(np.array([1.]),np.array([2.]),np.array([99.]),[True],[True],[.99])
    assert a[0]==-1 and r[0]==1

def test_timeout_bootstraps_but_never_leaks_into_next_episode():
    a,r=advantages(np.array([1.,100.]),np.array([2.,0.]),np.array([4.,0.]),[False,True],[True,True],[.5,.5])
    assert a.tolist()==[1.,100.] and r.tolist()==[3.,100.]

def test_contiguous_gae():
    a,r=advantages(np.ones(3),np.zeros(3),np.zeros(3),[0,0,1],[0,0,1],[1,1,1],1.)
    assert a.tolist()==[3,2,1]

def test_reward_cannot_farm_infinite_damage():
    reward=Reward();total=0
    for percent in range(1000):
        previous=N(players={1:N(stock=3,percent=0),2:N(stock=3,percent=percent)})
        current=N(players={1:N(stock=3,percent=0),2:N(stock=3,percent=percent+1)})
        r,result=reward.step(previous,current);total+=r
    assert total==pytest.approx(2.)

def test_respawn_not_damage_credit():
    before=N(players={1:N(stock=3,percent=120),2:N(stock=1,percent=150)})
    after=N(players={1:N(stock=3,percent=120),2:N(stock=0,percent=0)})
    value,result=Reward().step(before,after)
    assert value==16 and result=='win'

def test_polar_range():
    for i in range(49):assert all(0<=x<=1 for x in polar(i))
    assert polar(0)==(.5,.5)

def test_checkpoint_contract_and_immutability(tmp_path,config):
    seed_all(3);model=Policy();optimizer=torch.optim.Adam(model.parameters());path=tmp_path/'one.pt'
    identity=save(path,model,optimizer,config,{'steps':128,'updates':1,'curriculum':Curriculum(config).state()})
    assert identity['sha256']==digest(path)
    assert load(path,Policy(),config)['steps']==128
    with pytest.raises(ValueError):load(path,Policy(),replace(config,character='FOX'))
    with pytest.raises(ValueError):load(path,Policy(),replace(config,action_frames=1))
    with pytest.raises(ValueError):save(path,model,optimizer,config,{})

def test_hard_opponents_keep_uniform_floor(config):
    config=replace(config,opponents=['FOX','FALCO']);curriculum=Curriculum(config)
    for _ in range(20):
        curriculum.record(dict(opponent='FOX',cpu_level=1,result='win'))
        curriculum.record(dict(opponent='FALCO',cpu_level=1,result='loss'))
    state=curriculum.state()
    assert state['level']==1 and state['weights'][1]>state['weights'][0]>=.25
    restored=Curriculum(config,state).state();assert restored==state

def test_promote_requires_every_matchup_and_counts_timeouts(config):
    config=replace(config,opponents=['FOX','FALCO']);c=Curriculum(config)
    for _ in range(10):c.record(dict(opponent='FOX',cpu_level=1,result='win'))
    assert c.level==1
    for _ in range(4):c.record(dict(opponent='FALCO',cpu_level=1,result='timeout'))
    for _ in range(6):c.record(dict(opponent='FALCO',cpu_level=1,result='win'))
    assert c.level==1
    for _ in range(10):c.record(dict(opponent='FALCO',cpu_level=1,result='win'))
    assert c.level==2

def test_frame_discount_config(config):
    assert config.gamma_frame**2 < config.gamma_frame
    with pytest.raises(ValueError):replace(config,workers=0).validate()
    with pytest.raises(ValueError):replace(config,cpu_level=0).validate()
    with pytest.raises(ValueError):replace(config,opponents=[]).validate()

def test_policy_joint_probability_and_finite_update(config):
    seed_all(1);model=Policy();x=torch.zeros(64,OBS)
    with torch.no_grad():a,logp,v=model.act(x)
    scored,entropy,v=model.score(x,a)
    assert torch.allclose(logp,scored) and torch.all(entropy>0)
    before=model.actor.weight.detach().clone()
    batch=dict(obs=x.numpy(),actions=a.numpy(),logp=logp.numpy(),advantages=np.linspace(-1,1,64,dtype=np.float32),returns=np.ones(64,np.float32))
    metrics=update(model,torch.optim.Adam(model.parameters(),lr=.001),[batch],config)
    assert np.isfinite(metrics['loss']) and not torch.equal(before,model.actor.weight)

def test_slow_reset_worker_does_not_block_learning(tmp_path,config):
    c=replace(config,workers=2,mock_slow_worker=1,mock_slow_seconds=30)
    result=train(c,tmp_path/'async')
    assert result['status']=='completed' and result['steps']==128
    assert read(tmp_path/'async/worker-01/status.json')['collected']==0
    assert read(tmp_path/'async/worker-00/status.json')['collected']>=128
    assert result['cleanup_complete']

def test_resume_preserves_steps_and_curriculum(tmp_path,config):
    first=train(config,tmp_path/'first');checkpoint=first['checkpoint']['path']
    second=train(config,tmp_path/'second',checkpoint)
    assert second['status']=='completed' and second['steps']==256
    assert second['updates']==first['updates']+2
    assert second['parent']['sha256']==digest(checkpoint)
    assert second['checkpoint']['curriculum']['history']['FOX'][:len(first['checkpoint']['curriculum']['history']['FOX'])]==first['checkpoint']['curriculum']['history']['FOX']

def test_evaluation_checkpoint_bound_and_synthetic_label(tmp_path,config):
    seed_all(1);path=tmp_path/'test.pt';identity=save(path,Policy(),None,config,{'steps':0,'updates':0})
    result=evaluate(config,tmp_path/'eval',path,2)
    assert result['attempts']==2 and result['wins']==0 and not result['valid_melee_evidence']
    assert all(r['checkpoint_sha256']==identity['sha256'] for r in result['results'])
    assert result['outcomes']['loss']==2

def test_wilson_not_perfect_on_small_sample():
    low,high=wilson(2,2);assert low<.5 and high==pytest.approx(1)


def test_api_local_origin_and_invalid_request():
    from fastapi.testclient import TestClient
    from melee_next.server import app
    client=TestClient(app)
    assert client.get('/').status_code==200
    assert client.post('/api/runs',json={'mode':'invalid'}).status_code==400
    assert client.post('/api/runs',json={'mode':'train'},headers={'Origin':'https://evil.example'}).status_code==403
    assert client.get('/api/state',headers={'Host':'evil.example'}).status_code==403
