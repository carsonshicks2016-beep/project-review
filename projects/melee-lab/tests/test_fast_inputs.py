import json
from dataclasses import replace
from types import SimpleNamespace
import gymnasium as gym
import numpy as np
import pytest
import torch
from melee_lab.actions import ACTIONS, EXTENDED_ACTIONS, apply, vocabulary, describe
from melee_lab.config import Config
from melee_lab.state import OBS_SIZE
from melee_lab.storage import save_model, write_json, validate_checkpoint
from melee_lab.transfer import transfer_policy
from melee_lab.worker import new_model
from melee_lab.environment import MeleeEnv
from test_learning import Controller,LearningFixture
from test_environment import env_with_frames
from test_learning import game
from test_dashboard import lab,checkpoint
from test_arena import fake_spawn


def test_expanded_contract_preserves_prefix_and_adds_composable_inputs():
    assert EXTENDED_ACTIONS[:len(ACTIONS)]==ACTIONS
    assert len({a.name for a in EXTENDED_ACTIONS})==len(EXTENDED_ACTIONS)
    assert len(EXTENDED_ACTIONS)>100
    names={a.name for a in EXTENDED_ACTIONS}
    assert {'Light shield','Spot dodge','Forward tilt left','Raw B / down','DI down-right'}<=names
    info=describe(Config(action_set='expanded',action_frames=1))
    assert info['decision_floor_ms']==pytest.approx(16.67) and info['max_decisions_per_game_second']==60


def test_frame_level_decision_does_not_force_four_frames(tmp_path):
    env=env_with_frames(tmp_path,[game(frame=1)])
    env.config.action_frames=1
    _,_,done,trunc,info=env.step(0)
    assert info['frame']==1 and not done and not trunc


def test_macro_commitment_is_explicit_even_at_one_frame(tmp_path):
    env=env_with_frames(tmp_path,[game(frame=i) for i in range(1,6)])
    env.config.action_frames=1
    assert env.step(29)[4]['frame']==5


def test_light_shield_and_raw_release():
    import melee
    class Pad(Controller):
        def __init__(self): super().__init__();self.shoulders={}
        def press_shoulder(self,b,v): self.shoulders[b]=v
    c=Pad();idx=next(i for i,a in enumerate(EXTENDED_ACTIONS) if a.name=='Light shield')
    apply(c,idx,0,EXTENDED_ACTIONS)
    assert c.shoulders[melee.Button.BUTTON_L]==.35
    raw=next(i for i,a in enumerate(EXTENDED_ACTIONS) if a.name=='Raw Y / left')
    apply(c,raw,0,EXTENDED_ACTIONS)
    assert melee.Button.BUTTON_Y in c.buttons
    apply(c,0,0,EXTENDED_ACTIONS)
    assert not c.buttons


def test_explicit_weight_transfer_preserves_backbone_and_old_action_rows(tmp_path):
    torch.set_num_threads(1)
    legacy=LearningFixture();source=new_model(legacy,7)
    source.num_timesteps=123
    path=tmp_path/'old.zip';save_model(source,path,Config())
    class Expanded(LearningFixture):
        action_space=gym.spaces.Discrete(len(EXTENDED_ACTIONS))
        config=Config(action_set='expanded',action_frames=1)
    env=Expanded();target=transfer_policy(path,env,7)
    old=source.policy.state_dict();new=target.policy.state_dict()
    for k in old:
        if not k.startswith('action_net.'):
            assert torch.equal(old[k],new[k])
    assert torch.equal(old['action_net.weight'],new['action_net.weight'][:len(ACTIONS)])
    assert target.num_timesteps==123
    newpath=tmp_path/'new.zip';save_model(target,newpath,env.config)
    validate_checkpoint(newpath,env.config)
    with pytest.raises(ValueError,match='contract'):validate_checkpoint(newpath,Config())
    assert int(target.predict(np.zeros(OBS_SIZE),deterministic=True)[0])<len(EXTENDED_ACTIONS)


def test_fast_training_is_explicit_and_checkpoint_remains_immutable(lab,monkeypatch):
    manager,client=lab;path=checkpoint(manager.root);original=path.read_bytes();calls=fake_spawn(monkeypatch)
    result=client.post('/api/start',json={'checkpoint':'runs/source/latest.zip','fast_inputs':True})
    assert result.status_code==200,result.text
    directory=manager.runs/result.json()['id'];config=json.loads((directory/'config.json').read_text())
    assert config['action_frames']==1 and config['action_set']=='controller'
    assert path.read_bytes()==original
    assert 'melee_lab.supervisor' in calls[0] and '--fast-inputs' in calls[0]


def test_menu_failure_restarts_only_with_a_bounded_retry(tmp_path,monkeypatch):
    env=MeleeEnv(Config(),tmp_path);attempts=[];closed=[]
    def reset(**kw):
        attempts.append(1)
        if len(attempts)==1:raise TimeoutError('character select')
        return 'new match',{}
    env._reset_once=reset;env.close=lambda:closed.append(1)
    # Advance fake time through the retry backoff without blocking the test.
    ticks=iter(range(100));monkeypatch.setattr('melee_lab.environment.time.monotonic',lambda:next(ticks)*10)
    monkeypatch.setattr('melee_lab.environment.time.sleep',lambda n:None)
    assert env.reset()[0]=='new match' and len(attempts)==2 and closed==[1]
    assert env.recoveries==1


def test_supervisor_respects_remaining_budget_and_stop(tmp_path,monkeypatch):
    from melee_lab import supervisor
    write_json(tmp_path/'request.json',{'checkpoint_identity':{'steps':100}})
    calls=[]
    def popen(command):
        calls.append(command.copy())
        if len(calls)==1:
            (tmp_path/'latest.zip').write_bytes(b'fixture')
            write_json(tmp_path/'latest.json',{'steps':140})
            write_json(tmp_path/'status.json',{'status':'failed','error':'fixture failure'})
            return SimpleNamespace(wait=lambda:1)
        write_json(tmp_path/'status.json',{'status':'completed'})
        return SimpleNamespace(wait=lambda:0)
    monkeypatch.setattr(supervisor.subprocess,'Popen',popen)
    ticks=iter(range(100));monkeypatch.setattr(supervisor.time,'monotonic',lambda:next(ticks)*100)
    args=['--mode','train','--run-dir',str(tmp_path),'--steps','100','--checkpoint','source.zip','--fast-inputs']
    assert supervisor.supervise(args)==0
    assert calls[1][calls[1].index('--steps')+1]=='60'
    assert '--fast-inputs' not in calls[1]
    assert json.loads((tmp_path/'recovery.json').read_text())['attempts'][0]['steps']==140


def test_validate_checkpoint_backward_compat_and_error_messages(tmp_path):
    from melee_lab.controller import contract as controller_contract
    from melee_lab.state import SCHEMA, OBS_SIZE

    cfg = Config(character='JIGGLYPUFF', action_set='controller', stage='FINAL_DESTINATION', stocks=3, action_frames=1, opponent='FOX', randomize_opponent=True)

    # 1. Controller contract dict works
    cp_path = tmp_path / 'controller_valid.zip'
    meta = {
        'schema': SCHEMA,
        'observation_size': OBS_SIZE,
        'actions': controller_contract(),
        'config': {
            'character': 'JIGGLYPUFF',
            'stage': 'FINAL_DESTINATION',
            'stocks': 3,
            'action_frames': 1,
            'action_set': 'controller',
            'opponent': 'FOX',
            'randomize_opponent': True,
        }
    }
    write_json(cp_path.with_suffix('.json'), meta)
    assert validate_checkpoint(cp_path, cfg) == meta

    # 2. Backward-compatible 'actions': 'controller' string works
    cp_compat = tmp_path / 'controller_compat.zip'
    meta_compat = dict(meta, actions='controller')
    write_json(cp_compat.with_suffix('.json'), meta_compat)
    assert validate_checkpoint(cp_compat, cfg) == meta_compat

    # 3. Discrete actions with controller config gives informative error
    cp_discrete = tmp_path / 'discrete.zip'
    meta_discrete = dict(meta, actions=['Neutral', 'Attack'], config=dict(meta['config'], action_set='legacy'))
    write_json(cp_discrete.with_suffix('.json'), meta_discrete)
    with pytest.raises(ValueError, match="discrete 2-action space.*uses the 7-axis controller"):
        validate_checkpoint(cp_discrete, cfg)

    # 4. Character mismatch gives clear error
    cp_fox = tmp_path / 'fox.zip'
    meta_fox = dict(meta, config=dict(meta['config'], character='FOX'))
    write_json(cp_fox.with_suffix('.json'), meta_fox)
    with pytest.raises(ValueError, match=r"Checkpoint character \('FOX'\) does not match current configuration \('JIGGLYPUFF'\)"):
        validate_checkpoint(cp_fox, cfg)


def test_load_model_detects_recurrent_policy_from_zip():
    from melee_lab.storage import load_model, is_recurrent_checkpoint
    from pathlib import Path
    p = Path('runs/imported-20260916-214301-34afc/latest.zip')
    if p.exists():
        assert is_recurrent_checkpoint(p) is True
        model = load_model(p)
        assert type(model).__name__ == 'RecurrentPPO'
        assert type(model.policy).__name__ == 'RecurrentActorCriticPolicy'


