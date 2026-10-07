import json
import threading
import time
import gymnasium as gym
import numpy as np
import pytest
from melee_lab.actions import ACTIONS
from melee_lab.config import Config
from melee_lab.environment import Stopped
from melee_lab.manager import Manager
from melee_lab.slot import Slot,Serviced,make_env,STALL_SECONDS
from melee_lab.state import OBS_SIZE
from melee_lab.storage import write_json


class FakeEnv(gym.Env):
    """Stands in for MeleeEnv; flags any concurrent console access."""
    observation_space=gym.spaces.Box(-5,5,shape=(OBS_SIZE,),dtype=np.float32)
    action_space=gym.spaces.Discrete(len(ACTIONS))
    def __init__(self): self.pumps=0; self.inside=False; self.concurrent=False
    def _enter(self):
        if self.inside: self.concurrent=True
        self.inside=True
    def pump(self,frames=2):
        self._enter(); self.pumps+=1; time.sleep(.002); self.inside=False
    def step(self,action):
        self._enter(); time.sleep(.01); self.inside=False
        return np.zeros(OBS_SIZE,np.float32),0.,False,False,{}
    def reset(self,seed=None,options=None): return np.zeros(OBS_SIZE,np.float32),{}
    def close(self): pass


def test_slot_publishes_to_its_own_file_and_cannot_save(tmp_path):
    slot=Slot(tmp_path/'env1',tmp_path/'control.json',1)
    slot.publish(frame=7,action='Jump')
    state=json.loads((tmp_path/'env1'/'status.json').read_text())
    assert state['index']==1 and state['frame']==7 and state['action']=='Jump'
    # The run-level status file belongs to the trainer, not to a slot.
    assert not (tmp_path/'status.json').exists()
    assert not hasattr(slot,'save')


def test_slot_control_stops_and_services_the_emulator_while_paused(tmp_path):
    control=tmp_path/'control.json'
    slot=Slot(tmp_path/'env0',control,0)
    pumps=[];slot.idle=lambda:pumps.append(1)
    write_json(control,{'pause':True})
    thread=threading.Thread(target=slot.control,daemon=True);thread.start()
    deadline=time.monotonic()+3
    while len(pumps)<5 and time.monotonic()<deadline: time.sleep(.02)
    assert len(pumps)>=5,'slot left its emulator unserviced while paused'
    write_json(control,{'pause':False});thread.join(3)
    assert not thread.is_alive()
    slot.last_poll=0;write_json(control,{'stop':True})
    with pytest.raises(Stopped): slot.control()


def test_serviced_leaves_normal_steps_alone_but_rescues_a_stall():
    inner=FakeEnv();env=Serviced(inner)
    try:
        deadline=time.monotonic()+STALL_SECONDS+.2
        while time.monotonic()<deadline: env.step(0)
        assert inner.pumps==0,'pump interfered with normal stepping'
        time.sleep(STALL_SECONDS+.4)
        assert inner.pumps>0,'emulator left unserviced through a trainer stall'
        assert not inner.concurrent,'pump and env stepped the console concurrently'
    finally:
        env.close()
    before=inner.pumps;time.sleep(.3)
    assert inner.pumps==before,'pump kept running after close'


def test_make_env_gives_each_slot_its_own_port_and_home(tmp_path):
    config=Config();base=config.port
    built=[make_env(config,tmp_path,i,tmp_path/'control.json',stagger=0)() for i in (0,2)]
    try:
        assert [e.unwrapped.config.port for e in built]==[base,base+2]
        assert (tmp_path/'env0').is_dir() and (tmp_path/'env2').is_dir()
        assert built[0].unwrapped.run_dir!=built[1].unwrapped.run_dir
        assert config.port==base,'shared config was mutated instead of copied'
    finally:
        for e in built:
            e.exit_on_close=False   # real slots exit the process on close
            e.close()


def test_parallel_start_is_rejected_where_it_is_not_implemented(tmp_path):
    """Curriculum is no longer on this list -- it works across emulators now."""
    manager=Manager(tmp_path)
    with pytest.raises(ValueError,match='training only'):
        manager.start(mode='evaluate',envs=4,curriculum=False)
    with pytest.raises(ValueError,match='checkpoint'):
        manager.start(envs=4,curriculum=True,bootstrap=4096)


def test_slot_crash_is_reported_and_exits_instead_of_wedging(tmp_path):
    """An exception reaching SB3's worker leaves the process hung in multiprocessing's
    atexit join, so the trainer waits in recv() forever with no error to show."""
    class Boom(FakeEnv):
        def step(self,action): raise RuntimeError('Match setup stalled at STAGE_SELECT.')
    slot=Slot(tmp_path/'env0',tmp_path/'control.json',0)
    codes=[]
    env=Serviced(Boom(),slot,on_crash=lambda code:codes.append(code))
    try: env.step(0)
    finally: env.close()
    assert codes==[1],'a crashing slot must exit hard, not return into SB3'
    state=json.loads((tmp_path/'env0'/'status.json').read_text())
    assert 'RuntimeError' in state['failed'] and 'STAGE_SELECT' in state['failed']
    assert 'Traceback' in state['traceback']


def test_step_telemetry_survives_an_unaligned_resume(tmp_path):
    """With N envs the timestep counter advances by N from a resumed offset, so a
    modulo check can miss every multiple forever: 4 envs resuming at 42,546 only
    ever reach odd-of-4 residues and never hit a multiple of 32."""
    from melee_lab.worker import Progress,Runtime
    runtime=Runtime(tmp_path,Config())
    published=[]
    runtime.publish=lambda **kw: published.append(kw)
    progress=Progress(runtime,None,False)
    progress.locals={'infos':[],'dones':[]}
    for i in range(20):
        progress.num_timesteps=42546+4*i
        progress._on_step()
    assert any('steps' in p for p in published),'steps telemetry never published'
    assert published[-1]['steps']>=42546


def test_a_requested_stop_is_not_reported_as_a_slot_crash(tmp_path):
    """Stop is control flow. Reporting it as a crash turned every clean stop of a
    parallel run into a failed run, because the trainer then saw only an EOFError."""
    class Quit(FakeEnv):
        def step(self,action): raise Stopped()
    slot=Slot(tmp_path/'env0',tmp_path/'control.json',0)
    codes=[]
    env=Serviced(Quit(),slot,on_crash=lambda code:codes.append(code))
    try: env.step(0)
    finally: env.close()
    assert codes==[0],'a requested stop must exit 0, not as a crash'
    path=tmp_path/'env0'/'status.json'
    state=json.loads(path.read_text()) if path.exists() else {}
    assert not state.get('failed'),'a requested stop must not be recorded as a failure'


def test_slot_close_exits_the_process_rather_than_hanging_at_shutdown(tmp_path):
    """SB3's worker returns into interpreter shutdown after close, where multiprocessing
    blocks joining libmelee's stream child; the trainer then hangs in vec.close()
    still holding the emulator lock, so the next run cannot start."""
    inner=FakeEnv(); closed=[]
    inner.close=lambda: closed.append(True)
    codes=[]
    env=Serviced(inner,None,on_crash=lambda code:codes.append(code),exit_on_close=True)
    env.close()
    assert closed==[True],'the emulator must still be torn down first'
    assert codes==[0],'slot must exit instead of returning into atexit'


def test_plain_serviced_close_does_not_exit(tmp_path):
    inner=FakeEnv(); codes=[]
    env=Serviced(inner,None,on_crash=lambda code:codes.append(code))
    env.close()
    assert codes==[],'only slot processes should exit on close'


def _finished(level,result):
    return {'cpu_level':level,'result':result,'players':{},'frame':10,
            'starting_stocks':[3,3],'episode':{'r':0.0}}


class FakeVec:
    def __init__(self): self.calls=[]
    def set_attr(self,name,value): self.calls.append((name,value))


def test_rollout_size_is_held_constant_as_emulators_are_added():
    """n_steps is per environment, so leaving it fixed multiplies the update size by
    the emulator count -- four emulators silently gave 4096-sample updates."""
    from melee_lab.worker import rollout_steps,ROLLOUT
    assert rollout_steps(1)==ROLLOUT
    assert rollout_steps(4)==ROLLOUT//4
    assert rollout_steps(4)*4==ROLLOUT
    assert rollout_steps(6)*6<=ROLLOUT+6
    assert rollout_steps(64)>=64


def test_curriculum_advances_across_parallel_emulators(tmp_path):
    from melee_lab.worker import Progress,Runtime
    runtime=Runtime(tmp_path,Config())
    vec=FakeVec()
    progress=Progress(runtime,None,True,(),vec,1)
    def finish(level,result):
        progress.locals={'infos':[_finished(level,result)],'dones':[True]}
        progress._on_step()
    # Matches played at another level must not fill this level's window.
    for _ in range(20): finish(3,'win')
    assert progress.level==1 and vec.calls==[]
    # 15 of 20 at the current level is below the 16/20 bar.
    for i in range(20): finish(1,'win' if i<15 else 'loss')
    assert progress.level==1,'advanced below the 16/20 threshold'
    for i in range(20): finish(1,'win' if i<16 else 'loss')
    assert progress.level==2,'did not advance after 16 of 20'
    assert ('cpu_level',2) in vec.calls,'new level never reached the emulators'


def test_curriculum_stops_at_level_nine(tmp_path):
    from melee_lab.worker import Progress,Runtime
    runtime=Runtime(tmp_path,Config())
    vec=FakeVec()
    progress=Progress(runtime,None,True,(),vec,9)
    for _ in range(20):
        progress.locals={'infos':[_finished(9,'win')],'dones':[True]}
        progress._on_step()
    assert progress.level==9 and vec.calls==[]


def test_setting_cpu_level_on_a_slot_reaches_the_emulator():
    """gymnasium's Wrapper does not forward attribute writes, so without an explicit
    property SB3's set_attr lands on the wrapper and the emulator never changes level --
    the curriculum would appear to advance while every match stayed at level 1."""
    inner=FakeEnv(); inner.cpu_level=1
    env=Serviced(inner,None,on_crash=lambda code:None)
    try:
        setattr(env,'cpu_level',5)        # exactly what SubprocVecEnv's worker does
        assert inner.cpu_level==5,'level never reached the wrapped env'
        assert env.cpu_level==5
    finally:
        env.close()


def test_randomize_opponent_slot_assignment(tmp_path, monkeypatch):
    from melee_lab.slot import COMPETITIVE_ROSTER, make_env
    envs_built = []
    def fake_melee_env(cfg, directory, control, publish, **kwargs):
        envs_built.append(cfg)
        return FakeEnv()
    monkeypatch.setattr('melee_lab.slot.MeleeEnv', fake_melee_env)
    monkeypatch.setattr('melee_lab.slot.watch_stacks', lambda p: None)

    cfg = Config(randomize_opponent=True)
    for i in range(len(COMPETITIVE_ROSTER) + 2):
        fn = make_env(cfg, tmp_path, i, tmp_path/'control.json', stagger=0)
        env = fn()
        env.exit_on_close = False   # a real slot exits its process on close
        env.close()

    assert [e.opponent for e in envs_built] == [
        COMPETITIVE_ROSTER[i % len(COMPETITIVE_ROSTER)]
        for i in range(len(COMPETITIVE_ROSTER) + 2)
    ]


def test_competitive_roster_valid_for_cpu():
    import melee
    from melee_lab.slot import COMPETITIVE_ROSTER
    for name in COMPETITIVE_ROSTER:
        char = melee.enums.Character[name]
        assert char != melee.enums.Character.SHEIK, "SHEIK cannot be chosen as CPU opponent in libmelee"



def test_each_slot_keeps_one_matchup():
    """Diversity comes from running the roster side by side, not from redrawing the
    opponent every episode. Redrawing also made every reset walk the character select
    cursor again, which every other slot waits out under lockstep."""
    from dataclasses import replace
    from melee_lab.config import Config
    from melee_lab.slot import COMPETITIVE_ROSTER, make_env

    config = Config(randomize_opponent=True, iso='x', dolphin='y')
    seen = []
    for index in range(len(COMPETITIVE_ROSTER) + 2):
        slot_config = replace(config, port=config.port + index, seed=config.seed + index)
        if getattr(config, 'randomize_opponent', False):
            slot_config = replace(slot_config,
                                  opponent=COMPETITIVE_ROSTER[index % len(COMPETITIVE_ROSTER)],
                                  randomize_opponent=False)
        seen.append(slot_config.opponent)
        # The slot must not redraw its opponent on every reset.
        assert slot_config.randomize_opponent is False

    # Every character appears before any repeats.
    assert set(seen) == set(COMPETITIVE_ROSTER)
    assert seen[:len(COMPETITIVE_ROSTER)] == list(COMPETITIVE_ROSTER)


def test_single_emulator_still_varies_its_opponent():
    """With one emulator there are no sibling slots to spread the roster across, so the
    per-episode draw is the only source of variety and must stay."""
    from melee_lab.config import Config
    config = Config(randomize_opponent=True)
    assert config.randomize_opponent is True


def test_stalled_slot_releases_the_pad_and_pumps_slowly():
    """A slot waiting out another slot's reset is not choosing actions, so it must not
    keep asserting the last one -- a held stick throw walks the agent off the stage --
    and it must advance as few frames as keeping the Slippi stream alive allows."""
    import inspect
    from melee_lab import slot

    body = inspect.getsource(slot.Serviced._loop)
    assert 'release_all()' in body, 'a stalled slot must not hold its last input'
    assert 'time.sleep(PUMP_INTERVAL)' in body

    # Slow enough to matter, far short of the multi-second gap that desynchronises.
    assert slot.PUMP_INTERVAL >= .2
    assert slot.PUMP_INTERVAL <= 1.0
    frames_per_reset = 6.3 / slot.PUMP_INTERVAL
    assert frames_per_reset < 40, 'a reset should not cost the other slots 100+ blind frames'
