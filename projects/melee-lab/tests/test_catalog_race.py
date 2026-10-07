"""The checkpoint listing runs while training is saving."""
import json
from pathlib import Path
from melee_lab.catalog import Catalog


def _checkpoint(directory, name, steps=100):
    directory.mkdir(parents=True, exist_ok=True)
    (directory/f'{name}.zip').write_bytes(b'not a real model, only needs to hash')
    (directory/f'{name}.json').write_text(json.dumps(
        {'schema': 'melee-lab-v1', 'steps': steps, 'config': {'cpu_level': 3}}))


def test_listing_survives_a_checkpoint_being_replaced(tmp_path, monkeypatch):
    """save_model writes latest.tmp.zip then renames it over latest.zip. The glob can
    catch the temp file and it is gone microseconds later; stat-ing it before the
    .tmp. filter applied returned a 500 from /api/state on every save."""
    _checkpoint(tmp_path/'runs'/'r1', 'latest')
    real_glob = Path.glob
    def glob(self, pattern, *a, **k):
        items = list(real_glob(self, pattern, *a, **k))
        if pattern == '*/*.zip':
            items.append(self/'r1'/'latest.tmp.zip')   # caught mid-rename, now gone
            items.append(self/'r1'/'vanished.zip')     # run directory deleted
        return iter(items)
    monkeypatch.setattr(Path, 'glob', glob)
    rows = Catalog(tmp_path).checkpoints()
    names = [r['path'] for r in rows]
    assert 'runs/r1/latest.zip' in names
    assert not any('.tmp.' in n or 'vanished' in n for n in names)


def test_listing_reports_real_checkpoints(tmp_path):
    _checkpoint(tmp_path/'runs'/'r1', 'latest', steps=2048)
    _checkpoint(tmp_path/'runs'/'r2', 'interrupted', steps=4096)
    rows = Catalog(tmp_path).checkpoints()
    assert {r['steps'] for r in rows} == {2048, 4096}
    assert all(r['updated'] > 0 for r in rows)


class _Remote:
    def __init__(self, dead=False): self.sent=[]; self.dead=dead; self.closed=False
    def send(self, message):
        if self.dead: raise BrokenPipeError('slot already exited')
        self.sent.append(message)
    def recv(self): raise AssertionError('shutdown must never wait on a slot reply')
    def close(self): self.closed=True


class _Process:
    def __init__(self, stubborn=False, pid=0):
        self.stubborn=stubborn; self.pid=pid
        self.joins=[]; self.terminated=False; self.killed=False
    def join(self, timeout=None):
        self.joins.append(timeout)
        assert timeout is not None, 'shutdown must join with a deadline'
    def is_alive(self): return self.stubborn and not self.killed
    def terminate(self): self.terminated=True
    def kill(self): self.killed=True


class _Vec:
    def __init__(self, remotes, processes):
        self.remotes=remotes; self.processes=processes
        self.waiting=True; self.closed=False


def test_shutdown_never_waits_on_a_slot_that_already_left():
    """A requested stop makes one slot exit mid-step, so the trainer's step_wait sees
    EOF with waiting still True. SB3's close() then drains the SURVIVING slots, which
    are blocked waiting for a command -- a mutual wait that stranded the emulators."""
    from melee_lab.slot import shutdown
    remotes=[_Remote(), _Remote(dead=True), _Remote()]
    processes=[_Process(pid=1), _Process(pid=2), _Process(stubborn=True, pid=3)]
    vec=_Vec(remotes, processes)
    forced=shutdown(vec, timeout=1)          # _Remote.recv would assert if called
    assert vec.waiting is False and vec.closed is True
    assert remotes[0].sent==[('close',None)] and remotes[2].sent==[('close',None)]
    assert all(r.closed for r in remotes)
    assert processes[2].terminated and processes[2].killed, 'stubborn slot not forced'
    assert not processes[0].terminated, 'a slot that exited cleanly was force-killed'
    assert forced==[3]


def test_shutdown_tolerates_no_vec():
    from melee_lab.slot import shutdown
    assert shutdown(None)==[]


def test_transfer_rebuilds_the_source_architecture(tmp_path, monkeypatch):
    """new_model's default width was widened after these checkpoints were trained.
    Building the transfer target from the default made every existing policy fail with
    'Policy architecture cannot be transferred to this controller.'"""
    import melee_lab.transfer as transfer
    seen = {}
    class _Policy:
        net_arch = {'pi': [128, 128], 'vf': [128, 128]}
        def state_dict(self): return {}
        def load_state_dict(self, state): pass
    class _Source:
        policy = _Policy(); num_timesteps = 4242
    class _Target:
        policy = _Policy(); num_timesteps = 0
    monkeypatch.setattr(transfer.PPO, 'load', staticmethod(lambda *a, **k: _Source()))
    monkeypatch.setattr(transfer, 'read_json', lambda p: {'config': {}})
    monkeypatch.setattr(transfer, 'Config', lambda **k: 'cfg')
    monkeypatch.setattr(transfer, 'vocabulary', lambda cfg: [])
    import melee_lab.worker as worker
    def fake_new_model(env, seed, n_envs=1, net_arch=None):
        seen['net_arch'] = net_arch
        return _Target()
    monkeypatch.setattr(worker, 'new_model', fake_new_model)
    class _Env:
        def get_attr(self, name): return ['cfg']
    result = transfer.transfer_policy('policy.zip', _Env(), seed=1, n_envs=2)
    assert seen['net_arch'] == {'pi': [128, 128], 'vf': [128, 128]}, \
        'transfer must rebuild the width the source was trained at'
    assert result.num_timesteps == 4242
