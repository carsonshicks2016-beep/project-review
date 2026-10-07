"""The learner end to end on synthetic fragments: no emulators, no worker processes."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from puffbot import actions, observation
from puffbot.config import Config
from puffbot.learner import Learner
from puffbot.model import load_policy


def fragment(version, T=16, seed=0, worker=0):
    rng = np.random.default_rng(seed)
    masks = np.ones((T + 1, actions.COUNT), bool)
    masks[:, actions.INDEX['shield']] = False
    acts = rng.integers(0, actions.COUNT, T)
    acts[acts == actions.INDEX['shield']] = 0
    disc = np.full(T, 0.991, np.float32)
    disc[T // 2] = 0.0
    return {'kind': 'frag', 'worker': worker, 'port': 1, 'version': version, 'frames': 3 * T,
            'floats': rng.normal(size=(T + 1, observation.FLOATS)).astype(np.float32),
            'ids': rng.integers(0, 30, (T + 1, observation.IDS)).astype(np.int64),
            'masks': masks, 'actions': acts.astype(np.int64),
            'logp': np.full(T, -np.log(actions.COUNT - 1), np.float32),
            'rewards': rng.normal(scale=0.1, size=T).astype(np.float32), 'discounts': disc}


def make(tmp_path, **kw):
    cfg = Config(iso='/dev/null', hidden=64, fragment=16, batch_steps=64, minibatch=32, device='cpu',
                 max_policy_lag=2, ladder_window=5, **kw)
    return Learner(cfg, tmp_path / 'run')


def test_update_publishes_new_policy(tmp_path):
    L = make(tmp_path)
    assert load_policy(L.run_dir / 'policy.pt')['version'] == 0
    for i in range(4):
        L.handle(fragment(0, seed=i))
    L.update()
    assert L.version == 1 and L.frames_trained == 4 * 48
    assert load_policy(L.run_dir / 'policy.pt')['version'] == 1
    assert np.isfinite(list(L.last_metrics.values())[:5]).all()


def test_stale_fragments_are_dropped(tmp_path):
    L = make(tmp_path)
    L.version = 10
    L.handle(fragment(5))
    assert L.dropped_stale == 1 and not L.pending
    L.handle(fragment(9))
    assert len(L.pending) == 1


def test_games_drive_the_ladder_by_level_played(tmp_path):
    L = make(tmp_path, cpu_start_level=3)
    game = {'kind': 'game', 'opponent_kind': 'cpu', 'opponent': 'FOX', 'cpu_requested': 3,
            'result': 'win', 'actions': [0] * actions.COUNT, 'stocks_lost': 0, 'self_destructs': 0,
            'rests': 0, 'rest_hits': 0, 'damage_dealt': 100, 'damage_received': 10}
    for _ in range(5):
        L.handle({**game, 'cpu_level': 0})       # the slider did not take: never credited
    assert L.ladder.frontier == 3
    for _ in range(5):
        L.handle({**game, 'cpu_level': 3})
    assert L.ladder.frontier == 4
    assert (L.run_dir / 'games.jsonl').read_text().count('\n') == 10


def test_checkpoint_resume_roundtrip(tmp_path):
    L = make(tmp_path)
    for i in range(4):
        L.handle(fragment(0, seed=i))
    L.update()
    L.checkpoint()
    before = {k: v.clone() for k, v in L.net.state_dict().items()}
    R = Learner(L.cfg, L.run_dir, resume=True)
    assert R.version == 1 and R.frames_trained == L.frames_trained
    for k, v in R.net.state_dict().items():
        torch.testing.assert_close(v, before[k])
    L.write_status()
    assert (L.run_dir / 'status.json').exists()


def test_checkpoint_retention_keeps_recent_and_hourly(tmp_path):
    import os
    L = make(tmp_path)
    d = L.run_dir / 'checkpoints'
    base = 1_000_000_000
    for i in range(40):                      # one file every 10 minutes for ~6.5 hours
        p = d / f'policy-{i:08.2f}M-v{i}.pt'
        p.write_bytes(b'x')
        os.utime(p, (base + 600 * i, base + 600 * i))
    L._prune_checkpoints()
    left = sorted(int(p.stem.split('-v')[1]) for p in d.glob('policy-*.pt'))
    assert left[-12:] == list(range(28, 40))
    assert {0, 6, 12, 18, 24} <= set(left)
    assert len(left) < 40


def test_mirror_experience_is_not_counted_as_game_time(tmp_path):
    L = make(tmp_path)
    for i in range(2):
        L.handle(fragment(0, seed=i))                       # port 1: the game's own frames
        L.handle({**fragment(0, seed=10 + i), 'port': 2})    # port 2 of a mirror game
    L.update()
    assert L.frames_trained == 4 * 48 and L.world_frames_trained == 2 * 48


def test_fragments_that_go_stale_while_waiting_are_dropped(tmp_path):
    L = make(tmp_path)
    for i in range(4):
        L.handle(fragment(0, seed=i))
    L.version = 5                                            # the learner moved on meanwhile
    L.update()
    assert L.version == 5 and L.dropped_stale == 4 and L.pending_steps == 0


def test_silent_live_worker_is_restarted(tmp_path):
    import time as _t
    from puffbot import learner as lm

    class Stuck:
        killed = False
        exitcode = None
        def is_alive(self): return not self.killed
        def kill(self): self.killed = True
        def join(self, timeout=None): pass
    L = make(tmp_path)
    L.procs[0] = Stuck()
    L.worker_seen[0] = _t.monotonic() - lm.STALL_SECONDS - 1
    L.supervise()
    assert L.procs[0].killed and 'no progress' in L.errors[-1]['error']


def test_pruning_keeps_evaluated_checkpoints(tmp_path, monkeypatch):
    import json, os
    from puffbot import config
    monkeypatch.setattr(config, 'RUNS', tmp_path / 'runs')
    monkeypatch.setattr(config, 'ROOT', tmp_path)
    L = make(tmp_path)
    d = L.run_dir / 'checkpoints'
    base = 1_000_000_000
    for i in range(20):
        p = d / f'policy-{i:08.2f}M-v{i}.pt'
        p.write_bytes(b'x')
        os.utime(p, (base + 60 * i, base + 60 * i))          # all within one hour
    ev = tmp_path / 'runs' / 'evals' / 'e1'
    ev.mkdir(parents=True)
    (ev / 'report.json').write_text(json.dumps({'checkpoint': str(d / 'policy-00010.00M-v10.pt')}))
    L._prune_checkpoints(recent=3)
    left = sorted(int(p.stem.split('-v')[1]) for p in d.glob('policy-*.pt'))
    assert left == [0, 10, 17, 18, 19], 'first (hourly), the evaluated one, and the 3 newest'


def test_game_time_estimate_stops_at_the_restored_checkpoint(tmp_path):
    import json
    from puffbot.learner import _logged_game_frames
    log = tmp_path / 'games.jsonl'
    log.write_text('\n'.join(json.dumps({'version': v, 'frames': 1000}) for v in (1, 5, 9, 20)))
    assert _logged_game_frames(log, 9) == 3000 and _logged_game_frames(log) == 4000


def test_heartbeat_without_frames_is_not_progress(tmp_path):
    import time as _t
    from puffbot import learner as lm

    class Alive:
        killed = False
        exitcode = None
        def is_alive(self): return not self.killed
        def kill(self): self.killed = True
        def join(self, timeout=None): pass
    L = make(tmp_path)
    L.procs[0] = Alive()
    L.handle({'kind': 'status', 'worker': 0, 'frames': 500, 'phase': 'playing'})
    L.worker_progress[0] = _t.monotonic() - lm.STALL_SECONDS - 1
    L.handle({'kind': 'status', 'worker': 0, 'frames': 500, 'phase': 'playing'})   # still talking
    L.supervise()
    assert L.procs[0].killed


def game(offstage, frames=3600, result='win'):
    return {'kind': 'game', 'worker': 0, 'opponent_kind': 'mirror', 'opponent': 'JIGGLYPUFF', 'cpu_level': 0,
            'result': result, 'frames': frames, 'offstage_frames': int(offstage * frames)}


def test_stalling_watch_warns_alarms_and_recovers_without_flapping(tmp_path):
    early = make(tmp_path / 'early', stall_window=10)
    for _ in range(9):
        early.handle(game(0.9))
    assert early.stall['level'] == 'ok', 'not judged before a full window'
    L = make(tmp_path, stall_window=10)
    notes = lambda: [e['error'] for e in L.errors if 'Stalling' in e['error']]
    for _ in range(10):
        L.handle(game(0.3))
    assert L.stall['level'] == 'ok' and not notes()
    for _ in range(10):
        L.handle(game(0.55))
    assert L.stall['level'] == 'warn' and notes()[-1].startswith('Stalling watch: 5')
    for _ in range(10):
        L.handle(game(0.7))
    assert L.stall['level'] == 'alarm' and notes()[-1].startswith('Stalling ALARM')
    for _ in range(10):
        L.handle(game(0.58))
    assert L.stall['level'] == 'alarm', 'just under the line is noise, not recovery'
    for _ in range(10):
        L.handle(game(0.2))
    assert L.stall['level'] == 'ok' and 'back to normal' in notes()[-1]
    assert [n.split(':')[0] for n in notes()] == ['Stalling watch', 'Stalling ALARM', 'Stalling watch',
                                                  'Stalling watch back to normal'], 'down through warn, once each'


def test_newest_checkpoint_before_stalling_is_kept(tmp_path):
    import os
    L = make(tmp_path, stall_window=10)
    L.checkpoint()
    good = L.last_ok_checkpoint
    for _ in range(10):
        L.handle(game(0.8))
    assert L.stall['level'] == 'alarm'
    assert Path(good).name in L.errors[-1]['error']
    old = 1_000_000_000
    os.utime(good, (old, old))
    d = L.run_dir / 'checkpoints'
    for i in range(20):                      # newer files that would push it out
        p = d / f'policy-{50 + i:08.2f}M-v{50 + i}.pt'
        p.write_bytes(b'x')
        os.utime(p, (old + 600 * (i + 1), old + 600 * (i + 1)))
    L.checkpoint()
    assert L.last_ok_checkpoint == good, 'checkpoints saved while stalling are not "last ok"'
    assert Path(good).exists()


def test_critic_warmup_trains_only_the_value_side(tmp_path):
    L = make(tmp_path, critic_warmup_updates=2)
    L._start_warmup(0.997)
    assert L.critic_warmup_left == 2 and 'gamma 0.997 -> 0.999' in L.errors_on_start[-1]
    before = {n: p.detach().clone() for n, p in L.net.named_parameters()}
    for step in range(2):
        for i in range(4):
            L.handle(fragment(L.version, seed=10 * step + i))
        L.update()
    for n, p in L.net.named_parameters():
        moved = not torch.equal(before[n], p.detach())
        assert moved == n.startswith(('v_body', 'v_head')), n
    assert L.critic_warmup_left == 0 and 'warm-up finished' in L.errors[-1]['error']
    for i in range(4):
        L.handle(fragment(L.version, seed=99 + i))
    L.update()
    assert not torch.equal(before['pi_head.weight'], L.net.pi_head.weight.detach())


def test_warmup_starts_only_when_the_horizon_changed(tmp_path):
    from puffbot.model import Policy, save_policy
    for gamma, expected in ((None, 7), (0.997, 7), (0.999, 0)):
        path = tmp_path / f'w-{gamma}.pt'
        save_policy(path, Policy(64), 5, 1000, None if gamma is None else {'gamma': gamma}, act_every=3)
        cfg = Config(iso='/dev/null', hidden=64, device='cpu', critic_warmup_updates=7)
        L = Learner(cfg, tmp_path / f'run-{gamma}', init_from=str(path))
        assert L.critic_warmup_left == expected, gamma
    L.critic_warmup_left = 3
    L.checkpoint()
    R = Learner(L.cfg, L.run_dir, resume=True)
    assert R.critic_warmup_left == 3, 'progress survives a stop/resume'


def test_scorecard_runs_on_schedule_one_at_a_time_and_reports(tmp_path, monkeypatch):
    import json
    import time as time_mod
    from puffbot import learner as learner_mod
    monkeypatch.setattr(learner_mod, 'RUNS', tmp_path / 'runs')
    L = make(tmp_path, scorecard_hours=3.0)
    launched = []

    class Proc:
        code = None
        def poll(self):
            return self.code

    def fake_spawn(args, out):
        launched.append((args, out))
        return Proc()
    monkeypatch.setattr(L, '_spawn_scorecard', fake_spawn)
    L.checkpoint()
    L.maybe_scorecard()
    assert not launched, 'not before scorecard_hours have passed'
    L.last_scorecard = time_mod.time() - 3 * 3600 - 1
    L.maybe_scorecard()
    L.last_scorecard = 0
    L.maybe_scorecard()
    assert len(launched) == 1, 'one at a time'
    args, out = launched[0]
    assert args[args.index('--label') + 1] == 'scorecard' and args[args.index('--level') + 1] == '9'
    assert 'Scorecard started' in L.errors[-1]['error']
    (out / 'report.json').write_text(json.dumps({
        'state': 'complete', 'checkpoint': str(L.last_policy_file), 'level': 9,
        'overall': {'wins': 50, 'games': 60, 'win_rate': 0.833, 'win_rate_95ci': [0.72, 0.91]}}))
    L.scorecard_proc.code = 0
    L._scorecard_finished()
    assert L.errors[-1]['error'].endswith('won 50/60 (83%, 95% CI 72%-91%) vs CPU 9')
    assert L.scorecard_proc is None


def test_events_are_kept_across_restarts(tmp_path):
    L = make(tmp_path, stall_window=10)
    for _ in range(10):
        L.handle(game(0.9))
    L.checkpoint()
    R = Learner(L.cfg, L.run_dir, resume=True)
    assert any('Stalling ALARM' in e['error'] for e in R.errors)
    lines = (L.run_dir / 'events.jsonl').read_text().splitlines()
    assert any('Stalling ALARM' in line for line in lines)
