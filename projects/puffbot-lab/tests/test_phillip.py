"""Phillip opponent plumbing, without TensorFlow: a fake server speaks the protocol."""
from __future__ import annotations

import sys
import textwrap
import threading
from types import SimpleNamespace as NS

import numpy as np
import pytest

from puffbot import league, phillip
from puffbot.config import Config
from puffbot.dolphin import EmulatorError, Setup
from puffbot.league import Opponent

FAKE = textwrap.dedent('''
    import pickle, struct, sys, time
    def send(m):
        d = pickle.dumps(m); sys.stdout.buffer.write(struct.pack('>I', len(d)) + d); sys.stdout.buffer.flush()
    def recv():
        h = sys.stdin.buffer.read(4)
        return None if len(h) < 4 else pickle.loads(sys.stdin.buffer.read(struct.unpack('>I', h)[0]))
    mode = sys.argv[sys.argv.index('--model') + 1]
    if mode == 'broken':
        send(('error', 'load failed: no such model')); sys.exit(0)
    send(('ready', {'delay': 21}))
    while True:
        m = recv()
        if m is None or m[0] == 'quit':
            break
        if mode == 'hang':
            time.sleep(60)
        if mode == 'crash':
            sys.exit(3)
        send(('action', {'buttons': ['BUTTON_A'], 'main': (0.0, 0.5), 'c': (0.5, 0.5), 'shoulder': 0.0,
                         'frame': m[1]['frame']}))
''')


@pytest.fixture
def fake(tmp_path):
    path = tmp_path / 'fake_server.py'
    path.write_text(FAKE)
    return lambda mode: phillip.Phillip(sys.executable, mode, tmp_path / 'p.log', server=path)


def test_steps_round_trip(fake):
    bot = fake('ok')
    try:
        assert bot.info['delay'] == 21
        assert bot.step({'frame': 7})['frame'] == 7 and bot.frames == 1
    finally:
        bot.close()
    assert not bot.alive


def test_a_silent_phillip_raises_instead_of_hanging(fake, monkeypatch):
    monkeypatch.setattr(phillip, 'STEP_SECONDS', 0.5)
    bot = fake('hang')
    with pytest.raises(phillip.PhillipError, match='stopped answering'):
        bot.step({'frame': 1})
    bot.close()


def test_a_crashed_phillip_raises_an_emulator_style_error(fake):
    bot = fake('crash')
    with pytest.raises(EmulatorError):
        bot.step({'frame': 1})
    bot.close()


def test_a_model_that_fails_to_load_is_reported(fake):
    with pytest.raises(phillip.PhillipError, match='no such model'):
        fake('broken')


def test_choose_phillip_uses_the_character_weights():
    cfg = Config(iso='/dev/null', phillip_characters={'FOX': 3.0, 'LUIGI': 1.0})
    rng = np.random.default_rng(0)
    picks = [league.choose_phillip(cfg, rng) for _ in range(400)]
    assert all(p.kind == 'phillip' and p.setup.cpu_level == 0 for p in picks)
    foxes = sum(p.setup.p2 == 'FOX' for p in picks)
    assert 250 < foxes < 350


def test_phillip_config_is_validated(tmp_path):
    iso = tmp_path / 'm.iso'
    iso.write_bytes(b'x')
    base = dict(iso=str(iso), workers=4, phillip_python=str(iso), phillip_model=str(iso))
    Config(**base, phillip_workers=2).validate()
    with pytest.raises(ValueError):
        Config(**base, phillip_workers=5).validate()
    with pytest.raises(ValueError):
        Config(**base, phillip_workers=1, phillip_characters={'SHEIK': 1.0}).validate()
    with pytest.raises(ValueError):
        Config(**{**base, 'phillip_model': str(tmp_path / 'missing')}, phillip_workers=1).validate()


# ------------------------------------------------------------ inside the worker
class Bot:
    def __init__(self, fail_at=None):
        self.frames, self.fail_at, self.alive = [], fail_at, True
        self.step_seconds = 0.0

    def step(self, g):
        if self.fail_at is not None and len(self.frames) >= self.fail_at:
            raise phillip.PhillipError('Phillip stopped answering.')
        self.frames.append(g.frame)
        return {'buttons': ['BUTTON_B'], 'main': (0.5, 0.0), 'c': (0.5, 0.5), 'shoulder': 0.0}

    def close(self):
        self.alive = False


def phillip_worker(tmp_path, bot):
    from tests.test_worker import FakeEmu, Sink
    from puffbot.actor import Worker
    cfg = Config(iso='/dev/null', hidden=32, fragment=16, games_per_setup=2)
    sink = Sink()
    w = Worker(0, cfg, tmp_path / 'run', sink, threading.Event(), seed=1)
    w.emu = FakeEmu(every=30)
    w.phillip_client = lambda: bot
    return w, sink


def test_worker_feeds_phillip_every_frame_including_the_countdown(tmp_path):
    bot = Bot()
    w, sink = phillip_worker(tmp_path, bot)
    w.play(Opponent('phillip', Setup('JIGGLYPUFF', 'FOX', 0)))
    games = [m for m in sink.msgs if m['kind'] == 'game']
    assert [g['opponent_kind'] for g in games] == ['phillip', 'phillip'] and games[0]['opponent'] == 'FOX'
    assert min(bot.frames) < 0, 'countdown frames are sent (Phillip resets on frame -123)'
    assert bot.frames.count(-5) == 2, 'every game, including the Instant Match restart'
    assert all(m['port'] == 1 for m in sink.msgs if m['kind'] == 'frag'), 'only Puff learns'


def test_phillip_failing_mid_game_cuts_the_game(tmp_path):
    bot = Bot(fail_at=50)
    w, sink = phillip_worker(tmp_path, bot)
    with pytest.raises(EmulatorError):
        w.play(Opponent('phillip', Setup('JIGGLYPUFF', 'FOX', 0)))
    assert [g['result'] for g in sink.msgs if g['kind'] == 'game'] == ['interrupted']
