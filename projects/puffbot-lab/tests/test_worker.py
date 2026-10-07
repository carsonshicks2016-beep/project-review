"""The worker's game loop against a scripted fake emulator: rewards, terminal
discounts, fragments and game reports, with no Dolphin involved."""
from __future__ import annotations

import threading
from types import SimpleNamespace as NS

import melee
import numpy as np
import pytest

from puffbot import actions, observation
from puffbot.actor import Worker
from puffbot.config import Config
from puffbot.dolphin import Setup
from puffbot.league import Opponent


class Pad:
    def release_all(self): pass
    def tilt_analog(self, *a): pass
    def press_button(self, *a): pass
    def press_shoulder(self, *a): pass


def pstate(stock, x=0.0, cpu=0):
    return NS(position=NS(x=x, y=0.0), percent=0.0, stock=stock, facing=True, on_ground=True,
              off_stage=False, jumps_left=6, shield_strength=60.0, invulnerable=False, hitlag_left=0,
              hitstun_frames_left=0, action_frame=1, speed_air_x_self=0.0, speed_y_self=0.0,
              speed_x_attack=0.0, speed_y_attack=0.0, speed_ground_x_self=0.0,
              action=NS(value=0x0E), character=NS(value=15), cpu_level=cpu)


class FakeEmu:
    """Games where port 2 loses a stock every `every` frames; after the last one the game
    freezes briefly and Instant Match restarts it at a negative frame."""

    def __init__(self, every=40, level=5, time_limit=None):
        self.pads, self.visible, self.launches = [Pad(), Pad()], False, 0
        self.every, self.level, self.time_limit = every, level, time_limit
        self.menu = melee.Menu.IN_GAME

    def launch(self):
        self.launches += 1

    def stop(self):
        pass

    def _state(self):
        return NS(menu_state=self.menu, frame=self.f, projectiles=[],
                  players={1: pstate(4, -20), 2: pstate(self.stock2, 20, self.level)})

    def enter_match(self, setup):
        self.f, self.stock2, self.freeze = -5, 4, None
        return self._state()

    def frame(self):
        if self.freeze is not None:
            self.freeze -= 1
            if self.freeze == 0:
                self.f, self.stock2, self.freeze = -5, 4, None
                return self._state()
        self.f += 1
        if self.time_limit and self.f >= self.time_limit:
            self.menu = melee.Menu.SUDDEN_DEATH
            return self._state()
        if self.f > 0 and self.f % self.every == 0 and self.stock2 > 0:
            self.stock2 -= 1
            if self.stock2 == 0:
                self.freeze = 6
        return self._state()


class Sink:
    def __init__(self):
        self.msgs = []

    def put(self, msg, timeout=None):
        self.msgs.append(msg)


def make_worker(tmp_path, **kw):
    cfg = Config(**{**dict(iso='/dev/null', hidden=32, fragment=16, act_every=3, games_per_setup=2), **kw})
    sink = Sink()
    w = Worker(0, cfg, tmp_path / 'run', sink, threading.Event(), seed=1)
    w.emu = FakeEmu()
    return w, sink


def test_cpu_games_produce_consistent_fragments(tmp_path):
    w, sink = make_worker(tmp_path)
    w.play(Opponent('cpu', Setup('JIGGLYPUFF', 'FOX', 5)))
    games = [m for m in sink.msgs if m['kind'] == 'game']
    frags = [m for m in sink.msgs if m['kind'] == 'frag']
    assert [g['result'] for g in games] == ['win', 'win']
    assert all(g['cpu_level'] == 5 and g['stocks_taken'] == 4 and g['stocks_lost'] == 0 for g in games)
    assert frags, 'expected at least one fragment'
    for f in frags:
        T = len(f['actions'])
        assert T == 16
        assert f['floats'].shape == (T + 1, observation.FLOATS)
        assert f['masks'].shape == (T + 1, actions.COUNT)
        assert f['masks'][np.arange(T), f['actions']].all(), 'chosen macros must be legal'
        assert np.all(f['frames'] >= 3 * T)
        d = f['discounts']
        gamma = w.cfg.gamma
        # One decision covers 3..MAX_FRAMES frames; a terminal step discounts to zero.
        assert np.all((d == 0) | ((d >= gamma ** actions.MAX_FRAMES - 1e-6) & (d <= gamma ** 3 + 1e-6)))
    # Everything the policy was paid: 4 stocks + the win, per game, up to discounting
    # within a decision (at most three frames of gamma).
    rows_r = np.concatenate([f['rewards'] for f in frags] + [np.array([r['r'] for r in w.buffers[1].rows])])
    rows_d = np.concatenate([f['discounts'] for f in frags] + [np.array([r['discount'] for r in w.buffers[1].rows])])
    assert (rows_d == 0).sum() == 2, 'exactly one terminal step per finished game'
    assert rows_r.sum() == pytest.approx(2 * (4 + 1), rel=0.02)


def test_mirror_trains_both_ports(tmp_path):
    w, sink = make_worker(tmp_path, fragment=8)
    w.play(Opponent('mirror', Setup('JIGGLYPUFF', 'JIGGLYPUFF', 0)))
    ports = {m['port'] for m in sink.msgs if m['kind'] == 'frag'}
    assert ports == {1, 2}
    loser = [m for m in sink.msgs if m['kind'] == 'frag' and m['port'] == 2]
    assert sum(float(f['rewards'].sum()) for f in loser) < 0
    games = [m for m in sink.msgs if m['kind'] == 'game']
    assert all(g['cpu_level'] == 0 and g['opponent_kind'] == 'mirror' for g in games)
    assert not w.buffers[2].rows, 'port 2 leftovers are cleared after a mirror setup'


def test_timeout_into_sudden_death_ends_the_setup_cleanly(tmp_path):
    """A timed-out game with tied stocks goes to Sudden Death; that used to raise and
    relaunch the emulator as if it had crashed."""
    w, sink = make_worker(tmp_path)
    w.emu = FakeEmu(every=10_000, time_limit=300)
    w.play(Opponent('mirror', Setup('JIGGLYPUFF', 'JIGGLYPUFF', 0)))
    games = [m for m in sink.msgs if m['kind'] == 'game']
    assert len(games) == 1 and games[0]['timeout'] and games[0]['result'] == 'draw'
    assert all(r['discount'] > 0 for r in w.buffers[1].rows[:-1]) and w.buffers[1].rows[-1]['discount'] == 0


def test_safety_cap_cuts_the_game_without_a_fake_ending(tmp_path):
    """GPT audit: max_game_frames was never enforced (a game ran 160 frames with a cap
    of 30). A capped game is recorded, but its unfinished data is discarded rather than
    taught as if the game had ended there."""
    w, sink = make_worker(tmp_path, max_game_frames=200)
    w.emu = FakeEmu(every=10_000)                      # nobody ever loses a stock
    w.play(Opponent('cpu', Setup('JIGGLYPUFF', 'FOX', 5)))
    games = [m for m in sink.msgs if m['kind'] == 'game']
    assert [g['result'] for g in games] == ['capped']
    assert w.emu.f <= 201, 'the loop stopped at the cap'
    assert not w.buffers[1].rows, 'unfinished rows of the capped game are dropped'
    for f in (m for m in sink.msgs if m['kind'] == 'frag'):
        assert (f['discounts'] > 0).all(), 'no invented terminal state'


class FailingEmu(FakeEmu):
    def frame(self):
        if self.f >= 150:
            from puffbot.dolphin import EmulatorError
            raise EmulatorError('Slippi stream disconnected.')
        return super().frame()


def test_emulator_failure_mid_game_is_not_a_game_ending(tmp_path):
    from puffbot.dolphin import EmulatorError
    w, sink = make_worker(tmp_path, fragment=8)
    w.emu = FailingEmu(every=100)                      # first stock falls at frame 100
    with pytest.raises(EmulatorError):
        w.play(Opponent('cpu', Setup('JIGGLYPUFF', 'FOX', 5)))
    games = [m for m in sink.msgs if m['kind'] == 'game']
    assert [g['result'] for g in games] == ['interrupted']
    rows = [r for f in sink.msgs if f['kind'] == 'frag' for r in f['discounts']] + \
           [r['discount'] for r in w.buffers[1].rows]
    assert all(d > 0 for d in rows), 'failure is not recorded as a terminal state'
    assert not w.buffers[1].rows


class FailInSecondGame(FakeEmu):
    """Plays one full game, then fails 30 frames into the next."""

    def __init__(self, **kw):
        super().__init__(**kw)
        self.restarts = 0

    def frame(self):
        from puffbot.dolphin import EmulatorError
        if self.restarts and self.f >= 30:
            raise EmulatorError('Dolphin exited (code -9).')
        was = self.freeze
        state = super().frame()
        if was == 1 and self.freeze is None:
            self.restarts += 1
        return state


def test_finished_games_before_a_failure_keep_their_data(tmp_path):
    from puffbot.dolphin import EmulatorError
    w, sink = make_worker(tmp_path, games_per_setup=3)
    w.emu = FailInSecondGame(every=20)
    with pytest.raises(EmulatorError):
        w.play(Opponent('cpu', Setup('JIGGLYPUFF', 'FOX', 5)))
    assert [g['result'] for g in sink.msgs if g['kind'] == 'game'] == ['win', 'interrupted']
    kept = [r['discount'] for r in w.buffers[1].rows]
    sent = [d for f in sink.msgs if f['kind'] == 'frag' for d in f['discounts']]
    assert (kept + sent).count(0.0) == 1, "the finished game's ending survives the cut"
    assert not kept or kept[-1] == 0.0, 'nothing of the interrupted game is left behind'


class FailRightAfterAWin(FakeEmu):
    """Wins a game, then the emulator dies before Instant Match resets the frame."""

    def frame(self):
        from puffbot.dolphin import EmulatorError
        if self.freeze is not None and self.freeze <= 5:
            raise EmulatorError('Slippi stream disconnected.')
        return super().frame()


@pytest.mark.parametrize('fragment', [8, 128])
@pytest.mark.parametrize('kind', ['cpu', 'mirror'])
def test_crash_right_after_a_win_keeps_that_games_data(tmp_path, fragment, kind):
    """GPT step-1 review R3: a failure between a win and the next frame reset trimmed
    the finished game back to its start, losing its terminal step."""
    from puffbot.dolphin import EmulatorError
    w, sink = make_worker(tmp_path, fragment=fragment)
    w.emu = FailRightAfterAWin(every=20)
    setup = Setup('JIGGLYPUFF', 'FOX', 5) if kind == 'cpu' else Setup('JIGGLYPUFF', 'JIGGLYPUFF', 0)
    with pytest.raises(EmulatorError):
        w.play(Opponent(kind, setup))
    assert [g['result'] for g in sink.msgs if g['kind'] == 'game'] == ['win']
    kept = [r['discount'] for r in w.buffers[1].rows]
    sent = [d for f in sink.msgs if f['kind'] == 'frag' and f['port'] == 1 for d in f['discounts']]
    assert (kept + sent).count(0.0) == 1, 'the finished game still ends in its terminal step'
