import numpy as np
import pytest
from collections import deque
from pathlib import Path
from melee_lab.config import Config
from melee_lab.controller import DIMENSIONS
from melee_lab.state import OBS_SIZE
from melee_lab.worker import Progress, Anchor, Runtime
import melee


def test_anchor_weight_respects_floor(tmp_path):
    path = tmp_path / 'demo.npz'
    np.savez_compressed(path, observations=np.zeros((4, OBS_SIZE), np.float32),
                        actions=np.zeros((4, len(DIMENSIONS)), np.int64))
    anchor = Anchor(path, None, coefficient=0.5, floor=0.12)
    anchor.total = 100
    anchor.num_timesteps = 0
    assert anchor.weight() == pytest.approx(0.5)
    anchor.num_timesteps = 50
    assert anchor.weight() == pytest.approx(0.25)
    anchor.num_timesteps = 100
    assert anchor.weight() == pytest.approx(0.12)
    anchor.num_timesteps = 200
    assert anchor.weight() == pytest.approx(0.12)


def test_curriculum_advances_at_configured_win_threshold(tmp_path):
    runtime = Runtime(tmp_path, Config())
    calls = []

    class FakeVec:
        def set_attr(self, name, val):
            calls.append((name, val))

    vec = FakeVec()
    progress = Progress(runtime, None, True, (), vec, level=5, win_threshold=0.60)

    def finish(level, result, opp='MARIO'):
        progress.locals = {
            'infos': [{
                'cpu_level': level,
                'result': result,
                'players': {
                    '1': {'character': 'FOX'},
                    '2': {'character': opp}
                },
                'frame': 100,
                'starting_stocks': [3, 3],
                'episode': {'r': 10.0}
            }],
            'dones': [True]
        }
        progress._on_step()

    # 11 wins out of 20 (55%) should NOT promote under 60%
    for i in range(20):
        finish(5, 'win' if i < 11 else 'loss')
    assert progress.level == 5

    # Replace 9 losses with 9 wins, making it 12+ wins in the window
    finish(5, 'win' if False else 'loss') # still at 5
    assert progress.level == 5

    # A fresh window with 12 wins out of 20 (60%)
    for i in range(20):
        finish(5, 'win' if i < 12 else 'loss')
    assert progress.level == 6
    assert calls == [('cpu_level', 6)]


def test_offstage_action_sanitizer():
    from melee_lab.environment import MeleeEnv

    env = MeleeEnv(Config(action_set='controller', action_frames=1))

    class MockPos:
        def __init__(self, x, y):
            self.x = x
            self.y = y

    class MockPlayer:
        def __init__(self, x, y, off_stage, on_ground, jumps):
            self.position = MockPos(x, y)
            self.off_stage = off_stage
            self.on_ground = on_ground
            self.jumps_left = jumps

    class MockState:
        def __init__(self, p1, p2):
            self.players = {1: p1, 2: p2}

    action = np.array([10, 3, 1, 1, 0, 0, 2], dtype=np.int64)

    # On stage -> untouched. Shielding and C-stick are the whole grounded game.
    env.state = MockState(MockPlayer(0, 0, off_stage=False, on_ground=True, jumps=2),
                          MockPlayer(50, 0, off_stage=False, on_ground=True, jumps=2))
    np.testing.assert_array_equal(env._sanitize_action(action), action)

    # Offstage but standing on the ledge -> untouched; the trigger is not an air-dodge.
    env.state = MockState(MockPlayer(-86, 0, off_stage=True, on_ground=True, jumps=2),
                          MockPlayer(0, 0, off_stage=False, on_ground=True, jumps=2))
    np.testing.assert_array_equal(env._sanitize_action(action), action)

    # Airborne offstage with jumps to spare -> the trigger is masked and nothing else.
    # This is the case that was killing the agent: an air-dodge here enters helpless
    # fall and the remaining jumps become unusable, whatever the character.
    env.state = MockState(MockPlayer(-90, -10, off_stage=True, on_ground=False, jumps=1),
                          MockPlayer(0, 0, off_stage=False, on_ground=True, jumps=2))
    sanitized = env._sanitize_action(action)
    np.testing.assert_array_equal(sanitized, np.array([10, 3, 1, 1, 0, 0, 0], dtype=np.int64))

    # Jigglypuff offstage with all five jumps -> still masked. The old guard waited for
    # jumps_left == 0, which a five-jump character effectively never reaches in time.
    env.state = MockState(MockPlayer(-120, -40, off_stage=True, on_ground=False, jumps=5),
                          MockPlayer(0, 0, off_stage=False, on_ground=True, jumps=2))
    assert env._sanitize_action(action)[6] == 0

    # Out of jumps but the opponent is right there -> keep the aerial, lose the trigger.
    env.state = MockState(MockPlayer(-90, -10, off_stage=True, on_ground=False, jumps=0),
                          MockPlayer(-80, -10, off_stage=True, on_ground=False, jumps=0))
    sanitized = env._sanitize_action(action)
    assert sanitized[2] == 1 and sanitized[1] == 3 and sanitized[6] == 0

    # Out of jumps and alone -> A, C-stick and trigger all masked; stick and B survive,
    # because those are the recovery.
    env.state = MockState(MockPlayer(-95, -20, off_stage=True, on_ground=False, jumps=0),
                          MockPlayer(0, 0, off_stage=False, on_ground=True, jumps=2))
    sanitized = env._sanitize_action(action)
    np.testing.assert_array_equal(sanitized, np.array([10, 0, 0, 1, 0, 0, 0], dtype=np.int64))


def test_recovery_shaping_is_potential_based_and_cannot_be_farmed():
    """This used to assert a -1.5 penalty for leaving the stage and a +1.5 bonus for
    getting back. Measured over a real match the agent crosses that boundary about 15
    times, so those two terms put roughly +/-22 of farmable reward against a 12-point
    stock signal.

    They are replaced by a potential over distance from the stage. Per-step reward is
    PHI(s') - PHI(s), which telescopes: any path that returns to a state it has already
    visited pays exactly nothing, and the optimal policy is provably unchanged."""
    from melee_lab.state import reward, RECOVERY_POTENTIAL

    class Pos:
        def __init__(self, x, y): self.x, self.y = x, y

    class Fighter:
        def __init__(self, x=0.0, y=0.0, stock=3, percent=0.0):
            self.position = Pos(x, y)
            self.stock, self.percent = stock, percent

    class State:
        def __init__(self, p1, p2): self.players = {1: p1, 2: p2}

    def at(x, y=0.0, stock=3):
        return State(Fighter(x, y, stock), Fighter(300.0, 0.0))

    # Drifting out to sea costs, coming back pays, and the two cancel exactly.
    out = reward(at(0.0), at(150.0))
    home = reward(at(150.0), at(0.0))
    assert out < 0 < home
    assert out + home == pytest.approx(0.0, abs=1e-9)

    # A hundred round trips are worth nothing at all.
    total = 0.0
    for _ in range(100):
        total += reward(at(0.0), at(140.0)) + reward(at(140.0), at(0.0))
    assert total == pytest.approx(0.0, abs=1e-6)

    # Magnitude stays small next to a stock.
    assert abs(out) < 4.0
    assert RECOVERY_POTENTIAL > 0


def test_dying_does_not_pay_for_the_respawn_teleport():
    """Respawning puts the agent back at the centre of the stage. Applied across a death,
    the recovery potential would read that teleport as a heroic recovery."""
    from melee_lab.state import reward, STOCK

    class Pos:
        def __init__(self, x, y): self.x, self.y = x, y

    class Fighter:
        def __init__(self, x=0.0, y=0.0, stock=3, percent=0.0):
            self.position = Pos(x, y)
            self.stock, self.percent = stock, percent

    class State:
        def __init__(self, p1, p2): self.players = {1: p1, 2: p2}

    far_out = State(Fighter(260.0, -200.0, stock=3, percent=90.0), Fighter(0.0))
    respawned = State(Fighter(0.0, 0.0, stock=2, percent=0.0), Fighter(0.0))
    assert reward(far_out, respawned) == pytest.approx(-STOCK)
def test_stock_losses_are_classified_as_kills_or_self_destructs():
    """The win rate cannot separate being outplayed from walking off the stage."""
    from melee_lab.environment import MeleeEnv, SELF_DESTRUCT_WINDOW

    class P:
        def __init__(self, stock, percent=0.0, hitstun=0, on_ground=True,
                     off_stage=False, action='STANDING'):
            self.stock = stock
            self.percent = percent
            self.hitstun_frames_left = hitstun
            self.on_ground = on_ground
            self.off_stage = off_stage
            self.action = type('A', (), {'name': action})()

    class S:
        def __init__(self, frame, p1):
            self.frame = frame
            self.players = {1: p1}

    def fresh():
        env = MeleeEnv(Config(action_set='controller', action_frames=1))
        env.stock_losses = env.self_destructs = 0
        env.last_hit_frame = 0
        env.controlled_since_hit = True
        return env

    def feed(env, states):
        for a, b in zip(states, states[1:]):
            env._note_stock_loss(a, b)

    # Safe on stage the whole time, then the stock is gone: the agent threw it away.
    env = fresh()
    feed(env, [S(0, P(3)), S(SELF_DESTRUCT_WINDOW + 20, P(3)), S(SELF_DESTRUCT_WINDOW + 21, P(2))])
    assert (env.stock_losses, env.self_destructs) == (1, 1)

    # Damage, then death moments later: the opponent took that one.
    env = fresh()
    feed(env, [S(0, P(3)), S(1, P(3, percent=55.0)), S(10, P(2, percent=55.0))])
    assert (env.stock_losses, env.self_destructs) == (1, 0)

    # THE CASE THE FIRST VERSION GOT WRONG: launched offstage, survives the hit, flies
    # out under no further damage, and dies long after the 45-frame window has passed.
    # No contact near the death, but the agent never got control back -- not a suicide.
    env = fresh()
    launched = [S(0, P(3)), S(1, P(3, percent=70.0, hitstun=30))]
    drift = [S(f, P(3, percent=70.0, on_ground=False, off_stage=True, action='FALLING'))
             for f in range(40, 300, 20)]
    feed(env, launched + drift + [S(320, P(2, percent=70.0, on_ground=False, off_stage=True))])
    assert (env.stock_losses, env.self_destructs) == (1, 0), 'a survived launch is still the opponent kill'

    # Launched, but it RECOVERS to the ledge and only then throws the stock away.
    # Regaining control hands responsibility back to the agent.
    env = fresh()
    feed(env, [S(0, P(3)), S(1, P(3, percent=70.0, hitstun=30)),
               S(60, P(3, percent=70.0, on_ground=False, action='EDGE_HANGING')),
               S(200, P(3, percent=70.0, on_ground=False, off_stage=True, action='FALLING')),
               S(400, P(2, percent=70.0, on_ground=False, off_stage=True))])
    assert (env.stock_losses, env.self_destructs) == (1, 1), 'recovered, then self-destructed'


def test_in_control_marks_the_end_of_a_launch():
    from melee_lab.state import in_control

    class P:
        def __init__(self, on_ground=True, off_stage=False, hitstun=0, action='STANDING'):
            self.on_ground = on_ground
            self.off_stage = off_stage
            self.hitstun_frames_left = hitstun
            self.action = type('A', (), {'name': action})()

    assert in_control(P())
    assert in_control(P(on_ground=False, off_stage=True, action='EDGE_HANGING'))
    assert in_control(P(on_ground=False, off_stage=True, action='EDGE_CATCHING'))
    assert not in_control(P(hitstun=5))
    assert not in_control(P(on_ground=False, off_stage=True, action='FALLING'))
    assert not in_control(P(on_ground=False, action='JUMPING_FORWARD'))
    # Hitstun on the ground is still not control -- the punish is ongoing.
    assert not in_control(P(hitstun=3))


def _offstage_env(character='FOX'):
    from melee_lab.environment import MeleeEnv
    env = MeleeEnv(Config(action_set='controller', action_frames=1))
    env.firefox_latch = None
    return env


class _Pos:
    def __init__(self, x, y): self.x, self.y = x, y


class _Fighter:
    def __init__(self, x, y, action='FALLING', character='FOX',
                 off_stage=True, on_ground=False, jumps=1):
        self.position = _Pos(x, y)
        self.action = type('A', (), {'name': action})()
        self.character = type('C', (), {'name': character})()
        self.off_stage, self.on_ground, self.jumps_left = off_stage, on_ground, jumps


class _State:
    def __init__(self, p1, p2): self.players = {1: p1, 2: p2}


def test_firefox_angle_is_latched_across_the_charge():
    """The charge is one decision in the game's terms; at 1 frame per decision the policy
    was re-picking the launch angle 42 times inside it."""
    from melee_lab.controller import aims_home
    env = _offstage_env()
    far = _Fighter(400, 0, off_stage=False, on_ground=True)

    # Enter the charge with a sane upward angle -- it must survive to the launch frame.
    env.state = _State(_Fighter(-110, -30, action='SWORD_DANCE_3_LOW'), far)
    first = env._sanitize_action(np.array([20, 0, 0, 1, 0, 0, 0], dtype=np.int64))
    assert first[0] == 20 and aims_home(20, -110.0)

    # Later charge frames ask for wildly different angles and are overridden.
    for wanted in (1, 48, 70, 12):
        env.state = _State(_Fighter(-115, -40, action='SWORD_DANCE_3_HIGH'), far)
        held = env._sanitize_action(np.array([wanted, 0, 0, 0, 0, 0, 0], dtype=np.int64))
        assert held[0] == 20, 'the angle chosen entering the charge must carry to launch'

    # Once the charge ends the policy gets its stick back.
    env.state = _State(_Fighter(-115, -40, action='DEAD_FALL'), far)
    assert env._sanitize_action(np.array([48, 0, 0, 0, 0, 0, 0], dtype=np.int64))[0] == 48


def test_a_suicidal_charge_angle_is_replaced_not_held():
    from melee_lab.controller import aims_home, toward_stage
    env = _offstage_env()
    far = _Fighter(400, 0, off_stage=False, on_ground=True)
    away = toward_stage(110.0)          # points up-LEFT, wrong side when already left
    assert not aims_home(away, -110.0)
    env.state = _State(_Fighter(-110, -30, action='SWORD_DANCE_3_LOW'), far)
    out = env._sanitize_action(np.array([away, 0, 0, 1, 0, 0, 0], dtype=np.int64))
    assert out[0] == toward_stage(-110.0)


def test_a_last_ditch_b_press_below_the_stage_is_aimed_up():
    """Only 32% of offstage B presses were aimed up; the rest came out as laser,
    illusion or shine, none of which recover."""
    from melee_lab.controller import toward_stage, main_stick
    env = _offstage_env()
    far = _Fighter(400, 0, off_stage=False, on_ground=True)

    # Neutral-stick B below the stage -> a laser. Aimed up instead.
    env.state = _State(_Fighter(-120, -50), far)
    out = env._sanitize_action(np.array([0, 0, 0, 1, 0, 0, 0], dtype=np.int64))
    assert out[0] == toward_stage(-120.0)
    assert main_stick(int(out[0]))[1] > 0.5

    # B with an angle that already comes home is left alone.
    env.state = _State(_Fighter(-120, -50), far)
    good = toward_stage(-120.0)
    assert env._sanitize_action(np.array([good, 0, 0, 1, 0, 0, 0], dtype=np.int64))[0] == good

    # No B pressed -> the stick is the policy's business.
    env.state = _State(_Fighter(-120, -50), far)
    assert env._sanitize_action(np.array([48, 0, 0, 0, 0, 0, 0], dtype=np.int64))[0] == 48

    # Above the stage surface a side-B back onto the stage is real recovery; hands off.
    env.state = _State(_Fighter(-120, 25), far)
    assert env._sanitize_action(np.array([48, 0, 0, 1, 0, 0, 0], dtype=np.int64))[0] == 48


def test_the_firefox_rules_do_not_touch_other_characters():
    """SWORD_DANCE_3_* is Marth's dancing blade sharing an id with Fox's up-B."""
    env = _offstage_env()
    far = _Fighter(400, 0, off_stage=False, on_ground=True)
    for character in ('JIGGLYPUFF', 'MARTH'):
        env.state = _State(_Fighter(-120, -50, action='SWORD_DANCE_3_LOW',
                                    character=character), far)
        out = env._sanitize_action(np.array([48, 0, 0, 1, 0, 0, 0], dtype=np.int64))
        assert out[0] == 48, f'{character} must keep its own stick'
        assert out[6] == 0, 'the air-dodge mask still applies to everyone'


def test_finetune_without_a_checkpoint_is_refused(tmp_path, monkeypatch):
    """3e-5 on a random initialisation is not a fine-tune, it is a run that cannot move.
    One launched this way spent 2.2M steps producing a state-blind policy.

    ROOT is redirected because run() takes the workspace emulator lock before it reaches
    this check; without that the test fails whenever a real run is in progress, which is
    a property of the machine rather than of the code under test."""
    import argparse
    import json
    from melee_lab import worker

    monkeypatch.setattr(worker, 'ROOT', tmp_path)
    args = argparse.Namespace(
        mode='train', config=None, run_dir=str(tmp_path / 'run'), steps=1024, bootstrap=0,
        checkpoint=None, finetune=True, episodes=1, envs=1, anchor=None,
        no_curriculum=True, recurrent=False, level=None)
    assert worker.run(args) == 1          # reports failure rather than training
    status = json.loads((tmp_path / 'run' / 'status.json').read_text())
    assert status['status'] == 'failed'
    assert '--finetune' in status['error'] and '--checkpoint' in status['error']


def test_pump_never_adopts_a_frame_it_cannot_encode():
    """A match ending at the frame cap returns to the menu. Pumping across that
    transition used to leave self.state on a player-less frame, and the next step()
    crashed encoding it -- KeyError 1, 75% of the way through a training run."""
    import melee
    from melee_lab.environment import MeleeEnv

    class FakeProc:
        def poll(self): return None

    class FakeConsole:
        def __init__(self, frames):
            self.frames = list(frames)
            self._process = FakeProc()
        def step(self):
            return self.frames.pop(0) if self.frames else None

    class G:
        def __init__(self, menu, players, frame):
            self.menu_state, self.players, self.frame = menu, players, frame

    good = G(melee.Menu.IN_GAME, {1: object(), 2: object()}, 100)
    menu = G(melee.Menu.POSTGAME_SCORES, {}, 0)
    one_player = G(melee.Menu.IN_GAME, {1: object()}, 101)

    env = MeleeEnv(Config(action_set='controller', action_frames=1))
    env.state = good
    env.console = FakeConsole([menu, one_player, menu])
    env.pump(3)
    assert env.state is good, 'an unencodable frame must not replace a usable one'

    # A later in-game frame with both players is adopted normally.
    later = G(melee.Menu.IN_GAME, {1: object(), 2: object()}, 140)
    env.console = FakeConsole([menu, later])
    env.pump(2)
    assert env.state is later


def test_a_match_whose_cpu_level_did_not_take_is_caught():
    """10% of one training run was played against a level-0 CPU -- an opponent that does
    not act -- because the level was the one match-start property never asserted."""
    import re
    from pathlib import Path
    source = Path('melee_lab/environment.py').read_text()
    start = source.index('def _reset_once')
    body = source[start:source.index('def step', start)]
    # The stage and characters were already refused; the level must be too.
    assert 'Unexpected stage at match start' in body
    assert 'Unexpected characters at match start' in body
    assert 'CPU level did not take' in body
    # And it must compare against the level actually requested for this match.
    assert 'wanted_cpu' in body and 'setup_level' in body
    # The wait before starting without a confirmed level must be a named constant, so it
    # can be tuned against the stall it causes. Its VALUE is a trade-off, not an invariant:
    # every second here freezes every other slot, because stepping is vectorised.
    assert re.search(r'time_stuck>CSS_LEVEL_PATIENCE', body)
    from melee_lab.environment import CSS_LEVEL_PATIENCE
    assert CSS_LEVEL_PATIENCE > 0


def test_a_wrong_cpu_level_retries_the_menus_without_relaunching_dolphin():
    """The first version raised a bare RuntimeError, which reset() recovers from by
    relaunching the emulator -- so a menu problem cost a six-minute restart and killed a
    training run before its first step."""
    from melee_lab.environment import MeleeEnv, LevelNotSet

    env = MeleeEnv(Config(action_set='controller', action_frames=1))
    closes = []
    env.close = lambda: closes.append(1)
    env.telemetry = lambda **kw: None

    calls = {'n': 0}
    def flaky(seed=None, options=None):
        calls['n'] += 1
        if calls['n'] < 3:
            raise LevelNotSet('CPU level did not take: asked for 5, got 0.')
        return 'observation', {}
    env._reset_once = flaky
    assert env.reset() == ('observation', {})
    assert calls['n'] == 3, 'it should have walked the menus again, twice'
    assert closes == [], 'a menu problem must never relaunch the emulator'


def test_reset_hands_down_a_shrinking_retry_budget():
    """The level failure is per-slot and persistent -- env0 missed it four times running
    while env1 never did. _reset_once needs to know when retrying is pointless so it can
    play the match instead of failing the run."""
    from melee_lab.environment import MeleeEnv, LEVEL_RETRIES, LevelNotSet

    env = MeleeEnv(Config(action_set='controller', action_frames=1))
    env.close = lambda: None
    env.telemetry = lambda **kw: None
    budgets = []
    def record(seed=None, options=None):
        budgets.append(env._level_retries_left)
        raise LevelNotSet('CPU level did not take: asked for 5, got 0.')
    env._reset_once = record
    try: env.reset()
    except LevelNotSet: pass
    assert budgets == sorted(budgets, reverse=True), 'the budget must shrink'
    assert budgets[0] == LEVEL_RETRIES - 1
    assert budgets[-1] == 0, 'the last attempt must be told not to retry'


def test_connection_failures_still_relaunch_the_emulator():
    """The menu-retry path must not swallow the failures it was never meant to handle."""
    from melee_lab.environment import MeleeEnv

    env = MeleeEnv(Config(action_set='controller', action_frames=1))
    closes = []
    env.close = lambda: closes.append(1)
    env.telemetry = lambda **kw: None
    env.control = lambda: None

    calls = {'n': 0}
    def broken(seed=None, options=None):
        calls['n'] += 1
        if calls['n'] < 2:
            raise TimeoutError('No frames from Dolphin for 30 seconds.')
        return 'observation', {}
    env._reset_once = broken
    assert env.reset() == ('observation', {})
    assert closes == [1], 'a dead emulator must still be torn down'


def test_character_select_recovery_fires_before_it_starves_the_farm():
    """Stepping is vectorised, so a slot wedged in character select freezes every other
    slot for as long as it sits there. One 8-slot run had all eight log a stall.

    The recovery budget therefore has to fit well inside the setup deadline, with room
    for several attempts -- 12s to the first nudge behind a 90s deadline did not."""
    from melee_lab.environment import (CSS_NUDGE_SECONDS, CSS_NUDGE_LIMIT,
                                       RESET_REPORT_SECONDS, CSS_LEVEL_PATIENCE)

    # Sooner than the old 12s, but never under the ~6s ordinary navigation takes --
    # nudging a reset that was about to succeed drops the coin and makes it longer.
    assert 8 < CSS_NUDGE_SECONDS <= 10
    assert CSS_NUDGE_LIMIT >= 6, 'and there must be room to try repeatedly'
    # Every nudge must be spendable before the 90-second setup deadline, or the later
    # ones are decorative.
    assert CSS_NUDGE_SECONDS * CSS_NUDGE_LIMIT < 90
    # Waiting on the CPU level is itself a farm-wide stall, so it stays well under the
    # point at which we start reporting a reset as abnormal.
    assert CSS_LEVEL_PATIENCE < RESET_REPORT_SECONDS


def test_a_slow_reset_reports_where_it_is_stuck():
    """A stalled reset used to be invisible: no telemetry until it either succeeded or
    hit the 90s deadline, so the only symptom was every other emulator sitting frozen."""
    from pathlib import Path
    body = Path('melee_lab/environment.py').read_text()
    body = body[body.index('def _reset_once'):body.index('def step')]
    assert "connection='resetting'" in body
    assert 'reset_phase' in body and 'reset_waited' in body
    # It must report the live menu state, not a constant.
    assert 'g.menu_state.name' in body
