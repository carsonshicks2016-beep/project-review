import pytest
import melee
from melee_lab.environment import MeleeEnv
from melee_lab.config import Config
from test_learning import game,Controller

def _game(x=0.0, frame=0, stocks=(3, 3)):
    import melee
    g = melee.GameState(); g.frame = frame
    for i in (1, 2):
        p = melee.PlayerState(); p.stock = stocks[i-1]; p.percent = 0
        p.character = melee.Character.FOX if i == 1 else melee.Character.MARIO
        p.action = melee.Action.STANDING
        p.position.x = x + i * 10.0
        g.players[i] = p
    return g


def env_with_frames(tmp_path,frames):
    env=MeleeEnv(Config(),tmp_path)
    env.state=game();env.history.reset(env.state);env.ended=False
    env.start_frame=0;env.match_cpu=1
    env.controllers=[Controller(),Controller()]
    stream=iter(frames);env._frame=lambda:next(stream)
    return env


def test_final_stock_ends_immediately_and_never_steps_into_next_match(tmp_path):
    g=game((3,0),frame=1)
    env=env_with_frames(tmp_path,[g])
    _,r,done,trunc,info=env.step(0)
    assert done and not trunc and info['result']=='win'
    assert info['starting_stocks']==[3,3]
    with pytest.raises(RuntimeError,match='reset'): env.step(0)


def test_menu_without_ko_is_not_a_win(tmp_path):
    g=game();g.menu_state=melee.Menu.POSTGAME_SCORES
    env=env_with_frames(tmp_path,[g])
    _,r,done,trunc,info=env.step(0)
    assert trunc and not done and r==0 and info['result']=='interrupted'


def test_menu_transition_after_ko_is_recorded_as_decisive_outcome(tmp_path):
    g0=game((0,2),frame=100)
    g1=game((0,2),frame=101); g1.menu_state=melee.Menu.POSTGAME_SCORES
    env=env_with_frames(tmp_path,[g1])
    env.state=g0
    _,r,done,trunc,info=env.step(0)
    assert done and not trunc and info['result']=='loss'


def test_time_limit_is_truncation_for_ppo_bootstrap(tmp_path):
    env=env_with_frames(tmp_path,[game(frame=60)]);env.config.max_frames=60
    _,_,done,trunc,info=env.step(0)
    assert not done and trunc and info['result']=='timeout'


def test_curriculum_does_not_relabel_match_already_underway(tmp_path):
    env=env_with_frames(tmp_path,[game((3,0),frame=1)])
    env.cpu_level=2
    *_,info=env.step(0)
    assert info['cpu_level']==1


def test_stacked_observation_restarts_after_a_stall_gap():
    """Under SubprocVecEnv a slot parked behind another slot's menu navigation has its
    Dolphin advanced by the background pump for hundreds of frames with nobody acting.
    Stacking the frame after that gap onto the frame from before it implies velocities
    that never occurred, and the policy never trained on anything like it."""
    import numpy as np
    from melee_lab.environment import GAP_TOLERANCE
    from melee_lab.state import History, OBS_SIZE

    history = History()
    before = _game(x=-20.0, frame=1000)
    history.reset(before)

    # A normal step: a few frames on, the stack keeps both frames.
    near = _game(x=-18.0, frame=1000 + GAP_TOLERANCE - 1)
    paired = history.push(near)
    assert not np.array_equal(paired[:OBS_SIZE // 2], paired[OBS_SIZE // 2:])

    # After a stall the environment restarts the stack, so both halves match and the
    # implied velocity is zero rather than fabricated.
    history.reset(_game(x=60.0, frame=1000 + 400))
    fresh = history.get()
    assert np.array_equal(fresh[:OBS_SIZE // 2], fresh[OBS_SIZE // 2:])


def test_gap_tolerance_admits_servicing_but_not_a_stall():
    from melee_lab.environment import GAP_TOLERANCE
    # pump(2)/pump(4) between steps is routine and must not trip the guard.
    assert GAP_TOLERANCE >= 4
    # A slot waiting out another slot's reset advances ~20 frames a second; half a
    # second of that already dwarfs the tolerance.
    assert GAP_TOLERANCE < 20


def test_character_select_nudge_is_tried_long_before_the_relaunch():
    """libmelee's CPU-level slider grab sometimes fails and the selection never reads
    ready. Waiting out the hard deadline costs a Dolphin relaunch and returns to an
    empty character select, which makes the next reset worse -- so the cheap recovery
    has to come first, and with room to repeat."""
    from melee_lab.environment import CSS_NUDGE_SECONDS, CSS_NUDGE_LIMIT

    # Cheap recovery well inside the hard deadline, even for one emulator.
    assert CSS_NUDGE_SECONDS < 90
    # And every attempt must still fit, or the nudges never all get a turn.
    assert CSS_NUDGE_SECONDS * CSS_NUDGE_LIMIT < 90
    # Long enough that ordinary menu navigation (~6s) is never interrupted.
    assert CSS_NUDGE_SECONDS > 8


def test_reset_loop_drops_the_coin_and_rebuilds_the_helpers():
    import inspect
    from melee_lab.environment import MeleeEnv
    body = inspect.getsource(MeleeEnv._reset_once)
    # Held for several frames; a single frame's press may not register. The count is a
    # named constant rather than a literal because it had to grow from 4 to 24 once the
    # nudge started moving the cursor as well as dropping the coin.
    from melee_lab.environment import CSS_HOME_FRAMES
    assert 'nudge_frames=CSS_HOME_FRAMES' in body
    assert CSS_HOME_FRAMES >= 4
    assert 'press_button(melee.Button.BUTTON_B)' in body
    # A stale helper would walk straight back into the same stuck state.
    assert body.count('self.helpers=[melee.MenuHelper(),melee.MenuHelper()]') >= 2
    # The timer must clear whenever the screen is no longer stuck, or it fires later
    # on a reset that was progressing fine.
    assert 'else: stuck_since=None' in body


def test_the_character_select_nudge_moves_the_cursor_not_just_the_coin():
    """The wedge that cost a training run was a cursor position, not a coin state.

    libmelee toggles HMN->CPU by walking the cursor to the controller-type box
    (y=-2.2, wiggleroom 1) and pressing A every other frame. The stock counter sits in
    that same panel, so a cursor landing slightly off increments stocks instead -- observed
    climbing to x9, x10, x11 while a slot sat wedged for three minutes with the CPU level
    never taking. The old nudge pressed B and nothing else, so all eight attempts re-entered
    the identical cursor position.

    libmelee's own requirement: "All controller cursors must be above the character level
    for this to work."
    """
    from pathlib import Path
    from melee_lab.environment import CSS_HOME_FRAMES

    body = Path('melee_lab/environment.py').read_text()
    body = body[body.index('def _reset_once'):body.index('def step')]
    nudge = body[body.index('if nudge_frames>0:'):body.index('continue', body.index('if nudge_frames>0:'))]

    assert 'BUTTON_B' in nudge, 'still drop the coin'
    assert 'tilt_analog' in nudge and 'BUTTON_MAIN' in nudge, 'the cursor must actually be moved'
    # DOWNWARD. libmelee gates its whole CPU-level block on `coin_down or cursor_y < 0`,
    # and this nudge presses B, which clears the coin -- so an upward cursor lands in the
    # one state where the slider is never even attempted. Measured live: wedged slots sat
    # at cursor (-31.0, 1.2) with holding_cpu_slider False and all eight nudges spent.
    # The controller-type box libmelee navigates to is at y = -2.2.
    assert '.5,0.' in nudge.replace(' ', '') or '0.5,0.0' in nudge.replace(' ', ''), \
        'the cursor must be driven DOWN, below y=0, or libmelee skips the CPU-level logic'
    # And held long enough to travel off the panel.
    assert CSS_HOME_FRAMES >= 16, 'four frames dropped the coin without moving the cursor'


def test_a_wedged_reset_reports_the_cursor_and_slider_state():
    """Diagnosing the wedge took a screenshot of the stock counter. The telemetry should
    carry what a screenshot showed: where the cursor is, and whether the slider is held."""
    from pathlib import Path
    body = Path('melee_lab/environment.py').read_text()
    body = body[body.index('def _reset_once'):body.index('def step')]
    assert 'reset_cursor' in body
    assert 'holding_cpu_slider' in body
    assert 'is_holding_cpu_slider' in body, 'read libmelee\'s own slider-grab flag'
