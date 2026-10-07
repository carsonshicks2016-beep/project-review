"""Window placement and multi-frame action sequences."""
import configparser
import json
import pytest
from melee_lab import layout
from melee_lab.actions import ACTIONS, frames_for
from melee_lab.config import Config
from melee_lab.environment import MeleeEnv
from test_learning import game, Controller


def test_grid_fits_on_screen_without_overlapping():
    boxes = layout.grid(6)
    assert len(boxes) == 6
    for b in boxes:
        assert b['x'] >= 0 and b['y'] >= layout.MENU_BAR
        assert b['x']+b['width'] <= layout.SCREEN[0]
        assert b['y']+b['height']+layout.TITLE_BAR <= layout.SCREEN[1]
    for i, a in enumerate(boxes):
        for b in boxes[i+1:]:
            apart = (a['x']+a['width'] <= b['x'] or b['x']+b['width'] <= a['x']
                     or a['y']+a['height']+layout.TITLE_BAR <= b['y']
                     or b['y']+b['height']+layout.TITLE_BAR <= a['y'])
            assert apart, f'windows overlap: {a} {b}'


def test_saved_layout_is_reused_and_hand_edits_win(tmp_path):
    path = tmp_path/'window-layout.json'
    first = layout.load(2, path)
    assert json.loads(path.read_text())['2'] == first
    mine = [{'x': 11, 'y': 22, 'width': 333, 'height': 244},
            {'x': 55, 'y': 66, 'width': 333, 'height': 244}]
    path.write_text(json.dumps({'2': mine}))
    assert layout.load(2, path) == mine, 'a hand-edited layout must be honoured'


def test_geometry_is_written_where_dolphin_reads_it(tmp_path):
    ini = tmp_path/'Dolphin.ini'
    ini.write_text('[Display]\nfullscreen = False\nRenderWindowXPos = -1\n')
    layout.write_geometry(ini, {'x': 508, 'y': 32, 'width': 496, 'height': 372})
    parser = configparser.ConfigParser(); parser.read(ini)
    assert parser.get('Display', 'RenderWindowXPos') == '508'
    assert parser.get('Display', 'RenderWindowYPos') == '32'
    assert parser.get('Display', 'RenderWindowWidth') == '496'
    assert parser.get('Display', 'RenderWindowHeight') == '372'
    assert parser.get('Display', 'fullscreen') == 'False', 'existing settings must survive'


def _drive(tmp_path, index):
    """Run one env step and report how many game frames it consumed."""
    env = MeleeEnv(Config(), tmp_path)
    env.state = game(); env.history.reset(env.state); env.ended = False
    env.start_frame = 0; env.match_cpu = 1
    env.controllers = [Controller(), Controller()]
    seen = {'n': 0}
    def fake_frame():
        seen['n'] += 1
        return game(frame=seen['n'])
    env._frame = fake_frame
    env.step(index)
    return seen['n']


def test_macro_actions_run_to_completion(tmp_path):
    """action_frames is a floor, not a ceiling. While it was a ceiling, a 5-frame
    wavedash was cut to 4 and became byte-identical to a plain jump."""
    names = [a.name for a in ACTIONS]
    wavedash = names.index('Wavedash right')
    jump = names.index('Jump')
    assert frames_for(wavedash) > Config().action_frames, 'fixture assumes a long macro'
    assert _drive(tmp_path, wavedash) == frames_for(wavedash)
    assert _drive(tmp_path, jump) == Config().action_frames


def test_the_tech_actions_are_distinguishable_from_a_bare_jump():
    from melee_lab.actions import apply
    class Pad:
        def __init__(self): self.b=set(); self.s={}
        def release_all(self): self.b.clear(); self.s.clear()
        def press_button(self,x): self.b.add(x.name)
        def tilt_analog(self,btn,*xy): self.s[btn.name]=xy
    def trace(i):
        out=[]
        for f in range(max(Config().action_frames, frames_for(i))):
            pad=Pad(); apply(pad,i,f); out.append((pad.s.get('BUTTON_MAIN'), tuple(sorted(pad.b))))
        return out
    names=[a.name for a in ACTIONS]
    jump=trace(names.index('Jump'))
    for tech in ('Wavedash left','Wavedash right','Short hop attack',
                 'Short hop back-air left','Short hop back-air right'):
        assert trace(names.index(tech)) != jump, f'{tech} still collapses into Jump'


def test_the_reaper_never_kills_its_own_caller(monkeypatch):
    """The worker's command line names the workspace in its arguments. A substring
    match made reap_workspace_dolphins() SIGKILL the worker on the first line of
    run(), so every run died in about a second with an empty log."""
    import os
    from melee_lab import slot
    root = '/Users/REVIEW_USER/Desktop/melee bot'
    worker = f'/opt/homebrew/bin/python -u -m melee_lab.worker --config {root}/runs/x/config.json'
    dolphin = f'{root}/vendor/Slippi Dolphin.app/Contents/MacOS/Slippi Dolphin -e game.ciso'
    unrelated = '/Applications/Firefox.app/Contents/MacOS/firefox'
    listing = f'  4242 {worker}\n  4243 {dolphin}\n  4244 {unrelated}\n'
    monkeypatch.setattr(slot.subprocess, 'run',
                        lambda *a, **k: type('R', (), {'stdout': listing})())
    killed = []
    monkeypatch.setattr(slot.os, 'kill', lambda pid, sig: killed.append(pid))
    reaped = slot.reap_workspace_dolphins(root)
    assert 4242 not in killed, 'the reaper killed the worker that called it'
    assert 4244 not in killed, 'the reaper killed an unrelated process'
    assert killed == [4243] and reaped == [4243]


def test_panel_claims_the_band_the_emulator_grid_leaves():
    """The emulators are tiled first and never moved; the companion takes what is left."""
    from melee_lab import layout
    from melee_lab.panel import remainder

    boxes = layout.grid(8)
    band = remainder(boxes)
    assert band is not None
    # Below the lowest emulator, and clear of it.
    lowest = max(b['y'] + b['height'] + layout.TITLE_BAR for b in boxes)
    assert band['y'] >= lowest
    assert band['y'] + band['height'] <= layout.SCREEN[1]
    assert band['x'] >= 0 and band['x'] + band['width'] <= layout.SCREEN[0]
    # It must not overlap any emulator.
    for b in boxes:
        overlap = (band['x'] < b['x'] + b['width'] and band['x'] + band['width'] > b['x']
                   and band['y'] < b['y'] + b['height'] + layout.TITLE_BAR
                   and band['y'] + band['height'] > b['y'])
        assert not overlap


def test_panel_declines_when_the_grid_fills_the_screen():
    """Better no window than one laid over the emulators the user arranged."""
    from melee_lab import layout
    from melee_lab.panel import remainder
    # A 2x2 grid is height-constrained and uses the whole screen.
    assert remainder(layout.grid(4)) is None
    assert remainder([dict(x=0, y=0, width=layout.SCREEN[0], height=layout.SCREEN[1])]) is None


def test_panel_uses_the_whole_screen_when_nothing_is_tiled():
    from melee_lab.panel import remainder
    band = remainder([])
    assert band['width'] > 1000 and band['height'] > 800


def test_panel_never_raises_out_of_its_own_failures(tmp_path):
    """A browser that will not start must not take the training run with it."""
    from melee_lab.panel import open_window
    assert open_window(tmp_path/'missing-run', root=tmp_path) is None
