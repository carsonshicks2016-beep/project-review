import copy
import pytest
from flygarden.causal_stimulus_world import CausalStimulusWorld


def test_visual_fixture_is_mirrored_and_cannot_capture_or_feed():
    left=CausalStimulusWorld();right=CausalStimulusWorld()
    left.configure('loom_left');right.configure('loom_right')
    left.time=right.time=.75;left.update_stimulus();right.update_stimulus()
    a,b=left.arena['predator'],right.arena['predator']
    assert a['x']==b['x'] and a['y']==-b['y']
    assert a['enabled'] and a['visual_only']
    assert left.arena['foods']==[]
    assert left.advance(.0005,{'position':[a['x'],a['y'],1.], 'flipped':False})==0
    assert left.status=='running' and left.captures==left.collected==0
    assert left.arena['predator']['state']=='scripted_visual_fixture'


def test_fixture_restores_time_and_cue_exactly():
    world=CausalStimulusWorld();world.configure('loom_left')
    world.time=.8;world.update_stimulus();state=world.snapshot()
    other=CausalStimulusWorld();other.configure('loom_right');other.restore(state)
    assert other.cue=='loom_left' and other.arena==world.arena
    with pytest.raises(ValueError):other.configure('unknown')
