import pytest
from flygarden.odor_statistics import summarize


def effects(mirrored):
    rows=[]
    for i in range(20):
        side=1 if i%2==0 else -1
        heading=.2 if mirrored else side*.2
        rows.append({'seed':9601+i,'cue':'odor_left' if side==1 else 'odor_right',
            'motor_rms_vs_sensory_off':.05,'motor_rms_vs_steering_cut':.05,
            'toward_heading_vs_sensory_off':heading,'toward_heading_vs_steering_cut':heading,
            'final_xy_departure_mm':.3,'flipped_frames':0})
    return rows


def test_same_turn_for_both_cues_cannot_pass_orientation():
    result=summarize(effects(False),True)
    assert result['gates']['causal_motor_effect']
    assert result['counts']['toward_at_least_0_05_rad']==10
    assert not result['gates']['directional_orientation']
    assert not result['gates']['progression_ready']


def test_good_orientation_still_requires_all_replays_and_complete_seeds():
    assert summarize(effects(True),True)['gates']['progression_ready']
    assert not summarize(effects(True),False)['gates']['progression_ready']
    with pytest.raises(ValueError):summarize(effects(True)[:-1],True)
