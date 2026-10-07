import numpy as np
from flygarden.timed_odor_world import TimedOdorWorld
from flygarden.odor_only_senses import OdorOnlySenses


def test_timed_sources_are_mirrored_and_end_cleanly():
    left=TimedOdorWorld();right=TimedOdorWorld()
    left.configure('odor_left');right.configure('odor_right')
    obs={'position':[0,0,1],'flipped':False}
    assert not left.concentrations([0,.2,1]).any()
    for world in (left,right):world.advance(.5,obs)
    assert left.concentrations([0,.2,1])[0]==right.concentrations([0,-.2,1])[0]
    assert left.concentrations([0,.2,1])[0]>left.concentrations([0,-.2,1])[0]
    assert left.concentrations([0,.2,1])[1]==0
    left.advance(1.,obs)
    assert not left.concentrations([0,.2,1]).any()
    assert [event['kind'] for event in left.events]==['odor_onset','odor_offset']


def test_visual_rates_cannot_leak_into_odor_protocol():
    images=np.full((2,40,40,3),255,dtype=np.uint8)
    senses=OdorOnlySenses(images);world=TimedOdorWorld();world.configure('odor_left')
    obs={'antennae':[[0,.2,1],[0,-.2,1]]}
    for i,radius in enumerate((3,5,7)):
        frame=images.copy();frame[0,20-radius:20+radius,20-radius:20+radius]=0
        result=senses.sample(world,obs,frame,i*.025,.025)
        assert result['visual_lplc2_hz']==[0.,0.]
        assert not result['visual_neural_input_enabled']
    assert result['diagnostic_visual_lplc2_hz'][0]>0
