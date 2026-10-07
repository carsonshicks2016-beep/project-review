import numpy as np
from flygarden.vision_background import BackgroundExpansion


def fixture(radius):
    frames=np.full((2,120,120,3),255,dtype=np.uint8)
    yy,xx=np.indices((120,120))
    if radius:frames[0][(xx-60)**2+(yy-60)**2<=radius**2]=0
    return {'rgb':frames,'valid':np.ones((2,120,120),dtype=bool)}


def test_static_and_restored_motion_state():
    original=BackgroundExpansion(fixture(0))
    assert all(f['lplc2_hz']==0 for f in original.advance(fixture(0),.025))
    clone=BackgroundExpansion(fixture(0));clone.restore(original.snapshot())
    assert original.advance(fixture(2),.025)==clone.advance(fixture(2),.025)


def test_uncertain_textureless_change_is_flagged_and_silent():
    original=BackgroundExpansion(fixture(0));large=fixture(0);large['rgb'][:]=0
    features=original.advance(large,.025)
    assert all(not f['registration']['valid'] and f['lplc2_hz']==0 for f in features)
