import numpy as np
import pytest
from flygarden.local_senses_v2 import LocalSensesV2
from flygarden.world import World


def test_v2_live_sensory_state_restore_and_version_rejection():
    frames = np.full((2, 50, 50, 3), 255, dtype=np.uint8)
    world = World()
    observation = {'antennae': [[0, .1, 1], [0, -.1, 1]]}
    original = LocalSensesV2(frames)
    for i, radius in enumerate((3, 5, 7)):
        image = frames.copy(); image[0, 25-radius:25+radius, 25-radius:25+radius] = 0
        original.sample(world, observation, image, i*.025, .025)
    clone = LocalSensesV2(frames); clone.restore(original.snapshot())
    image = frames.copy(); image[0, 16:34, 16:34] = 0
    a = original.sample(world, observation, image, .075, .025)
    b = clone.sample(world, observation, image, .075, .025)
    assert a == b
    assert a['visual_lplc2_hz'][0] > 0
    assert a['sensory_adapter'] == original.version
    bad = original.snapshot(); bad['version'] = 'different'
    with pytest.raises(ValueError, match='identity'): clone.restore(bad)
