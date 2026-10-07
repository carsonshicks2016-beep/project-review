import numpy as np
import pytest
from flygarden.olfactory_release_reference import ReleaseReference


def test_invalid_event_does_not_mutate_state():
    state = ReleaseReference()
    before = vars(state).copy()
    for p in (-.1, 1.1, float('nan')):
        with pytest.raises(ValueError): state.event(1, p)
        assert vars(state) == before
    with pytest.raises(ValueError): state.event(1, .5)
    assert vars(state) == before
    state.event(1)
    before = vars(state).copy()
    for time in (.99, float('nan'), float('inf')):
        with pytest.raises(ValueError): state.event(time)
        assert vars(state) == before


def test_checkpoint_is_an_independent_snapshot_and_rejects_wrong_model():
    rng = np.random.default_rng(10)
    state = ReleaseReference()
    state.event(.1, .5, rng)
    checkpoint = state.checkpoint(rng)
    original = checkpoint['state'].copy()
    state.event(.2, .5, rng)
    assert checkpoint['state'] == original
    checkpoint['model'] = 'different'
    with pytest.raises(ValueError): ReleaseReference.restore(checkpoint, rng)
