import numpy as np
import pytest
from flygarden.graded_gain_reference import GradedGainReference, run_reference, PUBLISHED_PARAMETERS


def literal_source(stimulus, b, injection):
    """Independent array-layout translation of archived MATLAB loop, 1 ms."""
    t = len(stimulus)
    v = np.zeros((4, t + 1)); a = np.zeros((2, t + 1))
    a[:, 0] = .5; v[1, 0] = 1
    w = np.array([[1, 0], [1, 0], [0, 0], [0, 0]])
    m = np.array([[0, 0, -b[6], 0], [0, 0, 0, 0], [1, 0, 0, 0], [0, 0, 0, 0]])
    for i in range(t):
        v[:, i] += injection[i]
        s = np.full(2, stimulus[i] * b[2] + b[3])
        if not any(x == 0 for x in b[4:6]): s /= 1 + v[1, i] * np.array(b[4:6])
        a[:, i + 1] = a[:, i] + (-b[1] * s * a[:, i] + (1-a[:, i])/b[0])
        u = s * a[:, i]
        v[:, i + 1] = v[:, i] + (-v[:, i] + w @ u + m @ v[:, i])/15
        v[:, i + 1] = np.maximum(0, v[:, i + 1])
        for j in range(2):
            if a[j, i+1] <= 0: a[j, i+1] = 1/b[0]
            elif a[j, i+1] > 1: a[j, i+1] = 1
    return v[:, :-1].T, v[:, -1], a[:, -1]


@pytest.mark.parametrize('pre,post', [(True, True), (False, True), (True, False), (False, False)])
def test_independent_literal_source_parity(pre, post):
    rng = np.random.default_rng(11901)
    s = rng.uniform(0, 1, 1500)
    inj = rng.uniform(0, 2, (len(s), 4))
    b = list(PUBLISHED_PARAMETERS)
    if not pre: b[4] = 0
    if not post: b[6] = 0
    observed, _, state = run_reference(s, b, inj)
    expected, v, a = literal_source(s, b, inj)
    np.testing.assert_allclose(observed, expected, rtol=0, atol=1e-12)
    np.testing.assert_allclose(state.v, v, rtol=0, atol=1e-12)
    np.testing.assert_allclose(state.resources, a, rtol=0, atol=1e-12)


def test_checkpoint_boundary_and_rejection():
    state = GradedGainReference()
    for i in range(775): state.step(.7)
    saved = state.checkpoint(); restored = GradedGainReference.restore(saved)
    for i in range(1000):
        x, y = state.step(0), restored.step(0)
        for key in x: np.testing.assert_array_equal(x[key], y[key])
    assert saved['steps'] == 775
    saved['model'] = 'wrong'
    with pytest.raises(ValueError): GradedGainReference.restore(saved)


def test_invalid_input_is_atomic():
    state = GradedGainReference(); saved = state.checkpoint()
    for value in [-1, np.nan, np.inf]:
        with pytest.raises(ValueError): state.step(value)
        assert state.checkpoint() == saved
    for inj in [[1], [0, 0, -1, 0], [0, np.nan, 0, 0]]:
        with pytest.raises(ValueError): state.step(1, inj)
        assert state.checkpoint() == saved


def test_silent_zero_baseline_and_second_pulse():
    b = list(PUBLISHED_PARAMETERS); b[3] = 0
    s = np.zeros(6000); s[300:800] = 1; s[3300:3800] = 1
    v, _, _ = run_reference(s, b)
    assert v[300:800, 0].max() > 10
    assert v[3300:3800, 0].max() > 10
    assert v[2500:3000].max() < 1e-8
    assert v[5500:6000].max() < 1e-8
