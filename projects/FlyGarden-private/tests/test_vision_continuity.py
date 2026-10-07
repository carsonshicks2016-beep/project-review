import numpy as np
from flygarden.vision_continuity import ContinuousExpansion


def image(radius=0, x=40, y=40):
    frames = np.full((2, 80, 80, 3), 255, dtype=np.uint8)
    yy, xx = np.indices((80, 80))
    if radius: frames[0][(xx-x)**2 + (yy-y)**2 <= radius**2] = 0
    return frames


def rate(encoder, frame): return encoder.advance(frame, 1/30)[0]['lplc2_hz']


def test_appearance_static_translation_removal_and_global_dimming():
    encoder = ContinuousExpansion(image())
    assert rate(encoder, image(8)) == 0
    for x in range(40, 49): assert rate(encoder, image(8, x=x)) == 0
    assert rate(encoder, image()) == 0
    assert rate(encoder, np.zeros_like(image())) == 0


def test_continuous_expansion_and_exact_restored_history():
    encoder = ContinuousExpansion(image())
    rates = [rate(encoder, image(r)) for r in (5, 7, 9, 11)]
    assert rates[:2] == [0., 0.]
    assert rates[2] > 0 and rates[3] > 0
    clone = ContinuousExpansion(image()); clone.restore(encoder.snapshot())
    assert encoder.advance(image(13), 1/30) == clone.advance(image(13), 1/30)


def test_occlusion_shrink_does_not_signal_expansion():
    encoder = ContinuousExpansion(image())
    rate(encoder, image(12))
    for r in (10, 8, 6): assert rate(encoder, image(r)) == 0
