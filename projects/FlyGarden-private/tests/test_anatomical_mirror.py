import numpy as np
import pytest
from flygarden.anatomical_mirror import AnatomicalMirror

SOURCE=np.array([[0,0,0],[1,0,0],[0,1,0],[0,0,1],[1,1,1]],float)

def test_published_landmark_interpolation_and_bounded_batches():
    target=SOURCE+np.array([.1,.2,.3]);target[-1]+=.1
    m=AnatomicalMirror(SOURCE,target,2,batch_size=2)
    assert np.allclose(m.warp_flipped(SOURCE),target,atol=1e-10)
    other=AnatomicalMirror(SOURCE,target,2,batch_size=99)
    assert np.allclose(m.transform(SOURCE),other.transform(SOURCE),atol=1e-10)

def test_native_flip_precedes_warp():
    m=AnatomicalMirror(SOURCE,SOURCE,2)
    expected=SOURCE.copy();expected[:,0]=2-expected[:,0]
    assert np.allclose(m.transform(SOURCE),expected,atol=1e-10)

def test_reject_nonfinite_and_mismatched_landmarks():
    with pytest.raises(ValueError):AnatomicalMirror(SOURCE,SOURCE[:2],2)
    m=AnatomicalMirror(SOURCE,SOURCE,2)
    with pytest.raises(ValueError):m.transform([[np.nan,0,0]])
