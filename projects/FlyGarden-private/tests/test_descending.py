import numpy as np
import pytest
from flygarden.descending import DescendingDecoder

def rates(left=0,right=0,p9=65):
 return dict(DNp09_left=p9,DNp09_right=p9,DNa02_left=left,DNa02_right=right)

def test_measured_steering_mirror_and_zero_readout():
 left=DescendingDecoder().advance(.1,rates(50,0));right=DescendingDecoder().advance(.1,rates(0,50))
 assert np.allclose(left,right[::-1]) and left[1]>left[0]
 balanced=DescendingDecoder().advance(.1,rates(0,0));assert np.allclose(balanced,[.13,.13])
 assert np.array_equal(DescendingDecoder().advance(.1,rates(0,0,0)),[0,0])

def test_decoder_stays_bounded_and_rejects_invalid_telemetry():
 decoder=DescendingDecoder()
 for _ in range(100):motor=decoder.advance(.1,rates(10000,0,10000))
 assert np.isfinite(motor).all() and np.all(motor>=0) and np.all(motor<=1.2)
 with pytest.raises(ValueError):decoder.advance(.1,rates(float('nan'),0))
 with pytest.raises(ValueError):decoder.advance(0,rates())

def test_lateralized_cue_does_not_directly_stimulate_motor_outputs():
 from scripts.diagnose_sensorimotor_pathway import inputs
 assert np.array_equal(inputs('a_left',.4),[50,0,0,0,0,0,0,0])
 assert np.array_equal(inputs('a_right',.4),[0,50,0,0,0,0,0,0])
 assert np.array_equal(inputs('a_switch',1.4),[0,50,0,0,0,0,0,0])
 assert np.array_equal(inputs('a_left',.9),np.zeros(8))
 assert np.array_equal(inputs('walking_a_left',.9),[0,0,0,0,65,65,0,0])
