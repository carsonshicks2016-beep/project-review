import numpy as np,pytest
from flygarden.vision_encoding import EyeExpansion
def test_eye_sidedness_expansion_and_blanking():
 baseline=np.full((2,60,60,3),255,dtype=np.uint8);e=EyeExpansion(baseline)
 small=baseline.copy();small[0,27:33,27:33]=0;a=e.advance(small,.1)
 big=baseline.copy();big[0,22:38,22:38]=0;b=e.advance(big,.1)
 assert a[0]['lplc2_hz']>0 and b[0]['lplc2_hz']>a[0]['lplc2_hz'] and b[1]['lplc2_hz']==0
 assert e.advance(big,.1)[0]['lplc2_hz']==0
 blank=np.full_like(baseline,127);e=EyeExpansion(blank)
 assert all(f['lplc2_hz']==0 for f in e.advance(blank,.1))
def test_contraction_translation_and_global_dimming():
 baseline=np.full((2,60,60,3),255,dtype=np.uint8);e=EyeExpansion(baseline)
 big=baseline.copy();big[:,20:40,20:40]=0;e.advance(big,.1)
 small=baseline.copy();small[:,25:35,25:35]=0
 assert all(f['lplc2_hz']==0 for f in e.advance(small,.1))
 e=EyeExpansion(baseline);a=baseline.copy();a[:,20:30,20:30]=0;e.advance(a,.1);b=baseline.copy();b[:,20:30,30:40]=0
 assert all(f['lplc2_hz']==0 for f in e.advance(b,.1))
 e=EyeExpansion(baseline);assert all(f['lplc2_hz']==0 for f in e.advance(np.zeros_like(baseline),.1))
def test_invalid_frames_and_time():
 baseline=np.ones((2,10,10,3))*255;e=EyeExpansion(baseline)
 for dt in (0,-1,float('nan')):
  with pytest.raises(ValueError):e.advance(baseline,dt)
 for f in (baseline[0],baseline*np.nan,baseline+1):
  with pytest.raises(ValueError):e.advance(f,.1)
