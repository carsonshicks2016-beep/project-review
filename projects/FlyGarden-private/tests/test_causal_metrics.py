import numpy as np,pytest
from flygarden.causal_metrics import trajectory_metrics,paired_bootstrap

def test_heading_wrap_and_cue_direction_are_physical():
 initial={'position':[0,0,1],'heading':np.pi-.01}
 frames=[{'time':1.5,'body':{'position':[0,0,1],'heading':-np.pi+.01,'flipped':False}},{'time':3.,'body':{'position':[.02,0,1],'heading':-np.pi+.02,'flipped':False}}]
 rows=[{'applied_motor':[.02,0]}]
 right=trajectory_metrics(initial,frames,rows,'loom_right');left=trajectory_metrics(initial,frames,rows,'loom_left')
 assert np.isclose(right['away_heading_radians'],.03) and np.isclose(left['away_heading_radians'],-.03)
 assert right['stalled'] and right['motor_active'] and right['flipped_frames']==0
 assert not trajectory_metrics(initial,frames,[{'applied_motor':[0,0]}],'loom_right')['stalled']

def test_bootstrap_preserves_mirrored_balance_and_is_repeatable():
 sides=['loom_left']*10+['loom_right']*10;values=np.array([1.]*10+[3.]*10)
 a=paired_bootstrap(values,sides);assert a==paired_bootstrap(values,sides)
 assert a['mean']==2 and a['ci_97_5']==[2.,2.] and a['positive']
 zero=paired_bootstrap(np.zeros(20),sides);assert not zero['positive']
 with pytest.raises(ValueError):paired_bootstrap(np.zeros(20),['loom_left']*20)
