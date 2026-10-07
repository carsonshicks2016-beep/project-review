import numpy as np
from scripts.diagnose_recovery_interfaces import rates

def test_pulses_recovery_and_support_are_separate():
    assert np.array_equal(rates('quiet',.5),np.zeros(10))
    assert rates('odor_left',.5)[0]==50 and rates('odor_left',.5)[1]==0
    assert rates('odor_right',.5)[1]==50 and rates('odor_right',.5)[0]==0
    assert np.array_equal(rates('odor_left',.8),np.zeros(10))
    assert rates('supported_odor_left',1.)[2:4].tolist()==[65,65]
    assert rates('supported_odor_left',1.)[:2].tolist()==[0,0]
    assert np.array_equal(rates('direct_left',.2),np.zeros(10))
    assert rates('direct_left',.5)[4:6].tolist()==[50,0]
    assert rates('direct_right',.5)[4:6].tolist()==[0,50]

def test_visual_rates_use_recorded_frames_only_during_pulse():
    eyes=[{'features':[{'lplc2_hz':float(i)},{'lplc2_hz':0.}]} for i in range(90)]
    assert rates('loom_left',.5,eyes)[6:8].tolist()==[15,0]
    assert rates('loom_left',.8,eyes)[6:8].tolist()==[0,0]
