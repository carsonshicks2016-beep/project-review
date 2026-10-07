import numpy as np
from flygarden.synaptic_probe import signed_arrivals

def test_delayed_signed_arrivals_cross_recording_boundary():
    pos,neg=signed_arrivals([3,4,3],[.0233,.0234,.025],
                            np.array([3,4]),np.array([9,9]),np.array([2.,-1.]),[9],.025,.05)
    assert pos.shape==(1,25) and pos.sum()==4 and neg.sum()==-1
    assert pos[0,0]==2 and neg[0,0]==-1 and pos[0,1]==2
