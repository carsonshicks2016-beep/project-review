import numpy as np
import pandas as pd
import pytest
from flygarden.local_atlas_registration import role,balanced_center,translation

def test_exact_root_grouping_is_deterministic():
    root='720575940637666446'
    assert role(root)==role(int(root))
    assert role(root) in ('probe','calibration')
    with pytest.raises(ValueError):role(float(root))

def test_each_neuron_gets_one_center_vote():
    a=pd.DataFrame({'pre_root_id':['a']*100+['b','c'],'x':[0]*100+[10,20],'y':[0]*102,'z':[0]*102})
    assert np.array_equal(balanced_center(a),[10,0,0])

def test_shift_uses_only_calibration_and_rejects_sparse_data():
    class Mirror:
        def transform(self,p):return p*np.array([-1,1,1])
    left=pd.DataFrame({'pre_root_id':list('abcde')*10,'x':[10]*50,'y':[20]*50,'z':[30]*50})
    right=left.copy();right[['x','y','z']]=[-7,24,35]
    assert np.array_equal(translation(left,right,Mirror()),[3,4,5])
    assert translation(left.head(4),right,Mirror()) is None
