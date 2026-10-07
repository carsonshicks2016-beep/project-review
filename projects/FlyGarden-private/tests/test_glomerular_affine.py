import numpy as np
import pytest
from flygarden.glomerular_affine import balanced_moments,covariance_map,apply_affine

def test_repeating_one_neurons_sites_does_not_change_moments():
    p=np.array([[0,0,0],[2,2,2],[10,10,10]],float);r=np.array(['a','a','b'])
    first=balanced_moments(p,r)
    second=balanced_moments(np.vstack([p,p[:2]]),np.array(['a','a','b','a','a']))
    assert all(np.allclose(a,b) for a,b in zip(first,second))

def test_covariance_transport_and_identity():
    source=np.array([[5,1,0],[1,3,.2],[0,.2,2]],float)
    expected=np.array([[1.1,.1,0],[.1,.9,.05],[0,.05,1.2]])
    target=expected@source@expected.T
    actual=covariance_map(source,target)
    assert np.allclose(actual,expected,atol=1e-12)
    assert np.allclose(actual@source@actual.T,target)
    assert np.allclose(covariance_map(source,source),np.eye(3))
    assert np.linalg.det(actual)>0

def test_reject_degenerate_and_extreme_covariance_without_clamping():
    with pytest.raises(ValueError):covariance_map(np.diag([1,1,0]),np.eye(3))
    with pytest.raises(ValueError):covariance_map(np.eye(3),np.eye(3)*9)

def test_center_mapping():
    assert np.allclose(apply_affine([[1,2,3]],[1,2,3],[4,5,6],np.eye(3)),[[4,5,6]])
