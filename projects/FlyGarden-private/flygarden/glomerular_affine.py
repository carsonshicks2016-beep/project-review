"""Neuron-balanced affine anatomical candidate; no physiological parameters."""
import numpy as np

def balanced_moments(points,roots):
    points=np.asarray(points,float);roots=np.asarray(roots)
    if points.ndim!=2 or points.shape[1]!=3 or len(points)!=len(roots) or not len(points):
        raise ValueError('Expected positions with corresponding exact roots')
    if not np.isfinite(points).all():raise ValueError('Nonfinite positions')
    _,inverse,count=np.unique(roots,return_inverse=True,return_counts=True)
    weights=1/count[inverse];weights/=weights.sum()
    mean=np.sum(points*weights[:,None],axis=0)
    centered=points-mean
    return mean,(centered*weights[:,None]).T@centered

def _power(matrix,power):
    values,vectors=np.linalg.eigh(matrix)
    if values.min()<=max(1e-12,values.max()*1e-10):
        raise ValueError('Degenerate spatial covariance')
    return (vectors*(values**power))@vectors.T

def covariance_map(source,target,min_scale=.5,max_scale=2.):
    """Unique symmetric positive-definite Gaussian covariance transport.

    Maps source covariance to target without selecting rotation or stretch
    from probe results. Extreme fitted scales are rejected, never clamped.
    """
    source=np.asarray(source,float);target=np.asarray(target,float)
    if source.shape!=(3,3) or target.shape!=(3,3) or not np.isfinite(source).all() or not np.isfinite(target).all():
        raise ValueError('Expected finite 3 by 3 covariance matrices')
    if not np.allclose(source,source.T) or not np.allclose(target,target.T):
        raise ValueError('Covariance matrices must be symmetric')
    source_half=_power(source,.5);source_inv=_power(source,-.5)
    result=source_inv@_power(source_half@target@source_half,.5)@source_inv
    scales=np.linalg.eigvalsh(result)
    if scales.min()<min_scale or scales.max()>max_scale:
        raise ValueError('Fitted scale exceeds frozen bounds')
    return result

def apply_affine(points,source_center,target_center,matrix):
    return (np.asarray(points,float)-source_center)@matrix.T+target_center
