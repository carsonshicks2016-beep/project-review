"""Engineered, neuron-grouped anatomical registration; no physiology."""
import hashlib
import numpy as np

SPLIT_SALT='flygarden-local-registration-v1-783'

def role(root):
    key=str(root)
    if not key.isdigit():
        raise ValueError('Expected an exact decimal root ID')
    value=int.from_bytes(hashlib.sha256((SPLIT_SALT+':'+key).encode()).digest()[:8],'big')
    return 'probe' if value%5<2 else 'calibration'

def balanced_center(frame):
    """Median of per-ORN median positions; strong connections get no extra vote."""
    if frame.empty:
        raise ValueError('No calibration positions')
    per_root=frame.groupby('pre_root_id')[['x','y','z']].median().to_numpy()
    if not np.isfinite(per_root).all():
        raise ValueError('Nonfinite calibration coordinates')
    return np.median(per_root,axis=0)

def translation(left_calibration,right_calibration,mirror,min_roots=5,min_sites=50):
    for frame in (left_calibration,right_calibration):
        if len(frame)<min_sites or frame.pre_root_id.nunique()<min_roots:
            return None
    left=balanced_center(left_calibration)
    right=balanced_center(right_calibration)
    return right-mirror.transform(left.reshape(1,3))[0]
