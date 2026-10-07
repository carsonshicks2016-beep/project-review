"""Unpromoted candidate decoder, for measured descending rates only.

DNa02 provides ipsilateral steering evidence. The rate-to-gait transform is
engineered, not a model of DNa02 leg-by-leg control. No cue or location access.
"""
import numpy as np
class DescendingDecoder:
    version='candidate-dna02-v1'
    def __init__(self):self.motor=np.zeros(2)
    def advance(self,dt,rates):
        if not np.isfinite(dt) or dt<=0:raise ValueError('Positive finite interval required')
        values=np.array([rates[k] for k in ('DNp09_left','DNp09_right','DNa02_left','DNa02_right')],dtype=float)
        if not np.isfinite(values).all() or np.any(values<0):raise ValueError('Finite nonnegative annotated rates required')
        forward=np.clip(values[:2].mean()/100,0,1)
        turn=np.clip((values[2]-values[3])/100,-1,1)
        # Body uses right-minus-left drive for left steering; body test must verify.
        desired=np.clip([forward-.5*turn,forward+.5*turn],0,1.2)
        alpha=1-np.exp(-dt/.4481420117724551) # same .2 gain per100ms as existing adapter
        self.motor=(1-alpha)*self.motor+alpha*desired
        return self.motor.copy()
