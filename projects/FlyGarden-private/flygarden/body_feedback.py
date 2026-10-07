"""Engineered anterior contact proxy for supported TwoLumps candidates.

Measured movement is diagnostic only: no validated speed/joint-to-ascending
mapping is available. Native model force units are retained without claiming a
physiological firing-rate fit. Both TLA candidates receive the same pooled cue;
side-specific receptive fields are not inferred.
"""
import numpy as np
class BodyFeedback:
 version='two-lumps-pooled-force-proxy-v1'
 def __init__(self):self.previous=None;self.previous_time=None
 def advance(self,observation,time,enabled=False):
  if not np.isfinite(time) or time<0 or self.previous_time is not None and time<=self.previous_time:raise ValueError('Increasing finite observation time required')
  contacts=np.asarray(observation['contacts'],dtype=float)
  if contacts.shape!=(18,3) or not np.isfinite(contacts).all():raise ValueError('Expected six legs, three links, xyz contact forces')
  pos=np.asarray(observation['position'],dtype=float);heading=float(observation['heading'])
  if pos.shape!=(3,) or not np.isfinite(pos).all() or not np.isfinite(heading):raise ValueError('Invalid body pose')
  direction=np.array([np.cos(heading),np.sin(heading),0.]);forces=contacts.reshape(6,3,3)
  # Pinned LEGS ordering LF, LM, LH, RF, RM, RH; sum resisting foreleg forces.
  resisting=np.maximum(0,-forces[[0,3]]@direction).sum(axis=1);pooled=float(resisting.sum())
  velocity=np.zeros(3);yaw_rate=0.
  if self.previous is not None:
   dt=time-self.previous_time;velocity=(pos-self.previous[0])/dt;yaw_rate=float(np.arctan2(np.sin(heading-self.previous[1]),np.cos(heading-self.previous[1]))/dt)
  rate=float(100*np.clip((pooled-1.2)/4.8,0,1)) if enabled else 0.
  self.previous=(pos.copy(),heading);self.previous_time=float(time)
  return {'time':float(time),'foreleg_resisting_force_model_units':resisting.tolist(),'pooled_force_model_units':pooled,'velocity_mm_per_second':velocity.tolist(),'yaw_radians_per_second':yaw_rate,'tla_hz':[rate,rate],'encoding_enabled':bool(enabled),'movement_stimulation_enabled':False}
 def snapshot(self):return {'previous':None if self.previous is None else [self.previous[0].tolist(),self.previous[1]],'previous_time':self.previous_time}
 def restore(self,state):
  p=state['previous'];t=state['previous_time']
  if p is None:
   if t is not None:raise ValueError('Inconsistent feedback state')
   self.previous=None;self.previous_time=None;return
  pos=np.asarray(p[0],dtype=float);heading=float(p[1])
  if pos.shape!=(3,) or not np.isfinite(pos).all() or not np.isfinite(heading) or not np.isfinite(t) or t<0:raise ValueError('Invalid feedback state')
  self.previous=(pos.copy(),heading);self.previous_time=float(t)
