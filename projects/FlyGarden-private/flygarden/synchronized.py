"""Experimental causal coupling and sampling; production Body is preserved."""
from contextlib import contextmanager
import numpy as np
import mujoco as mj
from .body import Body
from flygym_demo.complex_terrain import HybridControllerObservation,apply_locomotion_action

class SynchronizedBody(Body):
 @contextmanager
 def observed_state(self):
  source=self.sim.mj_data
  if not hasattr(self,'_observed_data'):self._observed_data=mj.MjData(self.sim.mj_model)
  flags=mj.mjtState.mjSTATE_INTEGRATION;state=np.empty(mj.mj_stateSize(self.sim.mj_model,flags));mj.mj_getState(self.sim.mj_model,source,state,flags);mj.mj_setState(self.sim.mj_model,self._observed_data,state,flags);mj.mj_forward(self.sim.mj_model,self._observed_data)
  self.sim.mj_data=self._observed_data
  try:yield
  finally:self.sim.mj_data=source
 def observation(self):
  with self.observed_state():return super().observation()
 def eye_frames(self):
  with self.observed_state():return super().eye_frames()
 def advance(self,dt,motor,capture=None):
  n=round(dt/self.dt);stride=round(self.control_dt/self.dt)
  if abs(n*self.dt-dt)>1e-9 or n<=0 or n%stride:raise ValueError('Window must match physics and gait clocks')
  motor=np.asarray(motor,dtype=float)
  if motor.shape!=(2,) or not np.isfinite(motor).all():raise ValueError('Two finite motor commands required')
  motor=np.clip(motor,0,1.2)
  # Floor rather than round: short windows must not skip a 30Hz sample.
  next_sample=(int(np.floor(self.steps*self.dt*30+1e-9))+1)/30
  for _ in range(n//stride):
   obs=HybridControllerObservation.from_sim(self.sim,self.fly.name);action=self.controller.step(motor,obs);apply_locomotion_action(self.sim,self.fly.name,action);mj.mj_step(self.sim.mj_model,self.sim.mj_data,nstep=stride);self.steps+=stride
   if capture is not None and self.steps*self.dt+1e-10>=next_sample:
    capture(self.steps*self.dt,self.recording_pose());next_sample+=1/30
  if not all(np.isfinite(x).all() for x in (self.sim.mj_data.qpos,self.sim.mj_data.qvel,self.sim.mj_data.qacc)):raise RuntimeError('Non-finite body state')
  # Derived observations run on copied state; solver warm starts and gait sensing
  # must not change just because a recording/camera/control window ends.
  return self.observation()

class CausalCoupling:
 """Hold the previous command through an interval; publish its successor at end."""
 def __init__(self,dt):
  ticks=round(dt/.0001)
  if ticks<=0 or ticks%5 or abs(ticks*.0001-dt)>1e-9:raise ValueError('Interval must share 100us physics and 500us gait grids')
  self.ticks_per_window=ticks;self.ticks=0;self.motor=np.zeros(2);self.available_at=0.
 @property
 def time(self):return self.ticks*.0001
 @property
 def dt(self):return self.ticks_per_window*.0001
 def advance(self,brain_advance,body_advance):
  start=self.time;held=self.motor.copy();assert self.available_at<=start+1e-10
  successor=np.asarray(brain_advance(self.dt),dtype=float)
  if successor.shape!=(2,) or not np.isfinite(successor).all() or np.any(successor<0) or np.any(successor>1.2):raise ValueError('Invalid neural motor command')
  observation=body_advance(self.dt,held);self.ticks+=self.ticks_per_window;self.motor=successor.copy();self.available_at=self.time
  return {'start':start,'end':self.time,'applied_motor':held.tolist(),'next_motor':successor.tolist(),'next_available_at':self.available_at,'body':observation}
 def snapshot(self):return {'ticks':self.ticks,'ticks_per_window':self.ticks_per_window,'motor':self.motor.tolist(),'available_at':self.available_at}
 def restore(self,state):
  if state['ticks_per_window']!=self.ticks_per_window or state['ticks']<0:raise ValueError('Incompatible coupling clock')
  motor=np.asarray(state['motor'],dtype=float)
  if motor.shape!=(2,) or not np.isfinite(motor).all() or np.any(motor<0) or np.any(motor>1.2) or not np.isclose(state['available_at'],state['ticks']*.0001):raise ValueError('Invalid coupling state')
  self.ticks=int(state['ticks']);self.motor=motor.copy();self.available_at=float(state['available_at'])
