import numpy as np,pytest
from flygarden.synchronized import CausalCoupling,SynchronizedBody
from flygarden.body_feedback import BodyFeedback

def test_commands_cannot_act_before_they_are_computed_and_restore():
 s=CausalCoupling(.025);applied=[]
 def brain(dt):return [.4,.6]
 def body(dt,motor):applied.append(motor.copy());return {'time':s.time+dt}
 a=s.advance(brain,body);b=s.advance(brain,body)
 assert np.array_equal(applied[0],[0,0]) and np.array_equal(applied[1],[.4,.6])
 assert a['next_available_at']==b['start'] and a['next_motor']==b['applied_motor']
 r=CausalCoupling(.025);r.restore(s.snapshot());assert r.snapshot()==s.snapshot()
 for dt in (.0001,.0033,0):
  with pytest.raises(ValueError):CausalCoupling(dt)

def test_feedback_uses_only_measured_foreleg_forces_and_wraps_yaw():
 f=BodyFeedback();obs={'contacts':np.zeros((18,3)).tolist(),'position':[0,0,1],'heading':0}
 a=f.advance(obs,0,True);assert a['tla_hz']==[0,0]
 forces=np.zeros((18,3));forces[0,0]=-6;obs['contacts']=forces.tolist();obs['position']=[.1,0,1]
 b=f.advance(obs,.1,True);assert b['tla_hz']==[100,100] and b['velocity_mm_per_second']==[1,0,0] and not b['movement_stimulation_enabled']
 forces[:]=0;forces[3,0]=-10;obs['contacts']=forces.tolist();assert f.advance(obs,.2,True)['tla_hz']==[0,0]
 with pytest.raises(ValueError):f.advance(obs,.2,True)

def test_body_sampling_and_observation_do_not_change_physics():
 states=[];sample_times=[]
 for interval in (.1,.025):
  body=SynchronizedBody(seed=7199);frames=[]
  try:
   before=body.snapshot()['physics'].copy();body.observation();assert np.array_equal(before,body.snapshot()['physics'])
   for i in range(round(.3/interval)):
    body.advance(interval,[.4,.6],capture=lambda t,p:frames.append(t));body.observation()
   states.append(body.snapshot());sample_times.append(frames)
  finally:body.close()
 assert len(sample_times[0])==len(sample_times[1])==9 and np.array_equal(sample_times[0],sample_times[1])
 assert np.array_equal(states[0]['physics'],states[1]['physics'])
 for key in states[0]['cpg']:assert np.array_equal(states[0]['cpg'][key],states[1]['cpg'][key])

def test_contact_and_motion_history_survive_checkpoint():
 f=BodyFeedback();obs={'contacts':np.zeros((18,3)).tolist(),'position':[0,0,1],'heading':np.pi-.01}
 f.advance(obs,0,True);saved=f.snapshot();g=BodyFeedback();g.restore(saved)
 obs['position']=[.1,0,1];obs['heading']=-np.pi+.01
 a=f.advance(obs,.1,True);b=g.advance(obs,.1,True)
 assert a==b and np.isclose(a['yaw_radians_per_second'],.2)

def test_physical_checkpoint_continues_with_gait_and_wall_contact():
 body=SynchronizedBody(seed=7199,blocks=[{'x':4,'y':0,'w':.8,'h':12,'z':2}])
 try:
  for _ in range(60):body.advance(.025,[.65,.65])
  saved=body.snapshot()
  for _ in range(20):body.advance(.025,[.65,.65])
  expected=body.snapshot();body.restore(saved)
  for _ in range(20):body.advance(.025,[.65,.65])
  actual=body.snapshot();assert np.array_equal(expected['physics'],actual['physics'])
  for key in expected['cpg']:assert np.array_equal(expected['cpg'][key],actual['cpg'][key])
 finally:body.close()
