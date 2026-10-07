import numpy as np
from scripts.diagnose_steering_bias import weights_mask

def test_targeted_interventions_preserve_unrelated_edges():
 pre=np.array([1,2,3,4,7,8]);post=np.array([7,8,7,8,9,7]);signed=np.array([-2,-3,5,6,7,-9]);targets=[7,8]
 assert np.array_equal(weights_mask('right_negative_off',pre,post,signed,targets,[1]),[False,True,False,False,False,False])
 assert np.array_equal(weights_mask('PS049_negative_off',pre,post,signed,targets,[1]),[True,False,False,False,False,False])
 assert np.array_equal(weights_mask('steering_feedback_off',pre,post,signed,targets,[1]),[False,False,False,False,True,True])
 assert not weights_mask('intact_broad',pre,post,signed,targets,[1]).any()

def test_counter_is_observational_and_honors_refractory_gate():
 import brian2 as b
 from types import SimpleNamespace
 from scripts.diagnose_steering_bias import SteeringObserver
 b.start_scope();b.prefs.codegen.target='numpy';clock=b.Clock(dt=.1*b.ms)
 group=b.NeuronGroup(3,'dv/dt=0*mV/ms:volt (unless refractory)\ndg/dt=0*mV/ms:volt (unless refractory)',threshold='v>1*mV',reset='v=0*mV;g=0*mV',refractory=2.2*b.ms,clock=clock)
 source=b.SpikeGeneratorGroup(1,[0,0,0],[0,.3,3]*b.ms,clock=clock)
 stim=b.Synapses(source,group,on_pre='v_post+=2*mV',clock=clock);stim.connect(i=[0],j=[0])
 recurrent=b.Synapses(group,group,'w:volt',on_pre='g_post+=w*int(not_refractory_post)',delay=1.8*b.ms,clock=clock);recurrent.connect(i=[0,0],j=[1,2]);recurrent.w=[5,-2]*b.mV
 initial=b.SpikeGeneratorGroup(2,[0,1],[0,0]*b.ms,clock=clock)
 initial_stim=b.Synapses(initial,group,on_pre='v_post+=2*mV',clock=clock);initial_stim.connect(i=[0,1],j=[1,2])
 monitor=b.SpikeMonitor(group);net=b.Network(group,source,stim,recurrent,monitor,initial,initial_stim);brain=SimpleNamespace(neurons=group,synapses=recurrent,clock=clock,network=net)
 observer=SteeringObserver(brain,[1,2]);net.run(6*b.ms)
 counts=np.asarray(monitor.count[:]);assert counts[0]==2
 assert np.allclose(observer.counters(),[[5,0],[0,2]])
