import numpy as np
import brian2 as b
import pytest
from flygarden.adaptive_neurons import adaptive_neurons


def setup():
 b.start_scope();b.prefs.codegen.target='numpy';return b.Clock(dt=.1*b.ms)


def test_adaptation_increment_decay_during_refractory_and_no_reset_to_zero():
 clock=setup();n=adaptive_neurons(1,clock);n.v=-44*b.mV
 net=b.Network(n);net.run(.1*b.ms,namespace={});assert float(n.adapt[0]/b.mV)==pytest.approx(1.5)
 net.run(1*b.ms,namespace={});assert float(n.adapt[0]/b.mV)==pytest.approx(1.5*np.exp(-.001/.15),rel=1e-12)
 assert n.v[0]==-52*b.mV


def test_zero_increment_reference_parity_and_pending_delay_continuation():
 clock=setup();adapt=adaptive_neurons(1,clock,step_mv=0)
 original=b.NeuronGroup(1,'''dv/dt=(-52*mV-v+g)/(20*ms):volt (unless refractory)
 dg/dt=-g/(5*ms):volt (unless refractory)
 rfc:second''',threshold='v > -45*mV',reset='v=-52*mV;g=0*mV',refractory='rfc',method='linear',clock=clock)
 original.v=-52*b.mV;original.rfc=2.2*b.ms
 source=b.SpikeGeneratorGroup(1,np.zeros(30,int),np.arange(30)*2*b.ms,clock=clock)
 syn=[]
 for target in [original,adapt]:
  s=b.Synapses(source,target,on_pre='g_post+=35*mV',delay=1.8*b.ms,clock=clock);s.connect();syn.append(s)
 monitors=[b.SpikeMonitor(target) for target in [original,adapt]];net=b.Network(original,adapt,source,*syn,*monitors)
 net.run(31*b.ms,namespace={});net.store('pending');net.run(40*b.ms,namespace={})
 np.testing.assert_array_equal(monitors[0].t/b.second,monitors[1].t/b.second)
 np.testing.assert_allclose(original.v[:]/b.mV,adapt.v[:]/b.mV,atol=1e-10,rtol=0)
 expected=[np.asarray(getattr(adapt,k)[:]).copy() for k in ['v','g','adapt']]
 net.restore('pending');net.run(40*b.ms,namespace={})
 for key,value in zip(['v','g','adapt'],expected):np.testing.assert_array_equal(np.asarray(getattr(adapt,key)[:]),value)


def test_mask_and_invalid_parameters():
 clock=setup();n=adaptive_neurons(2,clock,mask=np.array([True,False]));np.testing.assert_array_equal(n.adapt_step[:]/b.mV,[1.5,0])
 with pytest.raises(ValueError):adaptive_neurons(1,clock,tau_s=0)
 with pytest.raises(ValueError):adaptive_neurons(1,clock,mask=np.array([1]))


def test_nonzero_adaptation_checkpoint_restores_exactly():
 clock=setup();n=adaptive_neurons(1,clock);source=b.SpikeGeneratorGroup(1,np.zeros(100,int),np.arange(100)*5*b.ms,clock=clock)
 syn=b.Synapses(source,n,on_pre='g_post+=100*mV',delay=1.8*b.ms,clock=clock);syn.connect();monitor=b.SpikeMonitor(n);net=b.Network(n,source,syn,monitor)
 net.run(201*b.ms,namespace={});assert n.adapt[0]>0*b.mV;net.store('pending')
 net.run(200*b.ms,namespace={});state=[np.asarray(getattr(n,k)[:]).copy() for k in ['v','g','adapt']];times=np.asarray(monitor.t[:]).copy()
 net.restore('pending');net.run(200*b.ms,namespace={})
 for key,expected in zip(['v','g','adapt'],state):np.testing.assert_array_equal(np.asarray(getattr(n,key)[:]),expected)
 np.testing.assert_array_equal(np.asarray(monitor.t[:]),times)
