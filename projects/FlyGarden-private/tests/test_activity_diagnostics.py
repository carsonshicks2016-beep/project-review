"""Check diagnostic counters against an analytically controlled gated circuit."""
from types import SimpleNamespace
import brian2 as b
import numpy as np
from scripts.diagnose_persistent_activity import IncomingObserver, values_at


def test_delivery_observer_respects_refractory_gate_and_signed_weights():
    clock=b.Clock(dt=.1*b.ms)
    neurons=b.NeuronGroup(2, '''dv/dt=(-52*mV-v+g)/(20*ms):volt (unless refractory)
        dg/dt=-g/(5*ms):volt (unless refractory)
        rfc:second''',threshold='v>-45*mV',reset='v=-52*mV;g=0*mV',refractory='rfc',method='linear',clock=clock)
    neurons.v=[-52,-44]*b.mV;neurons.g=0*b.mV;neurons.rfc=[0,2.2]*b.ms
    synapses=b.Synapses(neurons,neurons,'w:volt',on_pre='g+=w',delay=1.8*b.ms,clock=clock)
    synapses.connect(i=[0,0],j=[1,1]);synapses.w=[5,-2]*b.mV
    stimulus=b.SpikeGeneratorGroup(1,[0,0,0],[0,.3,3.1]*b.ms,clock=clock)
    external=b.Synapses(stimulus,neurons,on_pre='v_post+=68.75*mV',clock=clock)
    external.connect(i=[0],j=[0])
    fake=SimpleNamespace(neurons=neurons,synapses=synapses,input=stimulus,input_indices=np.array([0]),clock=clock)
    observer=IncomingObserver(fake,np.array([0,1],dtype=np.int32))
    monitor=b.SpikeMonitor(neurons)
    net=b.Network(neurons,synapses,stimulus,external,monitor,*observer.objects)
    net.run(5.2*b.ms)
    delivery=observer.counters()
    # Arrival at 1.9ms is blocked; arrivals at 2.2ms and 5.0ms are accepted.
    assert np.allclose(delivery[:2,1],[10,4])
    assert np.isclose(delivery[2,0],3*68.75)
    assert int(monitor.count[1])==1
    assert int(monitor.count[0])==3


def test_pulse_and_switch_use_simulated_clock_without_recovery_input():
    condition={'cue':'odor_a','values':[.5,0,0,0,0,0,0,0]}
    assert values_at(condition,1)==[0]*8
    assert values_at(condition,2)==condition['values']
    assert values_at(condition,4)==condition['values']
    assert values_at(condition,5)==[0]*8
    switch={**condition,'cue':'switch_ab'}
    assert values_at(switch,5)==[0,.5,0,0,0,0,0,0]
    assert values_at(switch,8)==[0]*8
