from types import SimpleNamespace
import numpy as np
import brian2 as b
from flygarden.neural_probe import NeuralProbe

def test_probe_drains_events_without_altering_neural_state():
    b.start_scope()
    clock=b.Clock(dt=.1*b.ms)
    neurons=b.NeuronGroup(2,'v:volt\ng:volt',threshold='v>1*mV',reset='v=0*mV',clock=clock)
    neurons.v=0*b.mV;neurons.g=0*b.mV
    inputs=b.SpikeGeneratorGroup(1,[0,0],[0,2]*b.ms,clock=clock)
    syn=b.Synapses(inputs,neurons,on_pre='v_post+=2*mV',clock=clock);syn.connect(i=0,j=0)
    monitor=b.SpikeMonitor(neurons)
    network=b.Network(neurons,inputs,syn,monitor)
    brain=SimpleNamespace(n=2,neurons=neurons,network=network,monitor=monitor,ids=np.array([101,102]))
    probe=NeuralProbe(brain,[0,1],inputs)
    network.run(3*b.ms)
    before=np.asarray(neurons.v[:]/b.mV).copy();data=probe.drain()
    assert data['root_ids'].tolist()==[101,102]
    assert data['spike_counts'].tolist()==[2,0]
    assert data['delivered_external_times_seconds'].tolist()==[0,.002]
    assert np.array_equal(before,np.asarray(neurons.v[:]/b.mV))
    network.run(1*b.ms);second=probe.drain()
    assert second['spike_counts'].tolist()==[0,0]
    assert len(second['delivered_external_indices'])==0
    assert len(second['time_seconds'])==1
    probe.close()

def test_probe_presence_preserves_recurrent_spike_trajectory():
    def execute(observed):
        b.start_scope()
        clock=b.Clock(dt=.1*b.ms)
        neurons=b.NeuronGroup(3,'dv/dt=(-52*mV-v+g)/(20*ms):volt\ndg/dt=-g/(5*ms):volt',
                             threshold='v>-45*mV',reset='v=-52*mV;g=0*mV',method='linear',clock=clock)
        neurons.v=-52*b.mV;neurons.g=0*b.mV
        inputs=b.SpikeGeneratorGroup(1,[0,0,0],[0,3,6]*b.ms,clock=clock)
        stimulus=b.Synapses(inputs,neurons,on_pre='v_post+=68.75*mV',clock=clock);stimulus.connect(i=0,j=0)
        recurrent=b.Synapses(neurons,neurons,'w:volt',on_pre='g_post+=w',delay=1.8*b.ms,clock=clock)
        recurrent.connect(i=[0,1,2],j=[1,2,1]);recurrent.w=[200,200,-100]*b.mV
        monitor=b.SpikeMonitor(neurons);network=b.Network(neurons,inputs,stimulus,recurrent,monitor)
        brain=SimpleNamespace(n=3,neurons=neurons,network=network,monitor=monitor,ids=np.array([101,102,103]))
        probe=NeuralProbe(brain,[0,1,2],inputs) if observed else None
        states=[]
        for _ in range(3):
            network.run(5*b.ms)
            states.append((np.asarray(neurons.v[:]/b.mV).copy(),np.asarray(neurons.g[:]/b.mV).copy()))
            if probe:probe.drain()
        return states,np.asarray(monitor.i[:]).copy(),np.asarray(monitor.t[:]/b.second).copy()
    a,ai,at=execute(False);c,ci,ct=execute(True)
    assert np.array_equal(ai,ci) and np.array_equal(at,ct)
    assert all(np.array_equal(av,cv) and np.array_equal(ag,cg) for (av,ag),(cv,cg) in zip(a,c))
