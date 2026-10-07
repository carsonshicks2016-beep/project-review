from types import SimpleNamespace
import numpy as np
import brian2 as b
from flygarden.candidate_inputs import INPUT_KEYS,CandidateInputs
from flygarden.continuous_candidate import ContinuousCandidate

def test_continuous_state_and_root_mapped_spikes():
    b.start_scope();clock=b.Clock(dt=.1*b.ms)
    neurons=b.NeuronGroup(10,'dv/dt=(-52*mV-v+g)/(20*ms):volt (unless refractory)\ndg/dt=-g/(5*ms):volt (unless refractory)\nrfc:second',
                         threshold='v>-45*mV',reset='v=-52*mV;g=0*mV',refractory='rfc',method='linear',clock=clock)
    neurons.v=-52*b.mV;neurons.g=0*b.mV;neurons.rfc=2.2*b.ms
    unused=b.SpikeGeneratorGroup(1,[],[]*b.second,clock=clock)
    old=b.Synapses(unused,neurons,on_pre='v_post+=1*mV',clock=clock);old.connect(i=0,j=0)
    monitor=b.SpikeMonitor(neurons)
    network=b.Network(neurons,unused,old,monitor)
    def clear():monitor.resize(0);monitor.variables['N'].set_value(0)
    brain=SimpleNamespace(n=10,edges=0,ids=np.arange(100,110),clock=clock,neurons=neurons,
                          input=unused,inputs=old,monitor=monitor,network=network,rng=np.random.default_rng(3),
                          record_spikes=True,learning_enabled=False,plasticity=SimpleNamespace(enabled=False),clear_recorded_spikes=clear)
    mapping={key:[{'index':i,'root_id':str(100+i)}] for i,key in enumerate(INPUT_KEYS)}
    mapping.update(DNa02_left=[{'index':8,'root_id':'108'}],DNa02_right=[{'index':9,'root_id':'109'}])
    candidate=ContinuousCandidate(mapping,profile=CandidateInputs(65.),brain=brain)
    first=candidate.advance(.025,[[0,0],[0,0]],[0,0]);counts=candidate.previous.copy()
    assert candidate.time==.025 and len(candidate.last_spikes[0])==counts.sum()
    second=candidate.advance(.025,[[0,0],[0,0]],[0,0])
    assert candidate.time==.05 and np.all(candidate.previous>=counts)
    assert np.array_equal(candidate.previous-counts,candidate.last['spike_counts'])
    assert np.isfinite(first).all() and np.isfinite(second).all()
    assert candidate.manifest()['motor_override'] is False
    assert not unused.active and not old.active
