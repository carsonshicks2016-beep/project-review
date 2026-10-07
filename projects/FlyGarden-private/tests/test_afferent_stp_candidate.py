import numpy as np
import brian2 as b
from flygarden.afferent_stp_candidate import release_synapses


def test_release_depletes_even_when_postsynaptic_delivery_is_refractory():
    b.start_scope();b.prefs.codegen.target='numpy'
    clock=b.Clock(dt=.1*b.ms)
    group=b.NeuronGroup(2,'dg/dt=-g/(5*ms):volt (unless refractory)\nrfc:second',
        threshold='((i==0) and (t==0*ms)) or ((i==1) and (t==1*ms))',
        reset='g=0*mV',refractory='rfc',method='exact',clock=clock)
    group.rfc=2.2*b.ms;group.rfc[0]=0*b.ms
    syn=release_synapses(group,clock,np.array([0]),np.array([1]),np.array([2.])*b.mV)
    net=b.Network(group,syn);net.run(2*b.ms)
    assert float(syn.x[0])==.76 and float(syn.u[0])==.24
    assert float(group.g[1]/b.mV)==0


def test_resting_first_delivery_matches_original_weight_and_delay_queue_restores():
    b.start_scope();b.prefs.codegen.target='numpy'
    clock=b.Clock(dt=.1*b.ms)
    group=b.NeuronGroup(2,'g:volt',threshold='(i==0) and (t==0*ms)',reset='',clock=clock)
    syn=release_synapses(group,clock,np.array([0]),np.array([1]),np.array([2.])*b.mV)
    net=b.Network(group,syn);net.run(1*b.ms)
    assert float(group.g[1]/b.mV)==0
    net.store('pending');net.run(1*b.ms)
    expected=np.array(group.g[:]/b.mV)
    assert expected.tolist()==[0,2]
    net.restore('pending');net.run(1*b.ms)
    assert np.array_equal(expected,group.g[:]/b.mV)
