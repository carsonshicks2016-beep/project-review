"""Small contract checks complement the separate full-network audit."""
import numpy as np
import brian2 as b
from scripts.audit_brain_reference import params


def test_pinned_reference_constants_and_stimulation_units():
    p = params()
    assert float(p['v_0'] / b.mV) == -52
    assert float(p['v_th'] / b.mV) == -45
    assert float(p['t_mbr'] / b.ms) == 20
    assert float(p['tau'] / b.ms) == 5
    assert float(p['t_rfc'] / b.ms) == 2.2
    assert float(p['t_dly'] / b.ms) == 1.8
    assert np.isclose(float(p['w_syn'] * p['f_poi'] / b.mV), 68.75)


def test_upstream_extra_reset_assignment_is_behaviorally_inert():
    p = params()
    clock = b.Clock(dt=.1*b.ms)
    original = b.NeuronGroup(1, p['eqs'], threshold=p['eq_th'],
                            reset=p['eq_rst'], refractory='rfc',
                            method='linear', clock=clock, namespace=p)
    corrected = b.NeuronGroup(1, p['eqs'], threshold=p['eq_th'],
                             reset='v = v_rst; g = 0 * mV', refractory='rfc',
                             method='linear', clock=b.Clock(dt=.1*b.ms), namespace=p)
    for neurons in (original, corrected):
        neurons.v = -44*b.mV
        neurons.g = 0*b.mV
        neurons.rfc = p['t_rfc']
    monitors = [b.SpikeMonitor(neurons) for neurons in (original, corrected)]
    b.Network(original, corrected, *monitors).run(3*b.ms)
    assert np.array_equal(monitors[0].t[:], monitors[1].t[:])
    assert int(monitors[0].count[0]) == int(monitors[1].count[0]) == 1
    assert np.array_equal(original.v[:], corrected.v[:])
    assert np.array_equal(original.g[:], corrected.g[:])
    assert 'w' not in original.variables  # Brian treats this assignment as a local temporary.
