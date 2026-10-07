"""Phenomenological adaptive LIF hypothesis; adaptation is voltage-equivalent.

No calcium, potassium-channel, receptor or cell-specific fit is claimed.
The baseline v/g, threshold, refractory and reset conventions are preserved.
"""
import numpy as np
import brian2 as b


def adaptive_neurons(n,clock,tau_s=.150,step_mv=1.5,mask=None):
    if not np.isfinite(tau_s) or tau_s<=0 or not np.isfinite(step_mv) or step_mv<0:raise ValueError('Invalid adaptation')
    mask=np.ones(n,bool) if mask is None else np.asarray(mask)
    if mask.shape!=(n,) or mask.dtype!=np.dtype(bool):raise ValueError('Boolean neuron mask required')
    neurons=b.NeuronGroup(n,'''dv/dt=(-52*mV-v+g-adapt)/(20*ms):volt (unless refractory)
        dg/dt=-g/(5*ms):volt (unless refractory)
        dadapt/dt=-adapt/tau_adapt:volt
        tau_adapt:second (shared, constant)
        adapt_step:volt (constant)
        rfc:second''',threshold='v > -45*mV',reset='v=-52*mV;g=0*mV;adapt+=adapt_step',refractory='rfc',method='linear',clock=clock)
    neurons.v=-52*b.mV;neurons.g=0*b.mV;neurons.adapt=0*b.mV
    neurons.tau_adapt=tau_s*b.second;neurons.adapt_step=mask.astype(float)*step_mv*b.mV;neurons.rfc=2.2*b.ms
    return neurons
