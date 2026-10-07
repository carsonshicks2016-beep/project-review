"""Read-only, bounded probes for interface diagnostics; not controller inputs."""
import numpy as np
import brian2 as b

class NeuralProbe:
    def __init__(self,brain,indices,input_group=None):
        self.indices=np.asarray(indices,dtype=np.int32)
        if len(set(self.indices.tolist()))!=len(self.indices) or np.any(self.indices<0) or np.any(self.indices>=brain.n):
            raise ValueError('Probe requires unique valid modeled neuron indices')
        self.brain=brain
        # End-of-tick sampling makes voltage/conductance timing explicit.
        self.state=b.StateMonitor(brain.neurons,('v','g'),record=self.indices,
                                  dt=1*b.ms,when='end')
        brain.network.add(self.state)
        self.inputs=b.SpikeMonitor(input_group) if input_group is not None else None
        if self.inputs is not None:brain.network.add(self.inputs)
        self.previous=np.zeros(len(self.indices),dtype=np.int64)

    def drain(self):
        counts=np.asarray(self.brain.monitor.count[self.indices],dtype=np.int64)
        delta=counts-self.previous;self.previous=counts.copy()
        data={'neuron_indices':self.indices.copy(),
              'root_ids':np.asarray(self.brain.ids)[self.indices].copy(),
              'time_seconds':np.asarray(self.state.t[:]/b.second).copy(),
              'voltage_mV':np.asarray(self.state.v[:]/b.mV).copy(),
              'net_synaptic_conductance_equivalent_mV':np.asarray(self.state.g[:]/b.mV).copy(),
              'spike_counts':delta,
              'sampling_scope':'1ms end-of-tick voltage and net g; not separate excitatory/inhibitory currents.'}
        self.state.resize(0)
        if self.inputs is not None:
            data['delivered_external_indices']=np.asarray(self.inputs.i[:],dtype=np.int32).copy()
            data['delivered_external_times_seconds']=np.asarray(self.inputs.t[:]/b.second).copy()
            self.inputs.resize(0);self.inputs.variables['N'].set_value(0)
        return data

    def close(self):
        self.brain.network.remove(self.state)
        if self.inputs is not None:self.brain.network.remove(self.inputs)
