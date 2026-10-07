"""Reconstruct nominal signed synaptic arrivals from raw presynaptic spikes.

Arrival sums are on_pre g increments, not membrane currents or voltage effects;
refractory state, resets and decay must be read from the separate state probe.
"""
import numpy as np

def signed_arrivals(spike_i,spike_t,pre,post,weight_mV,selected,start,end,delay=.0018):
    selected=np.asarray(selected);bins=round((end-start)/.001)
    positive=np.zeros((len(selected),bins));negative=np.zeros_like(positive)
    arrivals=np.asarray(spike_t)+delay
    mask=(arrivals>=start-1e-12)&(arrivals<end-1e-12)
    ids=np.asarray(spike_i)[mask];slots=np.floor((arrivals[mask]-start+1e-12)/.001).astype(int)
    for row,target in enumerate(selected):
        edges=np.flatnonzero(np.asarray(post)==target)
        for edge in edges:
            matching=slots[ids==pre[edge]]
            np.add.at(positive[row] if weight_mV[edge]>=0 else negative[row],matching,weight_mV[edge])
    return positive,negative
