"""Offline delayed-event accounting; no neural parameter mutations."""
import numpy as np
from scipy.signal import lfilter
from scipy.sparse import coo_matrix

def delayed_impulses(spike_indices,spike_ticks,weights,steps,delay_ticks):
    """weights is target x source CSR; output is step x target."""
    i=np.asarray(spike_indices,int);t=np.asarray(spike_ticks,int)+delay_ticks
    if i.shape!=t.shape or np.any(i<0) or np.any(i>=weights.shape[1]):raise ValueError('Invalid source events')
    valid=(t>=0)&(t<steps)
    events=coo_matrix((np.ones(valid.sum()),(i[valid],t[valid])),shape=(weights.shape[1],steps)).tocsc()
    return (weights@events).toarray().T

def filtered_impulses(impulses,dt_s=.0001,tau_s=.005):
    if dt_s<=0 or tau_s<=0 or not np.isfinite(impulses).all():raise ValueError('Invalid filter')
    return lfilter([1.],[1.,-np.exp(-dt_s/tau_s)],impulses,axis=0)

def refractory_gate(spike_ticks,steps,refractory_ticks=22):
    ticks=np.asarray(spike_ticks,int)
    if refractory_ticks<0 or np.any(ticks<0) or np.any(ticks>=steps):raise ValueError('Invalid refractory input')
    last=np.full(steps,-10**9,dtype=np.int64);last[ticks]=ticks
    last=np.maximum.accumulate(last)
    return np.arange(steps)-last>=refractory_ticks
