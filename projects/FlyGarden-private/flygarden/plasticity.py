"""Engineered three-factor depression at anatomically selected KCg -> MBON connections.
Compartment mappings: Li et al. eLife 2020 doi:10.7554/eLife.62576.
Aversive gamma1pedc depression: Huang et al. Nature 2024 doi:10.1038/s41586-024-07819-w.
This is not a fitted molecular or complete dopamine model.
"""
import numpy as np
class Plasticity:
    def __init__(self,base,pre,compartment,learning_rate=.08,tau=2.):
        self.base=np.asarray(base,float).copy();self.weights=self.base.copy();self.pre=np.asarray(pre,int);self.compartment=np.asarray(compartment,int)
        self.eligibility=np.zeros(len(self.base));self.learning_rate=learning_rate;self.tau=tau;self.enabled=True;self.updates=0
    def observe(self,dt,spikes):
        self.eligibility*=np.exp(-dt/self.tau)
        self.eligibility+=np.minimum(np.asarray(spikes)[self.pre],1.)
        self.eligibility=np.minimum(self.eligibility,10.)
    def reinforce(self,event):
        if not self.enabled or event==0:return
        # Positive food modulates gamma5/MBON01; aversive event gamma1pedc/MBON11.
        selected=self.compartment==(0 if event>0 else 1)
        magnitude=np.abs(self.weights)
        magnitude[selected]-=self.learning_rate*abs(event)*np.abs(self.base[selected])*np.minimum(self.eligibility[selected],1)
        self.weights=np.sign(self.base)*np.clip(magnitude,.1*np.abs(self.base),np.abs(self.base))
        self.updates+=1
    def reset_transient(self):self.eligibility[:]=0
    def snapshot(self):return dict(weights=self.weights.copy(),eligibility=self.eligibility.copy(),enabled=self.enabled,updates=self.updates)
    def restore(self,s):
        for k in ('weights','eligibility'):setattr(self,k,np.asarray(s[k]).copy())
        self.enabled=s['enabled'];self.updates=s['updates']
