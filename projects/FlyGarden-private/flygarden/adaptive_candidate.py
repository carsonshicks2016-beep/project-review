"""Separate adaptation-only candidate, with original graph and signs."""
import hashlib,json
from pathlib import Path
import numpy as np
import brian2 as b
from .adaptive_brain import AdaptiveBrain
from .continuous_candidate import ContinuousCandidate,file_sha

class AdaptiveCandidate(ContinuousCandidate):
    version='adaptive-lif-hypothesis-v1'
    def __init__(self,mapping,domain,seed=0,tau_s=.150,step_mv=1.5):
        self.domain=domain;self.tau_s=tau_s;self.step_mv=step_mv
        brain=AdaptiveBrain(seed=seed,learning=False,record_spikes=True,adaptation_indices=domain['indices'],tau_s=tau_s,step_mv=step_mv)
        for i,root in zip(domain['indices'],domain['root_ids']):
            if str(brain.ids[i])!=root:raise ValueError('Adaptation root/index mismatch')
        super().__init__(mapping,seed,brain=brain)
        for name in ['adaptive_candidate.py','adaptive_brain.py','adaptive_neurons.py']:self.source_hashes[name]=file_sha(Path(__file__).parent/name)
    def manifest(self):
        m=super().manifest();m['mechanism']={'id':self.version,'domain':self.domain['name'],'domain_sha256':hashlib.sha256(json.dumps(self.domain,sort_keys=True).encode()).hexdigest(),'adapted_neurons':len(self.domain['indices']),'tau_s':self.tau_s,'step_mv':self.step_mv,'subthreshold_coupling':0.,'local_inhibition_combined':False,'claim':'Engineering hypothesis; no calcium/potassium channel or individual-cell fit.'};return m
