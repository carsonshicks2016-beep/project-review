"""Annotated sign hypotheses only; retains uniform spiking-neuron limitations."""
import json,hashlib
import numpy as np
import brian2 as b
from .continuous_candidate import ContinuousCandidate,file_sha
class PatchySignCandidate(ContinuousCandidate):
    version='patchy-sign-factorial-full-network-v1'
    def __init__(self,mapping,sign_mapping,seed=0,profile=None):
        super().__init__(mapping,seed,profile)
        self.sign_mapping=sign_mapping;self.rows=np.array(sign_mapping['rows'],dtype=np.int64)
        roots=set(sign_mapping['root_ids']);indices=[self.brain.index[int(x)] for x in roots]
        expected=np.flatnonzero(np.isin(np.asarray(self.brain.synapses.i[:]),indices))
        if not np.array_equal(expected,self.rows):raise ValueError('Exact all-outgoing root mapping required')
        w=self.brain.synapses.w[self.rows].copy()
        if not np.all(w>0*b.mV):raise ValueError('Expected positive imported weights')
        self.original_selected_weight_sha256=hashlib.sha256(np.asarray(w/b.mV).tobytes()).hexdigest();self.brain.synapses.w[self.rows]=-w
        self.source_hashes['patchy_sign_candidate.py']=file_sha(__file__);self.mapping_sha256=hashlib.sha256(json.dumps(sign_mapping,sort_keys=True).encode()).hexdigest()
    def manifest(self):
        m=super().manifest();m.update(signed_connectome_weights_modified=True,mechanism={'id':self.version,'root_ids':self.sign_mapping['root_ids'],'aggregate_records':len(self.rows),'anatomical_synapses':self.sign_mapping['anatomical_synapses'],'mapping_sha256':self.mapping_sha256,'change':'Specified exact-root outgoing signs flipped before simulation; absolute weights and all other parameters retained.','claim':'Annotation-supported sign hypothesis; nonspiking/graded release, compartmentation and peptide dynamics absent. Not validated biology or controller.'});return m
