"""Cell-type-supported two-root sign hypothesis; not an application controller."""
import hashlib,json
import numpy as np
import brian2 as b
from .continuous_candidate import ContinuousCandidate,file_sha

class Il3LN6SignCandidate(ContinuousCandidate):
    version='il3ln6-two-root-sign-full-network-v1'
    def __init__(self,mapping,sign_mapping,seed=0,profile=None):
        super().__init__(mapping,seed,profile)
        roots=['720575940623636701','720575940632403986']
        if sorted(sign_mapping['root_ids'])!=roots:raise ValueError('Unexpected sign-candidate roots')
        self.sign_mapping=sign_mapping
        self.rows=np.array(sign_mapping['rows'],dtype=np.int64)
        indices=np.array([self.brain.index[int(r)] for r in roots],dtype=int)
        expected=np.flatnonzero(np.isin(np.asarray(self.brain.synapses.i[:]),indices))
        if not np.array_equal(self.rows,expected):raise ValueError('Candidate must select exactly all outgoing records of two roots')
        weights=self.brain.synapses.w[self.rows].copy()
        if not np.all(weights>0*b.mV):raise ValueError('Sign candidate expects positive imported source weights')
        self.original_selected_weight_sha256=hashlib.sha256(np.asarray(weights/b.mV).tobytes()).hexdigest()
        self.brain.synapses.w[self.rows]=-weights
        self.source_hashes['il3ln6_sign_candidate.py']=file_sha(__file__)
        self.mapping_sha256=hashlib.sha256(json.dumps(sign_mapping,sort_keys=True).encode()).hexdigest()
    def manifest(self):
        m=super().manifest();m.update(signed_connectome_weights_modified=True,
          mechanism={'id':self.version,'root_ids':self.sign_mapping['root_ids'],
          'aggregate_records':len(self.rows),'anatomical_synapses':self.sign_mapping['anatomical_synapses'],
          'mapping_sha256':self.mapping_sha256,'original_selected_weight_sha256':self.original_selected_weight_sha256,
          'change':'Two exact-root outgoing weights multiplied by -1 before simulation; all absolute weights, topology, parameters and decoding retained.',
          'evidence':'Pinned exact-root GABA annotation plus il3LN6 inhibitory cell-type evidence; no target-specific receptor fit or compartment model.',
          'claim':'Source-supported sign hypothesis; not validated biological correction or navigation controller.'})
        return m
