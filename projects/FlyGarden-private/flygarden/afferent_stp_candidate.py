"""Engineered STP-only full-network diagnostic, never a promoted controller."""
import hashlib
import numpy as np
import brian2 as b
from .continuous_candidate import ContinuousCandidate, file_sha


def release_synapses(neurons, clock, pre, post, weights):
    syn = b.Synapses(neurons, neurons, '''w : volt
        dx/dt = (1-x)/(100*ms) : 1 (event-driven)
        du/dt = -u/(50*ms) : 1 (event-driven)''',
        on_pre='''u += 0.24*(1-u)
        g_post += w*(u*x/0.24)
        x *= 1-u''', delay=1.8*b.ms, clock=clock, name='fg_afferent_stp')
    syn.connect(i=pre, j=post);syn.w=weights;syn.x=1;syn.u=0
    return syn


class AfferentSTPCandidate(ContinuousCandidate):
    version='engineered-afferent-stp-only-full-network-v1'

    def __init__(self,mapping,afferent_mapping,seed=0,profile=None):
        super().__init__(mapping,seed,profile)
        edges=afferent_mapping['afferent_edges']
        self.rows=np.array([e['row_position'] for e in edges],dtype=np.int64)
        pre=np.array([e['pre_index'] for e in edges],dtype=np.int32)
        post=np.array([e['post_index'] for e in edges],dtype=np.int32)
        assert len(self.rows)==351 and len(np.unique(self.rows))==351
        assert np.array_equal(self.brain.synapses.i[self.rows],pre)
        assert np.array_equal(self.brain.synapses.j[self.rows],post)
        assert all(str(self.brain.ids[i])==e['pre_root_id'] and str(self.brain.ids[j])==e['post_root_id']
                   for i,j,e in zip(pre,post,edges))
        weights=self.brain.synapses.w[self.rows].copy()
        assert np.all(np.asarray(weights/b.mV)>0)
        assert np.allclose(weights/b.mV,np.array([e['signed_synapses'] for e in edges])*.275,rtol=0,atol=1e-12)
        self.original_selected_weights=np.asarray(weights/b.mV).copy()
        # Partition delivery, not topology: old selected weights become zero;
        # their sole effective delivery is the matching stateful synapse.
        self.brain.synapses.w[self.rows]=0*b.mV
        self.release=release_synapses(self.brain.neurons,self.brain.clock,pre,post,weights)
        self.brain.network.add(self.release)
        self.mapping_hash=hashlib.sha256(str(edges).encode()).hexdigest()
        self.source_hashes['afferent_stp_candidate.py']=file_sha(__file__)

    def manifest(self):
        m=super().manifest()
        m.update(signed_connectome_weights_modified=True,
                 effective_connection_records=self.brain.edges,
                 mechanism={'id':self.version,'afferent_records':len(self.rows),
                            'edge_mapping_sha256':self.mapping_hash,
                            'original_selected_weights_sha256':hashlib.sha256(self.original_selected_weights.tobytes()).hexdigest(),
                            'U':.24,'tauD_seconds':.1,'tauF_seconds':.05,
                            'first_release_scaling':'release/U; rested first event equals original weight; later facilitation can exceed original weight',
                            'presynaptic_inhibition':False,
                            'resource_location':'one independent state pair per selected aggregated neuron-pair connection',
                            'event_time':'delayed synaptic arrival, 1.8ms after presynaptic spike',
                            'transport_partition':'351 original storage weights zero;351 matching stateful deliveries;all other storage weights unchanged',
                            'claim':'Engineered generic-parameter hypothesis, not validated DM1/DM2 physiology'} )
        return m
