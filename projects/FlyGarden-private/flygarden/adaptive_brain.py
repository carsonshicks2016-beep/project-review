"""Experimental adaptive LIF; frozen baseline constructor otherwise retained.

Static constructor copy from brain.py SHA256
3f3a0aa6aae99c38ad01532957232efdfffb92ab654b0f1be3d4230e87561875.
No graph, signs, input encoding or decoder changes. No local-inhibition bridge.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import brian2 as b
from .brain import FullBrain
from .adaptive_neurons import adaptive_neurons
ROOT=Path(__file__).resolve().parents[1]

class AdaptiveBrain(FullBrain):
    def __init__(self, seed=0, stimulus='p9', learning=False, record_spikes=True, adaptation_indices=(), tau_s=.150, step_mv=1.5):
        b.prefs.codegen.target = 'cython'
        b.seed(seed)
        self.rng=np.random.default_rng(seed)
        comp = pd.read_csv(ROOT/'vendor/fly-brain/data/2025_Completeness_783.csv', index_col=0)
        columns=['Presynaptic_Index','Postsynaptic_Index','Excitatory x Connectivity','Connectivity']
        con = pd.read_parquet(ROOT/'vendor/fly-brain/data/2025_Connectivity_783.parquet',columns=columns)
        self.ids = comp.index.to_numpy(dtype=np.int64)
        self.index = {int(v):i for i,v in enumerate(self.ids)}
        self.n = len(comp)
        self.edges = len(con)
        self.anatomical_synapses = int(con.Connectivity.sum())
        self.clock = b.Clock(dt=0.1*b.ms)
        mask=np.zeros(self.n,dtype=bool)
        selected=np.asarray(adaptation_indices,dtype=np.int64)
        if np.any(selected<0) or np.any(selected>=self.n) or len(np.unique(selected))!=len(selected):
            raise ValueError('Invalid exact adaptation indices')
        mask[selected]=True
        self.neurons=adaptive_neurons(self.n,self.clock,tau_s,step_mv,mask)
        self.neurons.v=-52*b.mV
        self.neurons.g=0*b.mV
        self.neurons.rfc=2.2*b.ms
        self.synapses=b.Synapses(self.neurons,self.neurons,'w:volt',on_pre='g+=w',delay=1.8*b.ms,clock=self.clock)
        self.synapses.connect(i=con.Presynaptic_Index.to_numpy(dtype=np.int32),j=con.Postsynaptic_Index.to_numpy(dtype=np.int32))
        self.synapses.w=con['Excitatory x Connectivity'].to_numpy()*0.275*b.mV
        del con
        ann=pd.read_csv(ROOT/'data/annotations.tsv',sep='\t',low_memory=False).fillna('')
        ann=ann[ann.root_id.isin(self.ids)]
        def indices(mask): return np.array([self.index[int(v)] for v in ann.loc[mask,'root_id']],dtype=np.int32)
        self.populations={
          'odor_a':indices(ann.cell_type.eq('ORN_DM1')),
          'odor_b':indices(ann.cell_type.eq('ORN_DM2')),
          'kc':indices(ann.cell_class.eq('Kenyon_Cell')),
          'mbon':indices(ann.cell_class.eq('MBON')),
          'mbon_reward':indices(ann.cell_type.eq('MBON01')),
          'mbon_aversive':indices(ann.cell_type.eq('MBON11')),
          'kcg':indices(ann.cell_type.str.startswith('KCg')),
          'pam_gamma5':indices(ann.cell_type.eq('PAM01')),
          'ppl_gamma1':indices(ann.cell_type.eq('PPL101')),
          'loom_left':indices(ann.cell_type.eq('LPLC2') & ann.side.eq('left')),
          'loom_right':indices(ann.cell_type.eq('LPLC2') & ann.side.eq('right')),
          'p9_left':indices(ann.cell_type.eq('DNp09') & ann.side.eq('left')),
          'p9_right':indices(ann.cell_type.eq('DNp09') & ann.side.eq('right')),
        }
        if not all(len(self.populations[k]) for k in ('odor_a','odor_b','p9_left','p9_right')):
            raise ValueError('Required annotated populations absent; motor/sensory mapping is blocked')
        self.input_keys=('odor_a','odor_b','p9_left','p9_right','loom_left','loom_right','pam_gamma5','ppl_gamma1')
        self.input_indices=np.concatenate([self.populations[k] for k in self.input_keys])
        self.input=b.SpikeGeneratorGroup(len(self.input_indices),np.array([],dtype=np.int32),np.array([])*b.second,clock=self.clock)
        self.inputs=b.Synapses(self.input,self.neurons,on_pre='v_post += 68.75*mV',clock=self.clock)
        self.inputs.connect(i=np.arange(len(self.input_indices)),j=self.input_indices)
        self.neurons.rfc[self.input_indices]=0*b.ms
        self.record_spikes=record_spikes
        self.monitor=b.SpikeMonitor(self.neurons,record=record_spikes)
        self.last_spikes=(np.array([],dtype=np.int32),np.array([],dtype=np.float64))
        self.network=b.Network(self.neurons,self.synapses,self.input,self.inputs,self.monitor)
        self.stimulus=stimulus
        self.previous_counts=np.zeros(self.n,dtype=np.int32)
        self.last_rates={}
        self.motor=np.array([0.,0.])
        from .plasticity import Plasticity
        pre=np.asarray(self.synapses.i[:],dtype=np.int32);post=np.asarray(self.synapses.j[:],dtype=np.int32)
        mask=np.isin(pre,self.populations['kcg']) & np.isin(post,np.concatenate([self.populations['mbon_reward'],self.populations['mbon_aversive']]))
        self.plastic_indices=np.flatnonzero(mask)
        base=np.asarray(self.synapses.w[self.plastic_indices]/b.mV)
        compartment=np.where(np.isin(post[self.plastic_indices],self.populations['mbon_reward']),0,1)
        self.plasticity=Plasticity(base,pre[self.plastic_indices],compartment)
        self.plasticity.enabled=learning
        self.learning_enabled=learning
        self.learning_block=None if len(base) else 'No annotated learning edges found'
        self.initial_rates={}
        self.last_delta=np.zeros(self.n,dtype=np.int32)
