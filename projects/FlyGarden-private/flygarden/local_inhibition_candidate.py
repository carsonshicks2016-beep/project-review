"""Full imported network plus frozen partial local-inhibition hypothesis."""
import json
from pathlib import Path
import pandas as pd
import brian2 as b
from .continuous_candidate import ContinuousCandidate,file_sha
from .local_inhibition_bridge import LocalInhibitionBridge,residual_edge_allocation
from .recording import atomic_json

class LocalInhibitionCandidate(ContinuousCandidate):
    version='local-inhibition-clock-bridge-full-network-v1'
    def __init__(self,mapping,routing,seed=0,profile=None):
        super().__init__(mapping,seed,profile)
        self.bridge=LocalInhibitionBridge(self.brain.neurons,self.brain.clock,self.brain.ids,routing)
        chosen=set(routing['root_model_indices'].values())
        graph=pd.read_parquet(Path(__file__).resolve().parents[1]/'vendor/fly-brain/data/2025_Connectivity_783.parquet',columns=['Presynaptic_Index','Postsynaptic_Index','Connectivity','Excitatory x Connectivity'])
        subset=graph[graph.Presynaptic_Index.isin(chosen)|graph.Postsynaptic_Index.isin(chosen)]
        rows,weights,audit=residual_edge_allocation(subset.reset_index(drop=True),routing,self.brain.ids)
        self.changed_rows=subset.index.to_numpy()[rows];self.brain.synapses.w[self.changed_rows]=weights*b.mV
        self.audit=audit;self.brain.network.add(*self.bridge.objects)
        for name in ['local_inhibition_candidate.py','local_inhibition_bridge.py','local_inhibition_adapter.py']:self.source_hashes[name]=file_sha(Path(__file__).parent/name)
    def manifest(self):
        m=super().manifest();m.update(signed_connectome_weights_modified=True,mechanism={'routing_sha256':self.bridge.identity,'allocation':self.audit,'filter_tau_s':self.bridge.filter_tau_s,'delay_s':self.bridge.delay_ticks*self.bridge.dt,'adapter':self.bridge.adapter.configuration(),'claim':'Partial hybrid activity hypothesis, not receptor, cable or peptide physiology; original spiking residual retained.'});return m
    def save(self,folder):
        super().save(folder);folder=Path(folder);atomic_json(folder/'local-bridge.json',self.bridge.checkpoint())
        atomic_json(folder/'integrity.json',{'files':{p.name:file_sha(p) for p in folder.iterdir() if p.is_file() and p.name!='integrity.json'}})
    def load(self,folder):
        super().load(folder);self.bridge.restore(json.loads((Path(folder)/'local-bridge.json').read_text()))
