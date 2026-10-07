"""Inactive experimental Brian2 bridge; all local kinetics are engineered.

A 5-ms exponential spike-rate estimator converts delayed spike arrivals to
normalized drive. Local outputs use the same 1.8-ms delay and increment the
original voltage-equivalent g state. Receptor/peptide dynamics are excluded.
"""
import hashlib,json
import numpy as np
import brian2 as b
from scipy.sparse import csr_matrix
from .local_inhibition_adapter import LocalInhibitionAdapter


class LocalInhibitionBridge:
    def __init__(self,neurons,clock,ids,routing,tau_s=.015,rate_scale_hz=100.,filter_tau_s=.005,delay_s=.0018):
        self.neurons=neurons;self.clock=clock;self.routing=routing
        self.dt=float(clock.dt/b.second);self.delay_ticks=round(delay_s/self.dt)
        if self.delay_ticks<1 or abs(self.delay_ticks*self.dt-delay_s)>1e-12:raise ValueError('Delay must align with clock')
        if not np.isfinite(filter_tau_s) or filter_tau_s<=0:raise ValueError('Invalid filter')
        self.filter_tau_s=float(filter_tau_s);self.rate_scale_hz=float(rate_scale_hz)
        self.identity=hashlib.sha256(json.dumps(routing,sort_keys=True).encode()).hexdigest()
        self.index={str(int(v)):i for i,v in enumerate(ids)}
        for root,index in routing['root_model_indices'].items():
            if self.index[root]!=index:raise ValueError('Exact root/model-index mismatch')
        self.coupling=np.asarray(routing['coupling'],float)
        n=len(routing['nodes']);self.adapter=LocalInhibitionAdapter(routing['nodes'],np.zeros((n,n)),tau_s,rate_scale_hz,self.dt)
        self.buffer=np.zeros((self.delay_ticks,n));self.cursor=0;self.ticks=0
        self.local=b.NeuronGroup(n,'dr/dt=-r/filter_tau:1\nfilter_tau:second (constant)',method='exact',clock=clock,name='fg_local_rate_filter*')
        self.local.filter_tau=filter_tau_s*b.second;self.local.r=0
        inputs=[r for r in routing['routes'] if r['pre_node']<0 and r['post_node']>=0]
        self.inputs=b.Synapses(neurons,self.local,'w:1',on_pre='r_post+=w',delay=delay_s*b.second,clock=clock,name='fg_local_delayed_inputs*')
        self.inputs.connect(i=[self.index[r['pre_root_id']] for r in inputs],j=[r['post_node'] for r in inputs])
        totals=np.asarray(routing['input_site_totals'],float)
        self.inputs.w=[r['original_sign']*r['site_count']/totals[r['post_node']]/rate_scale_hz/filter_tau_s for r in inputs]
        outputs=[r for r in routing['routes'] if r['pre_node']>=0 and r['post_node']<0]
        targets=sorted(set(self.index[r['post_root_id']] for r in outputs));self.targets=np.asarray(targets,int);lookup={t:i for i,t in enumerate(targets)}
        self.output_matrix=csr_matrix(([r['site_count'] for r in outputs],([lookup[self.index[r['post_root_id']]] for r in outputs],[r['pre_node'] for r in outputs])),shape=(len(targets),n))
        self.operation=b.NetworkOperation(self.tick,clock=clock,when='synapses',order=2,name='fg_local_continuous_release*')
        self.objects=(self.local,self.inputs,self.operation)

    def tick(self):
        delayed=self.buffer[self.cursor].copy()
        drive=np.asarray(self.local.r[:],float)+self.coupling@delayed
        self.adapter.step(drive)
        if len(self.targets):
            delta=-.275*self.rate_scale_hz*self.dt*(self.output_matrix@delayed)
            self.neurons.g[self.targets]=self.neurons.g[self.targets]+delta*b.mV
        self.buffer[self.cursor]=self.adapter.activity
        self.cursor=(self.cursor+1)%self.delay_ticks;self.ticks+=1

    def checkpoint(self):
        return {'version':1,'routing_sha256':self.identity,'filter_tau_s':self.filter_tau_s,'delay_ticks':self.delay_ticks,
                'adapter':self.adapter.checkpoint(),'buffer':self.buffer.tolist(),'cursor':self.cursor,'ticks':self.ticks}

    def restore(self,c):
        if (c.get('version')!=1 or c.get('routing_sha256')!=self.identity
                or c.get('filter_tau_s')!=self.filter_tau_s or c.get('delay_ticks')!=self.delay_ticks):raise ValueError('Incompatible bridge state')
        adapter=LocalInhibitionAdapter.restore(c['adapter'])
        if adapter.configuration()!=self.adapter.configuration():raise ValueError('Adapter configuration differs')
        buf=np.asarray(c['buffer'],float)
        if (buf.shape!=self.buffer.shape or not np.isfinite(buf).all() or np.any(buf<0) or np.any(buf>1)
                or type(c['cursor']) is not int or not 0<=c['cursor']<self.delay_ticks
                or type(c['ticks']) is not int or c['ticks']<0 or adapter.steps!=c['ticks']):raise ValueError('Invalid bridge state')
        self.adapter=adapter;self.buffer=buf.copy();self.cursor=c['cursor'];self.ticks=c['ticks']


def residual_edge_allocation(graph,routing,ids):
    """Every site belongs to exactly one original residual or upgraded route."""
    index={str(int(v)):i for i,v in enumerate(ids)}
    pair_rows={(int(i),int(j)):k for k,(i,j) in enumerate(zip(graph.Presynaptic_Index,graph.Postsynaptic_Index))}
    routed={};removed={}
    for r in routing['routes']:
        pair=(index[r['pre_root_id']],index[r['post_root_id']]);row=pair_rows[pair]
        routed[row]=routed.get(row,0)+r['site_count']
        if r['pre_node']>=0 or r['post_node']>=0:removed[row]=removed.get(row,0)+r['site_count']
    rows=np.array(sorted(routed),int);counts=graph.Connectivity.to_numpy();signed=graph['Excitatory x Connectivity'].to_numpy()
    assert all(counts[k]==v for k,v in routed.items()),'Pair conservation failed'
    assert np.all(np.abs(signed[rows])==counts[rows]),'Expected signed counts'
    removed_counts=np.array([removed.get(k,0) for k in rows],int);remaining=counts[rows]-removed_counts
    assert np.all(remaining>=0)
    weights=np.sign(signed[rows])*remaining*.275
    return rows,weights,{'audited_pairs':len(rows),'original_sites':int(counts[rows].sum()),'upgraded_sites':int(removed_counts.sum()),'residual_sites':int(remaining.sum()),'all_pair_counts_preserved':True}
