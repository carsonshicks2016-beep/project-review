"""Full imported network; no connectivity threshold or neuron reduction."""
from pathlib import Path
import pickle
import numpy as np
import pandas as pd
import brian2 as b

ROOT = Path(__file__).resolve().parents[1]

class FullBrain:
    def __init__(self, seed=0, stimulus='p9', learning=False, record_spikes=True):
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
        # Matches released voltage/conductance equations; removes undefined 'w' from upstream neuron reset.
        self.neurons = b.NeuronGroup(self.n, '''dv/dt = (-52*mV-v+g)/(20*ms) : volt (unless refractory)
            dg/dt = -g/(5*ms) : volt (unless refractory)
            rfc : second''', threshold='v > -45*mV', reset='v=-52*mV; g=0*mV',
            refractory='rfc', method='linear',clock=self.clock)
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
    def advance(self,dt,odor=(0.,0.),p9=(0.,0.),loom=(0.,0.),reinforcement=0.):
        rates=[]
        for key,value in zip(self.input_keys,(*odor,*p9,*loom,max(0,reinforcement),max(0,-reinforcement))):
            rates.extend([float(np.clip(value,0,1))*100]*len(self.populations[key]))
        # Same per-timestep Bernoulli rule as Brian2 PoissonGroup, with owned portable RNG.
        steps=round(dt/float(self.clock.dt/b.second))
        if abs(steps*float(self.clock.dt/b.second)-dt)>1e-9:raise ValueError('Brain window must match clock')
        sample=self.rng.random((steps,len(rates))) < np.asarray(rates)*float(self.clock.dt/b.second)
        ticks,neurons=np.nonzero(sample)
        times=(float(self.network.t/b.second)+ticks*float(self.clock.dt/b.second))*b.second
        self.input.set_spikes(neurons,times,sorted=True)
        self.network.run(dt*b.second,namespace={})
        if self.record_spikes:
            self.last_spikes=(np.asarray(self.monitor.i[:],dtype=np.int32).copy(),np.asarray(self.monitor.t[:]/b.second,dtype=np.float64).copy())
            self.clear_recorded_spikes()
        counts=np.asarray(self.monitor.count[:],dtype=np.int32)
        delta=counts-self.previous_counts
        self.previous_counts=counts.copy()
        self.last_delta=delta.copy()
        self.plasticity.observe(dt,delta)
        self.last_rates={k:float(delta[v].mean()/dt) if len(v) else 0 for k,v in self.populations.items()}
        desired=np.clip(np.array([self.last_rates['p9_left'],self.last_rates['p9_right']])/100,0,1)
        self.motor=0.8*self.motor+0.2*desired
        return self.motor.copy()
    def clear_recorded_spikes(self):
        # Brian2 resize does not reset N. Counts remain cumulative for the controller.
        if self.record_spikes:
            self.monitor.resize(0);self.monitor.variables['N'].set_value(0)
    def reinforce(self,event):
        self.plasticity.reinforce(event)
        self.synapses.w[self.plastic_indices]=self.plasticity.weights*b.mV
    def learned_state(self):
        return dict(weights=self.plasticity.weights.copy(),updates=self.plasticity.updates,enabled=self.plasticity.enabled)
    def restore_learned(self,state):
        if len(state['weights'])!=len(self.plastic_indices):raise ValueError('Learned state does not match mapping')
        self.plasticity.weights=np.asarray(state['weights']).copy();self.plasticity.updates=state['updates'];self.plasticity.enabled=state['enabled']
        self.synapses.w[self.plastic_indices]=self.plasticity.weights*b.mV
    def reset_transient(self,seed=None):
        learned=self.learned_state()
        if seed is not None:self.rng=np.random.default_rng(seed)
        self.network.restore('trial_initial',restore_random_state=False)
        self.clear_recorded_spikes();self.last_spikes=(np.array([],dtype=np.int32),np.array([],dtype=np.float64))
        self.restore_learned(learned);self.plasticity.reset_transient();self.previous_counts[:]=0;self.motor[:]=0;self.last_rates={};self.last_delta[:]=0
    def mark_initial(self):self.network.store('trial_initial')
    def save(self,path):
        self.clear_recorded_spikes()
        self.network.store('checkpoint',filename=str(path))
        with open(str(path)+'.adapter','wb') as f:pickle.dump(dict(version=2,rng=self.rng.bit_generator.state,motor=self.motor,previous=self.previous_counts,plasticity=self.plasticity.snapshot(),rates=self.last_rates),f)
    def load(self,path):
        with open(str(path)+'.adapter','rb') as f:header=pickle.load(f)
        if header.get('version')!=2:raise ValueError('Checkpoint predates portable random input state; restore learned weights instead')
        self.network.restore('checkpoint',filename=str(path),restore_random_state=False)
        self.clear_recorded_spikes()
        with open(str(path)+'.adapter','rb') as f:state=pickle.load(f)
        self.rng.bit_generator.state=state['rng']
        self.previous_counts=state['previous'];self.motor=state['motor'];self.last_rates=state['rates'];self.plasticity.restore(state['plasticity'])
