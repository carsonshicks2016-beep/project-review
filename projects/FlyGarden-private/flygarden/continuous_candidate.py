"""Continuous full-network candidate; not a promoted navigation controller."""
import json,pickle,hashlib
from pathlib import Path
import numpy as np
import brian2 as b
from .brain import FullBrain
from .candidate_inputs import CandidateInputs,INPUT_KEYS,exact_input_order
from .descending import DescendingDecoder
from .recording import atomic_json,space_check

def file_sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024**2),b''):digest.update(block)
    return digest.hexdigest()

class ContinuousCandidate:
    version='continuous-supported-connectome-v1'
    def __init__(self,mapping,seed=0,profile=None,brain=None):
        self.profile=profile or CandidateInputs();self.brain=brain or FullBrain(seed=seed,learning=False,record_spikes=True)
        if not self.brain.record_spikes or self.brain.learning_enabled or self.brain.plasticity.enabled:
            raise ValueError('Candidate requires recorded spikes and frozen learning')
        self.mapping=mapping;self.targets,self.channels=exact_input_order(self.brain.ids,mapping)
        self.populations={key:np.asarray([n['index'] for n in neurons],dtype=np.int32) for key,neurons in mapping.items()}
        for key in ('DNp09_left','DNp09_right','DNa02_left','DNa02_right'):
            if key not in self.populations or not len(self.populations[key]):raise ValueError('Missing descending annotation: '+key)
        for neurons in mapping.values():
            for neuron in neurons:
                if str(self.brain.ids[neuron['index']])!=neuron['root_id']:raise ValueError('Mapped output root/index mismatch')
        self.generator=b.SpikeGeneratorGroup(len(self.targets),[],[]*b.second,clock=self.brain.clock,name='fg_candidate_inputs')
        self.stimulation=b.Synapses(self.generator,self.brain.neurons,on_pre='v_post+=68.75*mV',clock=self.brain.clock,name='fg_candidate_stimulation')
        self.stimulation.connect(i=np.arange(len(self.targets)),j=self.targets)
        self.brain.input.active=False;self.brain.inputs.active=False
        self.brain.neurons.rfc=2.2*b.ms;self.brain.neurons.rfc[self.targets]=0*b.ms
        self.brain.network.add(self.generator,self.stimulation)
        self.decoder=DescendingDecoder();self.previous=np.zeros(self.brain.n,dtype=np.int64)
        self.source_hashes={name:file_sha(Path(__file__).parent/name) for name in
                            ('continuous_candidate.py','candidate_inputs.py','brain.py','descending.py')}
        self.last={};self.last_spikes=(np.array([],dtype=np.int32),np.array([]))

    @property
    def time(self):return float(self.brain.network.t/b.second)

    def advance(self,dt,antenna_odors,visual_lplc2_hz,sensory_enabled=True):
        clock=float(self.brain.clock.dt/b.second);steps=round(dt/clock)
        if steps<=0 or abs(steps*clock-dt)>1e-9:raise ValueError('Interval must match neural clock')
        requested=self.profile.rates(antenna_odors,visual_lplc2_hz,sensory_enabled)
        rates=np.asarray([requested[key] for key in INPUT_KEYS])
        local,ii=np.nonzero(self.brain.rng.random((steps,len(self.targets)))<rates[self.channels]*clock)
        start=self.time;events=start+local*clock
        self.generator.set_spikes(ii,events*b.second,sorted=True)
        self.brain.network.run(dt*b.second,namespace={})
        counts=np.asarray(self.brain.monitor.count[:],dtype=np.int64);delta=counts-self.previous;self.previous=counts.copy()
        self.last_spikes=(np.asarray(self.brain.monitor.i[:],dtype=np.int32).copy(),np.asarray(self.brain.monitor.t[:]/b.second).copy())
        if len(self.last_spikes[0])!=delta.sum():raise RuntimeError('Spike recording/count mismatch')
        population={key:float(delta[indices].mean()/dt) if len(indices) else None for key,indices in self.populations.items()}
        command=self.decoder.advance(dt,population)
        self.brain.previous_counts=counts.copy();self.brain.last_delta=delta.copy();self.brain.last_rates=population
        self.brain.clear_recorded_spikes()
        self.last={'start':start,'end':self.time,'requested_hz':requested,'population_hz':population,
                   'external_indices':ii.copy(),'external_times':events.copy(),'spike_counts':delta.copy(),
                   'candidate_motor':command.copy(),'sensory_enabled':bool(sensory_enabled)}
        return command

    def manifest(self):
        return {'id':self.version,'profile':self.profile.manifest(),'input_indices':self.targets.tolist(),
                'loaded_source_hashes':self.source_hashes,
                'neuron_ordering_sha256':hashlib.sha256(np.asarray(self.brain.ids,dtype=np.int64).tobytes()).hexdigest(),
                'population_mapping':{key:[{'index':int(i),'root_id':str(self.brain.ids[i])} for i in indices] for key,indices in self.populations.items()},
                'input_root_ids':[str(self.brain.ids[i]) for i in self.targets],
                'input_channels':self.channels.tolist(),'neurons':self.brain.n,'connections':self.brain.edges,
                'clock_seconds':float(self.brain.clock.dt/b.second),'external_jump_mV':68.75,
                'refractory_policy':'All possible candidate input roots zero refractory; other neurons2.2ms. Explicit engineered deviation from trial-specific reference selection.',
                'learning':False,'signed_connectome_weights_modified':False,'motor_override':False}

    def save(self,folder):
        folder=Path(folder);space_check(folder.parent,600*1024**2);folder.mkdir(exist_ok=False)
        self.brain.save(folder/'brain.state')
        with (folder/'candidate.pkl').open('wb') as stream:
            pickle.dump({'decoder_motor':self.decoder.motor.copy(),'previous':self.previous.copy()},stream,protocol=5)
        atomic_json(folder/'candidate.json',self.manifest())
        atomic_json(folder/'integrity.json',{'files':{p.name:file_sha(p) for p in folder.iterdir() if p.is_file()}})

    def load(self,folder):
        folder=Path(folder)
        receipt=json.loads((folder/'integrity.json').read_text())
        if not all(file_sha(folder/name)==digest for name,digest in receipt['files'].items()):
            raise ValueError('Candidate checkpoint integrity failure')
        if json.loads((folder/'candidate.json').read_text())!=self.manifest():raise ValueError('Candidate checkpoint identity differs')
        self.brain.load(folder/'brain.state')
        with (folder/'candidate.pkl').open('rb') as stream:state=pickle.load(stream)
        self.decoder.motor=np.asarray(state['decoder_motor']).copy();self.previous=np.asarray(state['previous']).copy()
        self.last={};self.last_spikes=(np.array([],dtype=np.int32),np.array([]))
