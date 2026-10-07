"""Full-network interface localization; no production controller promotion."""
import sys,json,time,resource,argparse,subprocess,fcntl
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.diagnose_sensorimotor_pathway import mapping as odor_mapping
from scripts.diagnose_causal_timing import mapping as body_mapping
from flygarden.body_trial import sha
from flygarden.recording import atomic_json,space_check
from flygarden.power import on_ac_power
from flygarden.descending import DescendingDecoder
OUT=ROOT/'reports/brain-integration/recovery/interface-matrix'
SEEDS=(9301,9302);DT=.025;DURATION=1.5
CASES=('quiet','support','direct_left','direct_right','odor_left','odor_right',
       'supported_odor_left','supported_odor_right','loom_left','loom_right',
       'supported_loom_left','supported_loom_right','translation','touch')
SOURCES=('scripts/diagnose_recovery_interfaces.py','flygarden/brain.py','flygarden/neural_probe.py',
         'flygarden/synaptic_probe.py','flygarden/descending.py','data/annotations.tsv',
         'vendor/fly-brain/data/2025_Completeness_783.csv','vendor/fly-brain/data/2025_Connectivity_783.parquet',
         'scripts/diagnose_sensorimotor_pathway.py','scripts/diagnose_causal_timing.py',
         'reports/brain-integration/stage7-20261006/feedback-mapping.json',
         'reports/brain-integration/stage6-20261006/protocol.json',
         'reports/brain-integration/stage6-20261006/eye-stimuli/manifest.json')

def mappings():
    ids,pops,_=odor_mapping();pops.update(body_mapping())
    return ids,pops

KEYS=('ORN_DM1_left','ORN_DM1_right','DNp09_left','DNp09_right','DNa02_left','DNa02_right',
      'LPLC2_left','LPLC2_right','AN_multi_63_left','AN_multi_63_right')

def rates(cue,t,eyes=None):
    result=np.zeros(10);pulse=.3<=t<.8
    if (cue=='support' or cue.startswith('supported_') or cue.startswith('direct_')) and t>=.3:
        result[2:4]=65
    if pulse:
        if cue.endswith('odor_left'):result[0]=50
        if cue.endswith('odor_right'):result[1]=50
        if cue=='direct_left':result[4]=50
        if cue=='direct_right':result[5]=50
        if cue.endswith('loom_left') or cue.endswith('loom_right') or cue=='translation':
            assert eyes is not None
            row=eyes[min(int(t*30+1e-9),len(eyes)-1)]
            result[6:8]=[feature['lplc2_hz'] for feature in row['features']]
        if cue=='touch':result[8:10]=50
    return result

def source_hashes():return {name:sha(ROOT/name) for name in SOURCES}

def prepare():
    OUT.mkdir(parents=True,exist_ok=True);ids,pops=mappings()
    selected=sorted({n['index'] for key,value in pops.items() for n in
                     (value if key.startswith(('DN','MDN')) else value[:2])})
    protocol={'version':1,'sources':source_hashes(),'seeds':SEEDS,'cases':CASES,
              'duration':DURATION,'interval':DT,'mapping':pops,'input_keys':KEYS,
              'selected_probe_indices':selected,'selected_probe_root_ids':[str(ids[i]) for i in selected],
              'probe_selection':'All mapped descending neurons; first two exact annotated roots per other population, deterministic annotation order. Not whole-population membrane coverage.',
              'schedule':'Rest0-.3s; sensory/direct pulse.3-.8s; recovery.8-1.5s. Supported cases65Hz per DNp09 continuously after.3s.',
              'reference_parameters':'FullBrain released voltage equations and weights;68.75mV external jumps; stimulated roots zero refractory; other roots2.2ms.',
              'scope':'Full network; fixed weights and learning disabled. Diagnostic inputs only: stationary eye fixtures and synthetic touch pulses, not live navigation or physical contact validation.',
              'measurements':['All neuron spikes','Delivered external events and requested rates','1ms selected voltage/net g','Nominal positive/negative synaptic arrivals','Population rates','Fixed candidate commands'],
              'decision':'Localize input/target/downstream/decoder bottlenecks; no behavioral success gate or parameter tuning. Existing independent body calibration supplies physical operating range.'}
    # JSON normalizes tuples before comparison.
    protocol=json.loads(json.dumps(protocol));path=OUT/'protocol.json'
    if path.exists():assert json.loads(path.read_text())==protocol,'Preserve previous protocol; new version required'
    else:atomic_json(path,protocol)
    return protocol

def worker(cue,seed,folder):
    import brian2 as b
    from flygarden.brain import FullBrain
    from flygarden.neural_probe import NeuralProbe
    from flygarden.synaptic_probe import signed_arrivals
    protocol=prepare();folder.mkdir(parents=True,exist_ok=False);started=time.perf_counter()
    brain=FullBrain(seed=seed,learning=False,record_spikes=True);pops=protocol['mapping']
    groups=[np.array([n['index'] for n in pops[key]],dtype=np.int32) for key in KEYS]
    targets=np.concatenate(groups);channels=np.concatenate([np.full(len(g),i,dtype=int) for i,g in enumerate(groups)])
    assert all(str(brain.ids[n['index']])==n['root_id'] for population in pops.values() for n in population)
    gen=b.SpikeGeneratorGroup(len(targets),[],[]*b.second,clock=brain.clock)
    ext=b.Synapses(gen,brain.neurons,on_pre='v_post+=68.75*mV',clock=brain.clock);ext.connect(i=np.arange(len(targets)),j=targets)
    brain.network.add(gen,ext);brain.input.active=False;brain.inputs.active=False;brain.neurons.rfc=2.2*b.ms
    eye_name='moving_pattern' if cue=='translation' else cue.removeprefix('supported_')
    eye_path=ROOT/'reports/brain-integration/stage6-20261006/eye-stimuli'/f'{eye_name}.json'
    eyes=json.loads(eye_path.read_text()) if eye_path.exists() else None
    active=np.any([rates(cue,i*.0001,eyes)>0 for i in range(round(DURATION/.0001))],axis=0)
    brain.neurons.rfc[targets[active[channels]]]=0*b.ms
    probe=NeuralProbe(brain,protocol['selected_probe_indices'],gen)
    pre=np.asarray(brain.synapses.i[:],dtype=np.int32);post=np.asarray(brain.synapses.j[:],dtype=np.int32)
    edge_mask=np.isin(post,probe.indices);pre=pre[edge_mask];post=post[edge_mask]
    weight=np.asarray(brain.synapses.w[:]/b.mV)[edge_mask]
    previous=np.zeros(brain.n,dtype=np.int64);rng=np.random.default_rng(seed);decoder=DescendingDecoder()
    tail_i=np.array([],dtype=np.int32);tail_t=np.array([]);rows=[]
    manifest={'status':'running','sources':protocol['sources'],'seed':seed,'cue':cue,'chunks':[],
              'neurons':brain.n,'connections':brain.edges,'input_indices':targets.tolist(),
              'input_root_ids':[str(brain.ids[i]) for i in targets],
              'zero_refractory_indices':targets[active[channels]].tolist(),'learning':False,
              'eye_fixture_sha256':sha(eye_path) if eyes else None,'scope':protocol['scope']}
    atomic_json(folder/'manifest.json',manifest)
    for tick in range(round(DURATION/DT)):
        start=tick*DT;micro=start+np.arange(250)*.0001
        requested=np.asarray([rates(cue,t,eyes) for t in micro])
        local,ii=np.nonzero(rng.random((250,len(targets)))<requested[:,channels]*.0001)
        gen.set_spikes(ii,micro[local]*b.second,sorted=True);brain.network.run(DT*b.second,namespace={})
        counts=np.asarray(brain.monitor.count[:],dtype=np.int64);delta=counts-previous;previous=counts.copy()
        si=np.asarray(brain.monitor.i[:],dtype=np.int32).copy();st=np.asarray(brain.monitor.t[:]/b.second).copy()
        assert len(si)==delta.sum()
        data=probe.drain();pos,neg=signed_arrivals(np.concatenate([tail_i,si]),np.concatenate([tail_t,st]),pre,post,weight,probe.indices,start,start+DT)
        keep=st>=start+DT-.0018-1e-12;tail_i=si[keep];tail_t=st[keep]
        population={key:float(delta[[n['index'] for n in value]].mean()/DT) if value else None for key,value in pops.items()}
        motor=decoder.advance(DT,population)
        assert len(data['delivered_external_indices'])==len(ii)
        assert np.isfinite(data['voltage_mV']).all() and np.isfinite(data['net_synaptic_conductance_equivalent_mV']).all()
        file=folder/f'window-{tick:04d}.npz';space_check(folder,8*1024**2)
        arrays={key:value for key,value in data.items() if not isinstance(value,str)}
        with file.with_suffix('.tmp').open('wb') as stream:
            np.savez_compressed(stream,**arrays,counts=delta,spike_i=si,spike_t=st,
                                nominal_positive_arrival_mV=pos,nominal_negative_arrival_mV=neg,
                                requested_hz=requested,requested_times=micro,requested_event_i=ii,requested_event_t=micro[local])
        file.with_suffix('.tmp').replace(file);brain.clear_recorded_spikes()
        rows.append({'time':start+DT,'population_hz':population,'candidate_motor':motor.tolist(),'spikes':len(si),'input_events':len(ii)})
        manifest['chunks'].append({'file':file.name,'sha256':sha(file),'time':start+DT,'spikes':len(si)})
        atomic_json(folder/'manifest.json',manifest);atomic_json(folder/'bins.json',rows)
    probe.close();manifest.update(status='complete',wall_seconds=time.perf_counter()-started,
                                 peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    assert brain.plasticity.updates==0;atomic_json(folder/'manifest.json',manifest)

def run():
    with (ROOT/'.runtime/experiment.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);prepare();done=[]
        for cue in CASES:
            for seed in SEEDS:
                if not on_ac_power():atomic_json(OUT/'progress.json',{'status':'paused_for_ac_power','completed':done});return
                folder=OUT/'trials'/f'{cue}-{seed}';path=folder/'manifest.json'
                if path.exists() and json.loads(path.read_text())['status']=='complete':done.append(folder.name);continue
                if folder.exists():raise RuntimeError('Preserve interrupted attempt before retry: '+str(folder))
                atomic_json(OUT/'progress.json',{'status':'running','completed':done,'current':folder.name,'planned':28})
                log=OUT/'logs'/f'{folder.name}.log';log.parent.mkdir(exist_ok=True)
                with log.open('w') as stream:subprocess.run([sys.executable,__file__,'--cue',cue,'--seed',str(seed)],stdout=stream,stderr=subprocess.STDOUT,cwd=ROOT,check=True)
                done.append(folder.name);print(f'{len(done)}/28 {folder.name}',flush=True)
        atomic_json(OUT/'progress.json',{'status':'complete','completed':done})

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--prepare',action='store_true');parser.add_argument('--cue',choices=CASES);parser.add_argument('--seed',type=int);args=parser.parse_args()
    if args.prepare:prepare()
    elif args.cue:worker(args.cue,args.seed,OUT/'trials'/f'{args.cue}-{args.seed}')
    else:run()
