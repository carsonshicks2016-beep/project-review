"""Fresh-process full-network observer parity; no controller or weight changes."""
import sys,json,argparse,subprocess,fcntl,time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from flygarden.body_trial import sha
from flygarden.recording import atomic_json,space_check
from flygarden.power import on_ac_power
OUT=ROOT/'reports/brain-integration/recovery/probe-parity'
SOURCES=('scripts/check_recovery_probe_parity.py','flygarden/brain.py','flygarden/neural_probe.py',
         'flygarden/descending.py','scripts/diagnose_sensorimotor_pathway.py','data/annotations.tsv',
         'vendor/fly-brain/data/2025_Completeness_783.csv','vendor/fly-brain/data/2025_Connectivity_783.parquet',
         'reports/brain-integration/recovery/interface-matrix/protocol.json')

def sources():return {name:sha(ROOT/name) for name in SOURCES}

def worker(observed):
    import brian2 as b
    from flygarden.brain import FullBrain
    from flygarden.neural_probe import NeuralProbe
    from flygarden.descending import DescendingDecoder
    from scripts.diagnose_sensorimotor_pathway import mapping
    folder=OUT/('observed' if observed else 'unobserved');folder.mkdir(exist_ok=False)
    started=time.perf_counter();brain=FullBrain(seed=9499,learning=False,record_spikes=True);_,pops,_=mapping()
    groups=[np.array([n['index'] for n in pops[key]],dtype=np.int32) for key in ('ORN_DM1_left','DNp09_left','DNp09_right')]
    targets=np.concatenate(groups);channel=np.concatenate([np.full(len(g),i,dtype=int) for i,g in enumerate(groups)])
    gen=b.SpikeGeneratorGroup(len(targets),[],[]*b.second,clock=brain.clock)
    ext=b.Synapses(gen,brain.neurons,on_pre='v_post+=68.75*mV',clock=brain.clock);ext.connect(i=np.arange(len(targets)),j=targets)
    brain.input.active=False;brain.inputs.active=False;brain.neurons.rfc=2.2*b.ms;brain.neurons.rfc[targets]=0*b.ms
    brain.network.add(gen,ext)
    selected=json.loads((ROOT/SOURCES[-1]).read_text())['selected_probe_indices']
    probe=NeuralProbe(brain,selected,gen) if observed else None
    rng=np.random.default_rng(9499);decoder=DescendingDecoder();previous=np.zeros(brain.n,dtype=np.int64)
    manifest={'status':'running','sources':sources(),'observed':observed,'seed':9499,'chunks':[],
              'duration':.5,'interval':.025,'neurons':brain.n,'connections':brain.edges,'learning':False,
              'scope':'Exact observer on/off trajectories under shared owned-RNG events; not native PoissonInput equivalence or behavioral competence.'}
    atomic_json(folder/'manifest.json',manifest)
    for tick in range(20):
        t=tick*.025;rate=np.array([50 if .1-1e-9<=t<.3-1e-9 else 0,65,65])
        local,ii=np.nonzero(rng.random((250,len(targets)))<rate[channel]*.0001)
        events=t+local*.0001;gen.set_spikes(ii,events*b.second,sorted=True);brain.network.run(.025*b.second,namespace={})
        counts=np.asarray(brain.monitor.count[:],dtype=np.int64);delta=counts-previous;previous=counts.copy()
        population={key:float(delta[[n['index'] for n in value]].mean()/.025) if value else None for key,value in pops.items()}
        motor=decoder.advance(.025,population)
        if probe:probe.drain()
        file=folder/f'window-{tick:03d}.npz';space_check(folder,16*1024**2)
        with file.with_suffix('.tmp').open('wb') as f:
            np.savez_compressed(f,voltage=np.asarray(brain.neurons.v[:]/b.mV),g=np.asarray(brain.neurons.g[:]/b.mV),
                                counts=counts,spike_i=np.asarray(brain.monitor.i[:]),spike_t=np.asarray(brain.monitor.t[:]/b.second),
                                input_i=ii,input_t=events,motor=motor)
        file.with_suffix('.tmp').replace(file);brain.clear_recorded_spikes()
        manifest['chunks'].append({'file':file.name,'sha256':sha(file)});atomic_json(folder/'manifest.json',manifest)
    if probe:probe.close()
    manifest.update(status='complete',wall_seconds=time.perf_counter()-started);atomic_json(folder/'manifest.json',manifest)

def run():
    with (ROOT/'.runtime/experiment.lock').open('a') as guard:
        fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if not on_ac_power():raise RuntimeError('Full-network parity requires AC power')
        OUT.mkdir(exist_ok=True);protocol={'version':1,'sources':sources(),'seed':9499,'duration':.5,'interval':.025,'criterion':'Exact all-neuron voltage,g,cumulative counts,raw spikes,external events and decoder commands across fresh processes.'}
        path=OUT/'protocol.json'
        if path.exists():assert json.loads(path.read_text())==protocol
        else:atomic_json(path,protocol)
        for name in ('unobserved','observed'):
            folder=OUT/name;path=folder/'manifest.json'
            if path.exists() and json.loads(path.read_text())['status']=='complete':continue
            if folder.exists():raise RuntimeError('Preserve incomplete parity attempt before retry')
            subprocess.run([sys.executable,__file__,'--worker',name],cwd=ROOT,check=True)
        a=json.loads((OUT/'unobserved/manifest.json').read_text());c=json.loads((OUT/'observed/manifest.json').read_text())
        assert a['sources']==c['sources']==protocol['sources'] and len(a['chunks'])==len(c['chunks'])==20
        for x,y in zip(a['chunks'],c['chunks']):
            ap=OUT/'unobserved'/x['file'];cp=OUT/'observed'/y['file'];assert sha(ap)==x['sha256'] and sha(cp)==y['sha256']
            with np.load(ap) as av,np.load(cp) as cv:
                assert set(av.files)==set(cv.files)
                for key in av.files:assert np.array_equal(av[key],cv[key]),'Observer changed '+key
        atomic_json(OUT/'results.json',{'status':'passed','full_network':True,'fresh_processes':2,'windows':20,
                                      'exact_all_neuron_state_spikes_inputs_and_commands':True,'sources':sources(),
                                      'scope':a['scope']})
        print('Full-network observer parity passed',flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--worker',choices=('observed','unobserved'));args=parser.parse_args()
    if args.worker:worker(args.worker=='observed')
    else:run()
