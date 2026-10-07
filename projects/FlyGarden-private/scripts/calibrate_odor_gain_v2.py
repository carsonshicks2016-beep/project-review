"""Prospective short full-network localization; never promotes/tunes a controller."""
import sys,json,subprocess,fcntl,argparse,time,resource,shutil
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from flygarden.continuous_candidate import file_sha
from flygarden.recording import atomic_json,space_check
from flygarden.candidate_inputs import INPUT_KEYS
from flygarden.odor_gain_candidate import OdorGainCandidate
from scripts.diagnose_recovery_interfaces import mappings
OUT=ROOT/'reports/brain-integration/recovery/odor-gain-calibration-v2'
SOURCES=('scripts/calibrate_odor_gain_v2.py','flygarden/odor_gain_candidate.py','flygarden/continuous_candidate.py','flygarden/candidate_inputs.py','flygarden/brain.py','flygarden/descending.py','flygarden/neural_probe.py','scripts/diagnose_recovery_interfaces.py','scripts/diagnose_sensorimotor_pathway.py','scripts/diagnose_causal_timing.py','data/annotations.tsv','vendor/fly-brain/data/2025_Completeness_783.csv','vendor/fly-brain/data/2025_Connectivity_783.parquet')
SEEDS=(9901,9902)
CASES=[{'cue':cue,'support':65,'refractory':'candidate','gain':gain} for gain in (1.,5.,10.) for cue in ('none','a_left','a_right','a_both')]


def stimulus(cue,t):
    odors=np.zeros((2,2))
    if not .3<=t<.8 or cue=='none':return odors
    identity=0 if cue.startswith('a_') else 1
    if 'gradient' in cue:
        odors[:,identity]=[.405,.395] if cue.endswith('left') else [.395,.405]
    elif cue.endswith('both'):odors[:,identity]=1.
    else:odors[0 if cue.endswith('left') else 1,identity]=1.
    return odors


def label(case,seed):return f"{case['cue']}-support{case['support']}-{case['refractory']}-gain{case['gain']:g}-{seed}"


def prepare():
    OUT.mkdir(parents=True,exist_ok=True);ids,pops=mappings()
    selected=sorted({n['index'] for key,pop in pops.items() for n in
                     (pop if key.startswith(('DN','DM','MDN')) else pop[:2])})
    protocol={'schema_version':1,'seeds':list(SEEDS),'cases':CASES,'duration':1.5,'interval':.025,
      'sources':{n:file_sha(ROOT/n) for n in SOURCES},'mapping':pops,'selected_probe_indices':selected,
      'schedule':'Support65Hz from time0; no odor0-.3; constant odorA .3-.8; recovery.8-1.5. Gains1,5,10Hz per root; unilateral, bilateral and matched no odor.',
      'refractory':'candidate zero on all possible input roots; selected_input zero only on roots with nonzero scheduled stimulation. Identical external RNG draw dimensions/events within matched policy pairs.',
      'scope':'Full fixed network, synthetic diagnostic input and neural decoder only. No body movement, navigation, learning, connectivity lesions or candidate promotion.',
      'decision':'Prospective gates in DECISION.md must pass for both calibration seeds before any frozen candidate or held-out body run. Historical evaluation seeds9601..9620 and diagnostic9801/9802 excluded. No weights, decoder or support changes.'}
    path=OUT/'diagnostic-protocol.json'
    if path.exists():assert json.loads(path.read_text())==protocol
    else:
        atomic_json(path,protocol)
        for name in SOURCES:
            if name.startswith(('vendor/','data/')):continue
            dest=OUT/'diagnostic-source'/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/name,dest)
    return protocol


def worker(case_index,seed):
    import brian2 as b
    from flygarden.continuous_candidate import ContinuousCandidate
    from flygarden.neural_probe import NeuralProbe
    protocol=prepare();case=protocol['cases'][case_index];folder=OUT/'diagnostics'/label(case,seed);folder.mkdir(parents=True,exist_ok=False)
    start=time.monotonic();cand=ContinuousCandidate(protocol['mapping'],seed,OdorGainCandidate(case['support'],case['gain']))
    if case['refractory']=='selected_input':
        enabled=np.any([np.array(list(cand.profile.rates(stimulus(case['cue'],t),[0,0]).values()))>0 for t in (0,.3,.5,.8)],axis=0)
        cand.brain.neurons.rfc=2.2*b.ms;cand.brain.neurons.rfc[cand.targets[enabled[cand.channels]]]=0*b.ms
    probe=NeuralProbe(cand.brain,protocol['selected_probe_indices'],cand.generator)
    weight_sha=file_sha(ROOT/'vendor/fly-brain/data/2025_Connectivity_783.parquet')
    manifest={'status':'running','seed':seed,'case':case,'sources':protocol['sources'],'controller':cand.manifest(),
      'actual_zero_refractory_indices':np.flatnonzero(np.asarray(cand.brain.neurons.rfc[:]/b.ms)==0).tolist(),
      'data_sha256':weight_sha,'chunks':[],'learning':False,'controller_promotion':False}
    atomic_json(folder/'manifest.json',manifest);rows=[]
    try:
        for tick in range(60):
            stamp=tick*250*.0001;odor=stimulus(case['cue'],stamp);command=cand.advance(.025,odor,[0,0]);data=probe.drain()
            assert np.array_equal(data['delivered_external_indices'],cand.last['external_indices'])
            assert np.allclose(data['delivered_external_times_seconds'],cand.last['external_times'],rtol=0,atol=1e-12)
            assert np.isfinite(data['voltage_mV']).all() and np.isfinite(data['net_synaptic_conductance_equivalent_mV']).all()
            file=folder/f'window-{tick:04d}.npz';space_check(folder,8*1024**2)
            arrays={k:v for k,v in data.items() if not isinstance(v,str)}
            with file.with_suffix('.tmp').open('wb') as stream:
                np.savez_compressed(stream,**arrays,counts=cand.last['spike_counts'],spike_i=cand.last_spikes[0],spike_t=cand.last_spikes[1],external_i=cand.last['external_indices'],external_t=cand.last['external_times'])
            file.with_suffix('.tmp').replace(file)
            rows.append({'start':stamp,'end':cand.time,'antenna_odors':odor.tolist(),'requested_hz':cand.last['requested_hz'],
              'population_hz':cand.last['population_hz'],'candidate_motor':command.tolist(),'spikes':len(cand.last_spikes[0])})
            manifest['chunks'].append({'file':file.name,'sha256':file_sha(file),'end':cand.time})
            atomic_json(folder/'manifest.json',manifest);atomic_json(folder/'rows.json',rows)
        assert cand.brain.plasticity.updates==0
        manifest.update(status='complete',wall_seconds=time.monotonic()-start,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        atomic_json(folder/'manifest.json',manifest);print(label(case,seed),'complete',round(manifest['wall_seconds'],1),'s',flush=True)
    except BaseException as exc:
        manifest.update(status='interrupted',error=repr(exc));atomic_json(folder/'manifest.json',manifest);raise
    finally:probe.close()


def run():
    with (ROOT/'.runtime/experiment.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);protocol=prepare();completed=[]
        for i,case in enumerate(protocol['cases']):
            for seed in protocol['seeds']:
                name=label(case,seed);folder=OUT/'diagnostics'/name;path=folder/'manifest.json'
                if path.exists():
                    m=json.loads(path.read_text());assert m['status']=='complete' and m['sources']==protocol['sources'];completed.append(name);continue
                if folder.exists():raise RuntimeError('Incomplete diagnostic retained; new attempt must preserve it')
                space_check(OUT,200*1024**2)
                atomic_json(OUT/'diagnostic-progress.json',{'status':'running','completed':completed,'current':name,'planned':24})
                subprocess.run([sys.executable,__file__,'--case-index',str(i),'--seed',str(seed)],cwd=ROOT,check=True)
                completed.append(name)
        atomic_json(OUT/'diagnostic-progress.json',{'status':'complete','completed':completed,'planned':24})

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--prepare',action='store_true');p.add_argument('--case-index',type=int);p.add_argument('--seed',type=int);a=p.parse_args()
    if a.prepare:prepare()
    elif a.case_index is not None:worker(a.case_index,a.seed)
    else:run()
