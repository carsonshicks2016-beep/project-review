"""Preregistered single-worker full-network STP-only hypothesis comparison."""
import sys,json,argparse,subprocess,fcntl,time,resource,shutil
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from flygarden.continuous_candidate import file_sha
from flygarden.recording import atomic_json,space_check
from scripts.diagnose_recovery_interfaces import mappings
OUT=ROOT/'reports/brain-integration/recovery/afferent-stp-full-network-v1'
MAP=ROOT/'reports/brain-integration/recovery/olfactory-spiking-mechanism-v1/mapping.json'
SEEDS=[11301,11302];DT=.025;STEPS=160
CASES=[{'model':model,'cue':cue} for model in ('original','stp_only') for cue in ('none','a_left','a_right')]
SOURCES=['scripts/compare_afferent_stp.py','flygarden/afferent_stp_candidate.py','flygarden/continuous_candidate.py','flygarden/brain.py','flygarden/candidate_inputs.py','flygarden/descending.py','scripts/diagnose_recovery_interfaces.py','scripts/diagnose_sensorimotor_pathway.py','scripts/diagnose_causal_timing.py','data/annotations.tsv','vendor/fly-brain/data/2025_Completeness_783.csv','vendor/fly-brain/data/2025_Connectivity_783.parquet',str(MAP.relative_to(ROOT))]


def prepare():
    OUT.mkdir(exist_ok=True);_,mapping=mappings()
    protocol={'version':1,'seeds':SEEDS,'cases':CASES,'duration':4.,'interval':DT,
        'sources':{n:file_sha(ROOT/n) for n in SOURCES},'mapping':mapping,
        'input':'65Hz DNp09 support throughout;50Hz odorA unilateral .3-.8s;no visual inputs;shared owned Bernoulli input draws',
        'model_change':'351 exact cognate DM1/DM2 ORN-to-lPN aggregate deliveries acquire independent x/u; generic U.24,tauD.1s,tauF.05s; normalized first-release amplitude. All nonselected weights and decoder fixed. No PI or learning.',
        'checkpoints':'At.775s save with transient resources and pending delay queues;continue to.9s;restart fresh worker and compare exact spikes/counts/external events/motor plus v/g within1e-10mV and resource state within1e-12.',
        'decision':'Descriptive two-seed mechanism screen, not population-level validation. Recovery prerequisite: both pulse trials in both seeds have final.5s DM1 PN and signed DNa02 differences within5Hz of same-model none, with pulse PN increase >=10Hz. No promotion or held-out body work without passing recovery and a separately registered direction gate.',
        'scope':'Full138639-neuron/15091983-record graph hypothesis; isolated diagnostics, application unchanged.'}
    path=OUT/'protocol.json'
    if path.exists():assert json.loads(path.read_text())==protocol
    else:
        atomic_json(path,protocol)
        for name in SOURCES:
            if name.startswith(('vendor/','data/','reports/')):continue
            dst=OUT/'source'/name;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/name,dst)
    return protocol


def candidate(protocol,case,seed):
    from flygarden.continuous_candidate import ContinuousCandidate
    from flygarden.afferent_stp_candidate import AfferentSTPCandidate
    if case['model']=='original':return ContinuousCandidate(protocol['mapping'],seed)
    return AfferentSTPCandidate(protocol['mapping'],json.loads(MAP.read_text()),seed)


def advance(cand,case,tick):
    odor=np.zeros((2,2))
    if case['cue']!='none' and 12<=tick<32:odor[0 if case['cue']=='a_left' else 1,0]=1
    cand.advance(DT,odor,[0,0])
    return {'counts':cand.last['spike_counts'],'spike_i':cand.last_spikes[0],
            'spike_t':cand.last_spikes[1],'external_i':cand.last['external_indices'],
            'external_t':cand.last['external_times'],'motor':cand.last['candidate_motor']}


def write_chunk(folder,tick,arrays):
    file=folder/f'window-{tick:04d}.npz';space_check(folder,8*1024**2)
    with file.with_suffix('.tmp').open('wb') as f:np.savez_compressed(f,**arrays)
    file.with_suffix('.tmp').replace(file)
    return {'file':file.name,'sha256':file_sha(file),'end':(tick+1)*DT}


def state(cand):
    import brian2 as b
    values={'v':np.asarray(cand.brain.neurons.v[:]/b.mV).copy(),
            'g':np.asarray(cand.brain.neurons.g[:]/b.mV).copy()}
    if hasattr(cand,'release'):
        values.update(x=np.asarray(cand.release.x[:]).copy(),u=np.asarray(cand.release.u[:]).copy(),
                      lastupdate=np.asarray(cand.release.lastupdate[:]/b.second).copy())
    return values


def worker(i,seed,resume=False):
    import brian2 as b
    protocol=prepare();case=CASES[i];folder=OUT/'trials'/f"{case['model']}-{case['cue']}-{seed}"
    start=time.monotonic();cand=candidate(protocol,case,seed)
    if resume:
        cand.load(folder/'checkpoint')
        for tick in range(31,36):
            arrays=advance(cand,case,tick)
            with np.load(folder/f'window-{tick:04d}.npz') as saved:
                assert all(np.array_equal(saved[k],v) for k,v in arrays.items()),f'Continuation mismatch at{tick}'
        current=state(cand)
        with np.load(folder/'continuation-state.npz') as saved:
            errors={k:float(np.max(np.abs(saved[k]-v))) for k,v in current.items()}
        assert all(v<=(1e-10 if k in ('v','g') else 1e-12) for k,v in errors.items())
        atomic_json(folder/'continuation-result.json',{'status':'passed','fresh_process':True,'windows':5,'errors':errors})
        print('Fresh-process continuation passed',case['model'],flush=True);return
    folder.mkdir(parents=True,exist_ok=False)
    original=np.asarray(cand.brain.synapses.w[:]/b.mV).copy()
    original_sha=__import__('hashlib').sha256(original.tobytes()).hexdigest();del original
    manifest={'status':'running','seed':seed,'case':case,'controller':cand.manifest(),
              'sources':protocol['sources'],'storage_weights_sha256':original_sha,'chunks':[]}
    atomic_json(folder/'manifest.json',manifest);rows=[]
    checkpoint_case=case['cue']=='a_left' and seed==SEEDS[0]
    try:
        for tick in range(STEPS):
            arrays=advance(cand,case,tick)
            assert np.array_equal(np.bincount(arrays['spike_i'],minlength=cand.brain.n),arrays['counts'])
            assert np.isfinite(arrays['motor']).all() and np.all((arrays['motor']>=0)&(arrays['motor']<=1.2))
            assert abs(cand.time-(tick+1)*DT)<1e-9
            rows.append({'end':cand.time,'population_hz':cand.last['population_hz'],
                         'motor':arrays['motor'].tolist(),'spikes':len(arrays['spike_i'])})
            manifest['chunks'].append(write_chunk(folder,tick,arrays))
            atomic_json(folder/'rows.json',rows);atomic_json(folder/'manifest.json',manifest)
            if checkpoint_case and tick==30:cand.save(folder/'checkpoint')
            if checkpoint_case and tick==35:
                with (folder/'continuation-state.npz').open('wb') as f:np.savez_compressed(f,**state(cand))
        ending=state(cand)
        assert all(np.isfinite(v).all() for v in ending.values())
        if hasattr(cand,'release'):
            assert np.all((ending['x']>=0)&(ending['x']<=1)) and np.all((ending['u']>=0)&(ending['u']<=1))
            assert np.array_equal(cand.release.w[:]/b.mV,cand.original_selected_weights)
        final_sha=__import__('hashlib').sha256(np.asarray(cand.brain.synapses.w[:]/b.mV).tobytes()).hexdigest()
        assert final_sha==original_sha and cand.brain.plasticity.updates==0
        manifest.update(status='complete',wall_seconds=time.monotonic()-start,
                        peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                        final_storage_weights_sha256=final_sha,finite_final_state=True)
        atomic_json(folder/'manifest.json',manifest)
        print(folder.name,'complete',round(manifest['wall_seconds'],1),'s',flush=True)
    except BaseException as e:
        manifest.update(status='interrupted',error=repr(e));atomic_json(folder/'manifest.json',manifest);raise


def run():
    with (ROOT/'.runtime/experiment.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);protocol=prepare();completed=[]
        for i,case in enumerate(CASES):
            for seed in SEEDS:
                folder=OUT/'trials'/f"{case['model']}-{case['cue']}-{seed}";path=folder/'manifest.json'
                if path.exists():
                    m=json.loads(path.read_text());assert m['status']=='complete' and m['sources']==protocol['sources']
                else:
                    space_check(OUT,1200*1024**2)
                    atomic_json(OUT/'progress.json',{'status':'running','completed':completed,'current':folder.name,'planned':12})
                    subprocess.run([sys.executable,__file__,'--case',str(i),'--seed',str(seed)],cwd=ROOT,check=True)
                if case['cue']=='a_left' and seed==SEEDS[0] and not (folder/'continuation-result.json').exists():
                    subprocess.run([sys.executable,__file__,'--case',str(i),'--seed',str(seed),'--resume'],cwd=ROOT,check=True)
                completed.append(folder.name)
        atomic_json(OUT/'progress.json',{'status':'complete','completed':completed,'planned':12})


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--prepare',action='store_true');p.add_argument('--case',type=int);p.add_argument('--seed',type=int);p.add_argument('--resume',action='store_true');a=p.parse_args()
    if a.prepare:prepare()
    elif a.case is not None:worker(a.case,a.seed,a.resume)
    else:run()
