"""Compare complete Brian state and physical/gait/RNG state across restart."""
import sys,json,pickle,hashlib,subprocess,argparse,fcntl
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.calibrate_continuous_candidate import OUT as CAL,sources
from flygarden.body_trial import sha
from flygarden.recording import atomic_json
from flygarden.power import on_ac_power
OUT=CAL/'full-state-continuation'

def digest(value):
    h=hashlib.sha256()
    def visit(v):
        if isinstance(v,np.ndarray):
            h.update(b'array');visit(v.dtype.str);visit(v.shape);h.update(np.ascontiguousarray(v).tobytes())
        elif isinstance(v,dict):
            h.update(b'dict');visit(len(v))
            for key in sorted(v):visit(key);visit(v[key])
        elif isinstance(v,(list,tuple)):
            h.update(type(v).__name__.encode());visit(len(v))
            for item in v:visit(item)
        elif isinstance(v,np.generic):visit(v.dtype.str);visit(v.item())
        elif v is None or isinstance(v,(str,int,float,bool,bytes)):
            payload=repr(v).encode();h.update(type(v).__name__.encode());h.update(str(len(payload)).encode()+b':'+payload)
        else:raise TypeError('Unsupported state value: '+str(type(v)))
    visit(value);return h.hexdigest()

def worker(name):
    from flygarden.continuous_candidate import ContinuousCandidate
    from flygarden.candidate_inputs import CandidateInputs
    from flygarden.synchronized import SynchronizedBody,CausalCoupling
    from scripts.diagnose_recovery_interfaces import mappings
    checkpoint=OUT/'split/checkpoint-2.0' if name=='split_second' else CAL/'trials/9501-65hz/checkpoint-1.5'
    if name=='split_second':assert sha(checkpoint/'body-driver.pkl')==json.loads((checkpoint/'driver-integrity.json').read_text())['body-driver.pkl']
    else:assert sha(checkpoint/'body-driver.pkl')==json.loads((checkpoint/'driver-integrity.json').read_text())['body-driver.pkl']
    with (checkpoint/'body-driver.pkl').open('rb') as f:driver=pickle.load(f)
    _,mapping=mappings();candidate=ContinuousCandidate(mapping,9501,CandidateInputs(65.));candidate.load(checkpoint)
    body=SynchronizedBody(seed=9501);body.restore(driver['body']);coupling=CausalCoupling(.025);coupling.restore(driver['coupling'])
    folder=OUT/('uninterrupted' if name=='uninterrupted' else 'split');folder.mkdir(exist_ok=True)
    entries=json.loads((folder/'states.json').read_text()) if name=='split_second' else []
    finish=80 if name=='split_first' else 100
    try:
        for tick in range(round(coupling.time/.025),finish):
            coupling.advance(lambda dt:candidate.advance(dt,[[0.,0.],[0.,0.]],[0.,0.],False),lambda dt,m:body.advance(dt,m))
            # Brian2's pinned _full_state includes all neuron/synapse variables,
            # delayed spike queues,monitors,input generators and clock state.
            network=candidate.brain.network._full_state()
            entries.append({'time':coupling.time,'network_objects':{key:digest(value) for key,value in sorted(network.items())},
                            'body_complete':digest(body.snapshot()),'coupling':digest(coupling.snapshot()),
                            'adapter':digest({'previous':candidate.previous,'decoder':candidate.decoder.motor,
                                              'rng':candidate.brain.rng.bit_generator.state,'plasticity':candidate.brain.plasticity.snapshot()})})
            del network;atomic_json(folder/'states.json',entries)
        if name=='split_first':
            p=folder/'checkpoint-2.0';candidate.save(p)
            with (p/'body-driver.pkl').open('wb') as f:pickle.dump({'body':body.snapshot(),'coupling':coupling.snapshot()},f,protocol=5)
            atomic_json(p/'driver-integrity.json',{'body-driver.pkl':sha(p/'body-driver.pkl')})
    finally:body.close()

def run():
    with (ROOT/'.runtime/experiment.lock').open('a') as guard:
        fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if not on_ac_power():raise RuntimeError('Native state proof requires AC power')
        OUT.mkdir(exist_ok=False)
        atomic_json(OUT/'protocol.json',{'sources':sources(),'checker_sha256':sha(Path(__file__)),'seed':9501,
                                       'start_time':1.5,'interruption_time':2.,'end_time':2.5,
                                       'criterion':'Exact canonical SHA256 of every Brian full-state object and complete body/gait/RNG,adapter and coupling state at every25ms boundary.'})
        for name in ('uninterrupted','split_first','split_second'):
            subprocess.run([sys.executable,__file__,'--worker',name],cwd=ROOT,check=True)
        a=json.loads((OUT/'uninterrupted/states.json').read_text());c=json.loads((OUT/'split/states.json').read_text())
        assert len(a)==len(c)==40
        for x,y in zip(a,c):assert x==y,'Full state mismatch at'+str(x['time'])
        atomic_json(OUT/'results.json',{'status':'passed','fresh_processes':3,'boundaries':40,
                                      'all_brian_objects_including_delayed_queues_exact':True,
                                      'complete_physics_gait_rng_adapter_coupling_exact':True,
                                      'protocol_sha256':sha(OUT/'protocol.json')})
        print('Full internal-state continuation passed',flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--worker',choices=('uninterrupted','split_first','split_second'));args=parser.parse_args()
    if args.worker:worker(args.worker)
    else:run()
