"""Ten held-out sixty-second rate-to-body tests with predefined quality gates."""
import sys,json,subprocess,time,fcntl,pickle,argparse
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from flygarden.body_trial import run_trial,read_trial,sha
from flygarden.recording import atomic_json
from flygarden.power import on_ac_power
OUT=ROOT/'reports/brain-integration/recovery/body-acceptance'
SOURCES=('scripts/accept_recovery_body.py','flygarden/body_trial.py','flygarden/physical_contacts.py',
         'flygarden/synchronized.py','flygarden/body.py','flygarden/descending.py','flygarden/power.py')
SEEDS=tuple(range(9201,9211));DT=.025

def command(t):
    phase=min(int(t//10),5)
    forward=65 if phase in (1,2,4,5) else 0
    rates={'DNp09_left':forward,'DNp09_right':forward,
           'DNa02_left':50 if phase==2 else 0,'DNa02_right':50 if phase==4 else 0}
    return 'rates',rates,('settle_stop','forward','left','stop_recovery','right','forward_recovery')[phase]

def sources():return {name:sha(ROOT/name) for name in SOURCES}

def equal(a,b):
    if isinstance(a,np.ndarray):return isinstance(b,np.ndarray) and np.array_equal(a,b)
    if isinstance(a,dict):return a.keys()==b.keys() and all(equal(v,b[k]) for k,v in a.items())
    if isinstance(a,(list,tuple)):return len(a)==len(b) and all(equal(x,y) for x,y in zip(a,b))
    return a==b

def verify_continuation():
    base=OUT/'continuation';base.mkdir(exist_ok=True)
    for name,stop in (('uninterrupted',None),('resumed',1.),('resumed',None)):
        args=[sys.executable,str(Path(__file__).resolve()),'--continuation-worker',name]
        if stop:args+=['--stop-after',str(stop)]
        subprocess.run(args,cwd=ROOT,check=True)
    a,ar,af=read_trial(base/'uninterrupted');b,br,bf=read_trial(base/'resumed')
    assert a['status']==b['status']=='complete' and ar==br and af==bf
    for x,y in zip(a['chunks'],b['chunks']):
        for folder,c in ((base/'uninterrupted',x),(base/'resumed',y)):
            assert sha(folder/c['checkpoint']['file'])==c['checkpoint']['sha256']
        with (base/'uninterrupted'/x['checkpoint']['file']).open('rb') as f:state_a=pickle.load(f)
        with (base/'resumed'/y['checkpoint']['file']).open('rb') as f:state_b=pickle.load(f)
        assert equal(state_a,state_b)
    atomic_json(base/'result.json',{'status':'complete','fresh_processes':3,'all_rows_frames_and_physics_checkpoints_exact':True,
                                  'seed':9299,'sources':sources(),'duration':3.,'interruption_time':1.})

def commit():
    OUT.mkdir(parents=True,exist_ok=True)
    calibration=ROOT/'reports/brain-integration/recovery/body-operating-range/results.json'
    result=json.loads(calibration.read_text());assert len(result['trials'])==22 and result['commands_reconstructed']
    protocol={'version':1,'sources':sources(),'calibration_result_sha256':sha(calibration),'seeds':list(SEEDS),
              'duration':60.,'interval':DT,'frozen_rates_hz':{'forward_per_DNp09':65,'unilateral_DNa02':50},
              'schedule':'10s each: stop,forward,left,stop,right,forward; continuous fixed decoder state',
              'gates':{'finite_complete':'All10 complete60s with finite physical observations',
                       'forward':'At least8/10 seeds: more than10mm travel and net displacement in each forward phase',
                       'turn':'At least8/10 seeds: left heading change>0.5rad and right heading change<-0.5rad',
                       'stopping':'At least8/10 seeds: mean xy speed during38-40s below0.1mm/s',
                       'no_flips':'At least9/10 seeds have no flipped samples',
                       'continuation':'Separate fresh-process interruption reproduces all recorded rows,poses and saved physical states exactly'},
              'secondary':['Contact-point tangential speeds under measured solver loads','Active stalls in1s blocks: motor RMS>0.1 and xy travel<0.1mm','Forward heading drift','Contact coverage'],
              'scope':'Artificial neuron-rate readouts through fixed decoder into supplied gait; no brain simulated, no navigation or learning claim. Engineering quality thresholds, not biological norms.'}
    p=OUT/'protocol.json'
    if p.exists():assert json.loads(p.read_text())==protocol,'Acceptance source/protocol changed'
    else:atomic_json(p,protocol)

def run():
    lock=ROOT/'.runtime/experiment.lock';lock.parent.mkdir(exist_ok=True)
    with lock.open('a') as guard:
        fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB);commit()
        if not on_ac_power():
            atomic_json(OUT/'progress.json',{'status':'paused_for_ac_power','completed':[]});return
        continuation=OUT/'continuation/result.json'
        if not continuation.exists():verify_continuation()
        else:assert json.loads(continuation.read_text())['sources']==sources()
        done=[];started=time.perf_counter()
        for seed in SEEDS:
            atomic_json(OUT/'progress.json',{'status':'running','completed':done,'current':seed,'planned':10})
            status=run_trial(OUT/'trials'/str(seed),seed,60.,command,sources())
            if status.startswith('paused'):
                atomic_json(OUT/'progress.json',{'status':status,'completed':done,'current':seed,'planned':10});return
            done.append(seed);print(f'{len(done)}/10 held-out60s trials',flush=True)
        atomic_json(OUT/'progress.json',{'status':'complete','completed':done,'wall_seconds':time.perf_counter()-started})

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--continuation-worker');parser.add_argument('--stop-after',type=float);args=parser.parse_args()
    if args.continuation_worker:
        if args.continuation_worker not in ('uninterrupted','resumed'):raise ValueError('Invalid diagnostic worker')
        # Include motion in the3s continuation fixture, rather than60s initial stopping.
        def fixture(t):return command(t+10.)
        run_trial(OUT/'continuation'/args.continuation_worker,9299,3.,fixture,sources(),stop_after=args.stop_after)
    else:run()
