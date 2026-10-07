"""Continue Stage 2 with one fresh full-network worker per independent trial.

This avoids retaining a large initial-state cache during low-space memory
pressure. Continuous neural state is preserved throughout every trial.
"""
import argparse
import json
import os
import subprocess
import sys
import time
import uuid
import resource
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from flygarden.recording import atomic_json,space_check
from scripts.audit_brain_reference import sha
from scripts.diagnose_persistent_activity import SEEDS,conditions,select,IncomingObserver,run_trial


def worker(root,index,seed):
    import brian2 as b
    from flygarden.brain import FullBrain
    condition=conditions()[index]
    brain=FullBrain(seed=seed,learning=False)
    selected,pops,_=select(brain)
    observer=IncomingObserver(brain,selected)
    state=b.StateMonitor(brain.neurons,['v','g','not_refractory'],record=selected,dt=.001*b.second,when='end')
    brain.diagnostic_input=b.SpikeMonitor(brain.input,record=False)
    brain.network.add(*observer.objects,state,brain.diagnostic_input)
    brain.inputs.pre.code='v_post += diagnostic_jump';brain.inputs.namespace['diagnostic_jump']=68.75*b.mV
    # This worker has never advanced, so no stored state or duplicate immutable
    # connectivity arrays are necessary. Refuse reuse after any advancement.
    def fresh_reset(reset_seed):
        assert float(brain.network.t/b.second)==0
        assert np.all(np.asarray(brain.neurons.v[:]/b.mV)==-52)
        assert np.all(np.asarray(brain.neurons.g[:]/b.mV)==0)
        assert np.all(np.asarray(brain.monitor.count[:])==0)
        assert np.all(brain.motor==0) and np.all(brain.previous_counts==0)
        brain.rng=np.random.default_rng(reset_seed)
    brain.reset_transient=fresh_reset
    result=run_trial(root,brain,observer,state,selected,pops,condition,seed)
    result['reconstruction']='Fresh full network; continuous state within trial; no initial-state cache; fixed original weights.'
    result['worker_peak_rss_bytes']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    atomic_json(root/'trials'/result['folder']/'summary.json',result)


def launch(root,index,seed,log):
    log.parent.mkdir(parents=True,exist_ok=True)
    with log.open('w') as stream:
        subprocess.run([sys.executable,str(Path(__file__).resolve()),str(root),'--trial',str(index),'--seed',str(seed)],
                       cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,check=True)
    condition=conditions()[index]
    folder=f"{condition['profile']}-{condition['cue']}-{seed}"
    return json.loads((root/'trials'/folder/'summary.json').read_text())


def run(root):
    protocol=json.loads((root/'protocol.json').read_text())
    assert all(sha(ROOT/name)==digest for name,digest in protocol['sources'].items()),'Committed sources changed'
    progress=json.loads((root/'progress.json').read_text())
    if progress['status']=='complete':return
    import psutil
    for process in psutil.process_iter(['pid','cmdline']):
        if process.info['pid']==os.getpid():continue
        if any(Path(arg).name in ('diagnose_persistent_activity.py','resume_activity_diagnostics.py')
               for arg in (process.info['cmdline'] or [])):
            raise RuntimeError('Another diagnostic worker is still running')
    space_check(root,64*1024**2)
    previous_wall=progress.get('wall_seconds',0.);started=time.perf_counter()
    progress.setdefault('continuations',[]).append({'started':time.time(),'tool_sha256':sha(Path(__file__)),
        'completed_before':len(progress['trials']),'source':str(Path(__file__).relative_to(ROOT)),
        'strategy':'One fresh full-network worker per unfinished independent trial; no initial-state cache; continuous state within trials.'})
    progress['status']='verifying_continuation';atomic_json(root/'progress.json',progress)
    complete={(r['profile'],r['cue'],r['seed']) for r in progress['trials']}
    for row in progress['trials']:
        assert json.loads((root/'trials'/row['folder']/'manifest.json').read_text())['status']=='complete'
    try:
        overlap=next(r for r in progress['trials'] if r['profile']=='current' and r['cue']=='odor_a')
        index=next(i for i,c in enumerate(conditions()) if c['profile']==overlap['profile'] and c['cue']==overlap['cue'])
        check_root=root/'continuation-checks'/uuid.uuid4().hex
        rerun=launch(check_root,index,overlap['seed'],root/'worker-logs'/f'continuation-{len(progress["continuations"])}-check.log')
        original=root/'trials'/overlap['folder'];repeated=check_root/'trials'/rerun['folder']
        for file in original.glob('window-*.npz'):
            with np.load(file) as a,np.load(repeated/file.name) as repeat:
                for field in ('i','t','counts','input_counts'):
                    assert np.array_equal(a[field],repeat[field]),f'Continuation differs: {file.name}/{field}'
                for field in ('v_mV','g_mV','delivered_mV'):
                    assert np.allclose(a[field],repeat[field],rtol=0,atol=1e-8),f'Continuation state differs: {file.name}/{field}'
        atomic_json(check_root/'equivalence.json',{'passed':True,'original':str(original.relative_to(root)),
            'scope':'Exact full-network spikes/counts/input counts plus matching sampled neural state and delivery counters for a repeated completed odor trial.'})
        print(f'Fresh-worker equivalence passed; retaining {len(complete)} completed trials',flush=True)
        progress['status']='running'
        for index,condition in enumerate(conditions()):
            for seed in SEEDS:
                if (condition['profile'],condition['cue'],seed) in complete:continue
                progress['current']={**condition,'seed':seed};atomic_json(root/'progress.json',progress)
                folder=root/'trials'/f"{condition['profile']}-{condition['cue']}-{seed}"
                if folder.exists():
                    interrupted=root/'interrupted-attempts'/f'{folder.name}-{uuid.uuid4().hex}'
                    interrupted.parent.mkdir(exist_ok=True)
                    manifest=json.loads((folder/'manifest.json').read_text())
                    manifest['status']='interrupted';atomic_json(folder/'manifest.json',manifest)
                    folder.rename(interrupted)
                    progress.setdefault('preserved_incomplete',[]).append(str(interrupted.relative_to(root)))
                trial=launch(root,index,seed,root/'worker-logs'/f'{index:02d}-{seed}.log')
                progress['trials'].append(trial)
                progress['wall_seconds']=previous_wall+time.perf_counter()-started
                progress['peak_rss_bytes']=max(progress.get('peak_rss_bytes',0),trial['worker_peak_rss_bytes'])
                atomic_json(root/'progress.json',progress)
                print(f"{len(progress['trials'])}/{progress['planned_trials']} {condition['profile']} {condition['cue']} seed {seed}: tail {trial['tail_spikes_per_second']:.0f} spikes/s",flush=True)
        unchanged={name:sha(ROOT/name)==digest for name,digest in protocol['sources'].items()}
        assert all(unchanged.values())
        progress.update(status='complete',sources_unchanged=unchanged)
        atomic_json(root/'progress.json',progress)
        print('Diagnostic batch complete',flush=True)
    except Exception as error:
        progress.update(status='paused_or_failed',error=str(error))
        atomic_json(root/'progress.json',progress)
        raise


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('folder',type=Path)
    parser.add_argument('--trial',type=int);parser.add_argument('--seed',type=int)
    args=parser.parse_args()
    if args.trial is not None:worker(args.folder.resolve(),args.trial,args.seed)
    else:run(args.folder.resolve())
