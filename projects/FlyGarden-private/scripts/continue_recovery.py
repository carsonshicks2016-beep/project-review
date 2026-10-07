"""Ownership-aware handoff: completed physical acceptance -> neural diagnostics.

Never restarts a missing body worker or bypasses a failed acceptance gate.
"""
import argparse,json,subprocess,sys,time
from pathlib import Path
import psutil
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from flygarden.recording import atomic_json
OUT=ROOT/'reports/brain-integration/recovery'

def run(pid,created):
    state=OUT/'handoff.json'
    try:
        process=psutil.Process(pid)
        if abs(process.create_time()-created)>.01:raise RuntimeError('Body worker PID was reused')
        if 'scripts/accept_recovery_body.py' not in ' '.join(process.cmdline()):raise RuntimeError('Unexpected worker command')
    except psutil.NoSuchProcess:
        process=None
    atomic_json(state,{'status':'waiting_for_body','pid':pid,'process_created':created})
    if process is not None:
        while True:
            try:process.wait(timeout=30);break
            except psutil.TimeoutExpired:continue
    progress=json.loads((OUT/'body-acceptance/progress.json').read_text())
    if progress['status']!='complete':raise RuntimeError('Body worker ended without complete acceptance trials; preserve evidence and diagnose')
    atomic_json(state,{'status':'auditing_body'})
    subprocess.run([sys.executable,str(ROOT/'scripts/report_recovery_body.py')],cwd=ROOT,check=True)
    results=json.loads((OUT/'body-acceptance/results.json').read_text())
    if results['status']!='passed':raise RuntimeError('Physical acceptance failed; neural promotion remains gated')
    subprocess.run([sys.executable,str(ROOT/'scripts/plot_recovery_body.py')],cwd=ROOT,check=True)
    p=OUT/'progress.json';summary=json.loads(p.read_text())
    summary['phases']['2'].update(status='acceptance_passed',remaining=[],
                                evidence=summary['phases']['2']['evidence']+['reports/brain-integration/recovery/body-acceptance/results.json'])
    summary['phases']['3']['status']='matrix_running'
    summary['next_action']='Observe existing full-brain interface matrix; audit after terminal success. Full12-phase objective remains active.'
    atomic_json(p,summary);atomic_json(state,{'status':'running_neural_diagnostics'})
    subprocess.run([sys.executable,str(ROOT/'scripts/diagnose_recovery_interfaces.py')],cwd=ROOT,check=True)
    subprocess.run([sys.executable,str(ROOT/'scripts/report_recovery_interfaces.py')],cwd=ROOT,check=True)
    atomic_json(state,{'status':'diagnostics_recorded_and_audited','interpretation_pending':True})

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--body-pid',type=int,required=True);parser.add_argument('--created',type=float,required=True);args=parser.parse_args()
    try:run(args.body_pid,args.created)
    except Exception as exc:
        atomic_json(OUT/'handoff.json',{'status':'failed','error':str(exc),'time':time.time()});raise
