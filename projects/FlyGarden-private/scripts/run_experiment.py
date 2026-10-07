"""Archive each experiment's identity, provenance and outcomes under a unique ID."""
import argparse,json,os,subprocess,sys,shutil,time,hashlib,fcntl
from pathlib import Path
root=Path(__file__).resolve().parents[1];parser=argparse.ArgumentParser();parser.add_argument('--protocol',choices=['learning','body','combined','baseline','recording30'],required=True);parser.add_argument('--id',required=True);args=parser.parse_args()
if len(args.id)!=32 or any(x not in '0123456789abcdef' for x in args.id):raise ValueError('Invalid experiment ID')
lock_folder=root/'.runtime';lock_folder.mkdir(exist_ok=True);lock=open(lock_folder/'experiment.lock','a')
try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
except BlockingIOError:print('Another owned experiment holds the simulation lock',flush=True);sys.exit(1)
folder=root/'data/experiments'/args.id;folder.mkdir(parents=True,exist_ok=False)
name,artifact={'learning':('evaluate_learning.py','learning-evaluation'),'body':('accept_body.py','body-acceptance'),'combined':('benchmark_combined.py','combined-benchmark'),'baseline':('accept_baseline.py','baseline-acceptance'),'recording30':('record_obstacle_trial.py','recording-delivery/obstacle-trial')}[args.protocol]
manifest={'id':args.id,'protocol':args.protocol,'started':time.time(),'provenance':json.loads((root/'reports/provenance.json').read_text()),'script_sha256':hashlib.sha256((root/'scripts'/name).read_bytes()).hexdigest()};(folder/'manifest.json').write_text(json.dumps(manifest,indent=2));shutil.copy2(root/'scripts'/name,folder/'runner-source.py');shutil.copy2(root/'requirements.lock',folder/'requirements.lock')
result=subprocess.run([sys.executable,str(root/'scripts'/name)],cwd=root,env={**os.environ,'FLYGARDEN_EXPERIMENT_DIR':str(folder)})
if args.protocol=='baseline' and result.returncode==0:
 outcome_result=subprocess.run([sys.executable,str(root/'scripts/attach_baseline_outcomes.py')],cwd=root)
 if outcome_result.returncode:result=outcome_result
if (root/'reports'/f'{artifact}.json').exists() and (root/'reports'/f'{artifact}.json').stat().st_mtime>=manifest['started']:shutil.copy2(root/'reports'/f'{artifact}.json',folder/'result.json')
if args.protocol=='combined':
 for f in ('combined-trace.json','body-geometry.json','full-brain-walking.mp4'):shutil.copy2(root/'reports'/f,folder/f)
(folder/'completion.json').write_text(json.dumps({'finished':time.time(),'exit_code':result.returncode,'status':'completed' if result.returncode==0 else 'failed'},indent=2));sys.exit(result.returncode)
