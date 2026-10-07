"""Finish already authorized Stage3 checks sequentially after neural trials."""
import json,sys,time,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
root=Path(sys.argv[1]).resolve()
while True:
 p=json.loads((root/'progress.json').read_text())
 if p['status']=='complete':break
 if p['status']=='interrupted':raise RuntimeError(p.get('error','Neural worker interrupted'))
 time.sleep(10)
for script in ('replay_descending_body.py','report_sensorimotor_pathway.py'):
 log=ROOT/'.runtime'/f'stage3-{script}.log'
 with log.open('w') as stream:
  subprocess.run([sys.executable,str(ROOT/'scripts'/script),str(root)],cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,check=True)
 print(script,'complete',flush=True)
