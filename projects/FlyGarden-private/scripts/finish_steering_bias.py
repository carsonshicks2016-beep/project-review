"""Sequential completion of the authorized Step4 diagnostic."""
import sys,json,time,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];root=Path(sys.argv[1]).resolve()
while True:
 p=json.loads((root/'progress.json').read_text())
 if p['status']=='complete':break
 if p['status']=='interrupted':raise RuntimeError(p.get('error','Primary worker interrupted'))
 time.sleep(10)
for script in ('report_steering_bias.py','check_active_steering_inhibitor.py','report_steering_bias.py'):
 with (ROOT/'.runtime'/f'stage4-{script}.log').open('w') as output:subprocess.run([sys.executable,str(ROOT/'scripts'/script),str(root)],cwd=ROOT,stdout=output,stderr=subprocess.STDOUT,check=True)
 print(script,'complete',flush=True)
