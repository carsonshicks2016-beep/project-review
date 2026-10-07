"""Run the bounded read-only audits once the verified archive is published."""
from pathlib import Path
import fcntl
import json
import os
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'reports/brain-integration/recovery/patchy-compartment-mapping-v2'

def main():
    with (OUT/'audit-worker.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        state=OUT/'audit-worker.json'
        def update(status,**kw):
            tmp=state.with_suffix('.tmp')
            tmp.write_text(json.dumps({'status':status,'pid':os.getpid(),'updated':time.time(),**kw},indent=2)+'\n')
            tmp.replace(state)
        update('waiting_for_verified_archive')
        while not (OUT/'archive-identity.json').exists():
            progress=json.loads((OUT/'acquisition-progress.json').read_text())
            if progress['status']=='interrupted_or_failed':
                update('acquisition_interrupted');return
            time.sleep(5)
        for script,log in [('audit_unthresholded_783.py','synapse.log'),('map_verified_783_sites.py','mapping.log')]:
            update('running',script=script)
            with (OUT/log).open('w') as output:
                result=subprocess.run([sys.executable,str(ROOT/'scripts'/script)],cwd=ROOT,
                    env={**os.environ,'PYTHONPATH':str(ROOT)},stdout=output,stderr=subprocess.STDOUT)
            if result.returncode:
                update('audit_failed',script=script,exit_code=result.returncode);return
        update('complete')

if __name__=='__main__':main()
