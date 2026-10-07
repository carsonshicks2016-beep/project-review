import json,subprocess,time
from pathlib import Path
import httpx,numpy as np
root=Path(__file__).resolve().parents[1]
c=httpx.Client(base_url='http://127.0.0.1:8794',timeout=180)
def cmd(action,**args):
 r=c.post('/api/command',json={'action':action,'arguments':args});r.raise_for_status();return r.json()
cmd('load',id=json.loads((root/'reports/api-verification.json').read_text())['checkpoint']);checkpoint=cmd('save',kind='scene');cmd('step');expected=c.get('/api/state').json()
subprocess.run([str(root/'Close Fly Garden.command')],check=True);subprocess.run([str(root/'Open Fly Garden.command'),'--no-browser'],check=True)
for _ in range(120):
 if c.get('/api/health').json()['ready']:break
 time.sleep(.5)
cmd('load',id=checkpoint['id']);cmd('step');actual=c.get('/api/state').json()
checks={'body':bool(np.allclose(expected['body']['position'],actual['body']['position'],rtol=0,atol=1e-7)),'rates':expected['rates']==actual['rates'],'motor':expected['motor']==actual['motor'],'clock':expected['world']['time']==actual['world']['time']}
r={'checks':checks,'passed':all(checks.values()),'checkpoint':checkpoint['id']};(root/'reports/cold-start.json').write_text(json.dumps(r,indent=2));print(r);assert r['passed']
