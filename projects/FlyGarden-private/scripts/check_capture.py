import json,sys
from pathlib import Path
import numpy as np,httpx
root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root));from flygarden.storage import Store
c=httpx.Client(base_url='http://127.0.0.1:8794',timeout=180)
def cmd(action,**args):
 r=c.post('/api/command',json={'action':action,'arguments':args});r.raise_for_status();return r.json()
cmd('mode',mode='full');cmd('reset',seed=20,level=1,activity='sandbox');cmd('step');cmd('edit',kind='predator',x=-22,y=0);before=cmd('save',kind='learned');cmd('step');after=cmd('save',kind='learned');s=c.get('/api/state').json();store=Store();a=store.load(before['id'])[0]['learned'];b=store.load(after['id'])[0]['learned'];checks={'captured':s['world']['status']=='captured','capture_before_food':s['world']['collected']==0,'unvalidated_capture_conditioning_disabled':np.array_equal(a['weights'],b['weights'])};cmd('reset');checks['reset_retains_learning']=c.get('/api/state').json()['learning_updates']==b['updates'];checks={k:bool(v) for k,v in checks.items()};r={'status':'completed','passed':all(checks.values()),'checks':checks};(root/'reports/capture-verification.json').write_text(json.dumps(r,indent=2));print(r);assert r['passed']
