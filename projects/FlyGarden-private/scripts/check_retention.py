import sys,json,pickle
from pathlib import Path
import httpx,numpy as np
root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root));from flygarden.storage import Store
c=httpx.Client(base_url='http://127.0.0.1:8794',timeout=180)
def cmd(action,**args):
 r=c.post('/api/command',json={'action':action,'arguments':args});r.raise_for_status();return r.json()
cmd('mode',mode='full');cmd('reset',level=1,activity='sandbox',seed=7);before=cmd('save',kind='learned');cmd('edit',kind='spawn',x=0,y=0);cmd('step');saved=cmd('save',kind='learned');s=c.get('/api/state').json();cmd('reset',seed=8);cmd('load',id=saved['id']);after=cmd('save',kind='learned');store=Store();a=store.load(before['id'])[0]['learned'];b=store.load(saved['id'])[0]['learned'];d=store.load(after['id'])[0]['learned'];checks={'actual_food_contact':s['world']['collected']==1,'weights_changed':not np.array_equal(a['weights'],b['weights']),'weights_retained_exactly':np.array_equal(b['weights'],d['weights']),'transient_reset':c.get('/api/state').json()['world']['time']==0};r={'status':'completed','passed':all(checks.values()),'checks':checks,'saved':saved['id'],'scope':'Engineered plastic weights retained; not evidence of learned behavioral preference'};(root/'reports/retention.json').write_text(json.dumps(r,indent=2));print(r);assert r['passed']
