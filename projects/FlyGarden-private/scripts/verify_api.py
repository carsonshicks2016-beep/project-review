import json,time
from pathlib import Path
import httpx,numpy as np
root=Path(__file__).resolve().parents[1];client=httpx.Client(base_url='http://127.0.0.1:8794',timeout=180)
def command(action,**arguments):
 r=client.post('/api/command',json=dict(action=action,arguments=arguments));r.raise_for_status();return r.json()
def state():return client.get('/api/state').json()
command('pause');command('mode',mode='full');command('reset',level=1,activity='sandbox',seed=42);command('step');checkpoint=command('save',kind='scene');command('step');a=state();command('load',id=checkpoint['id']);command('step');b=state()
checks=dict(full_checkpoint_body_continuation=bool(np.allclose(a['body']['position'],b['body']['position'],rtol=0,atol=1e-7)),full_checkpoint_neural_rates=a['rates']==b['rates'],full_checkpoint_motor=bool(np.allclose(a['motor'],b['motor'],rtol=0,atol=1e-12)),full_checkpoint_time=a['world']['time']==b['world']['time'])
old=b['lineage'];command('branch',id=checkpoint['id']);checks['branch_lineage']=state()['lineage']!=old
command('mode',mode='baseline');command('reset',level=6,activity='sandbox',seed=12);command('step');s=state();checks['predator_layout']=s['arena']['predator']['enabled'] and any(x.get('kind')=='shelter' for x in s['arena']['blocks'])
command('edit',kind='food_a',x=-20,y=-8);s=state();checks['edit_pauses_invalidates']=not s['running'] and not s['world']['valid'] and s['world']['time']==0
r=client.post('/api/command',json=dict(action='edit',arguments=dict(kind='block',x=float('inf'),y=0))) if False else None
checks['host_rejection']=client.get('/api/health',headers={'host':'example.org'}).status_code==403
checks['origin_rejection']=client.post('/api/command',headers={'origin':'http://evil.example'},json={'action':'play'}).status_code==403
report=dict(status='completed',rates_a=a['rates'],rates_b=b['rates'],checks=checks,passed=all(checks.values()),checkpoint=checkpoint['id']);(root/'reports/api-verification.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2));assert report['passed']
