"""Read-only integrity audit, independent of screening evaluator."""
import hashlib,json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'reports/brain-integration/recovery/adaptive-domain-screen-v1'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
results=[]
for folder in sorted((OUT/'trials').iterdir()):
 m=json.loads((folder/'manifest.json').read_text())
 if m['status']!='complete':continue
 assert len(m['chunks'])==240
 assert m['initial_weight_hash']==m['final_weight_hash']
 for name,digest in m['sources'].items():assert sha(ROOT/name)==digest
 spikes=0
 for tick,ch in enumerate(m['chunks']):
  assert abs(ch['end']-(tick+1)*.025)<1e-10
  f=folder/ch['file'];assert sha(f)==ch['sha256']
  with np.load(f) as z:
   assert np.array_equal(np.bincount(z['spike_i'],minlength=138639),z['counts'])
   assert all(np.isfinite(z[k]).all() for k in z.files)
   assert np.all(z['spike_t']>=tick*.025-1e-10)
   assert np.all(z['spike_t']<(tick+1)*.025+1e-10)
   spikes+=len(z['spike_i'])
 continuation=folder/'continuation-result.json'
 if m['case']=='a_left':assert continuation.exists() and json.loads(continuation.read_text())['status']=='passed'
 results.append({'trial':folder.name,'chunks':240,'spikes':spikes,'wall_seconds':m['wall_seconds'],'peak_rss_bytes':m['peak_rss_bytes'],'manifest_sha256':sha(folder/'manifest.json')})
result={'status':'passed','completed_trials':len(results),'chunks':len(results)*240,'trials':results,'scope':'Integrity, event/count agreement, simulation timestamps, frozen sources, fixed weights, continuation receipts. Does not establish biological validity or behavioral competence.'}
p=OUT/'independent-recording-audit.json';tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(result,indent=2));tmp.replace(p)
print(json.dumps({'status':'passed','trials':len(results),'chunks':len(results)*240}))
