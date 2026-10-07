"""Attach completed trial outcomes once; never rewrite an earlier outcome."""
import json,hashlib
from pathlib import Path
root=Path(__file__).resolve().parents[1]
report=json.loads((root/'reports/baseline-acceptance.json').read_text())
for row in report['runs']:
 folder=root/'data/runs'/row['recording'];manifest=json.loads((folder/'manifest.json').read_text())
 if manifest['seed']!=row['seed'] or manifest['controller']['sha256']!=report['controller_sha256']:raise ValueError('Trial/controller identity mismatch')
 payload={**row,'controller_sha256':report['controller_sha256'],'recording_sha256':hashlib.sha256((folder/'frames.jsonl').read_bytes()).hexdigest(),'scope':'Completed physical trial of supplied baseline only'}
 path=folder/'outcome.json'
 if path.exists():
  if json.loads(path.read_text())!=payload:raise ValueError('Existing immutable outcome differs')
 else:
  with path.open('x') as f:json.dump(payload,f,indent=2)
print('Completed trial outcomes attached:',len(report['runs']))
