"""Independent immutable copy of completed odor evidence and current source."""
import sys,json,subprocess,time,uuid,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from flygarden.continuous_candidate import file_sha
from flygarden.recording import atomic_json,space_check
OUT=ROOT/'reports/brain-integration/recovery/odor-bias-diagnosis'
source=ROOT/'reports/brain-integration/recovery/odor-causal-v2'
receipt=OUT/'preservation.json'
if receipt.exists():
 d=json.loads(receipt.read_text());assert all(file_sha(Path(d['snapshot'])/f)==h for f,h in d['files'].items());print('Existing preservation copy verified');sys.exit()
space_check(OUT,1024**3)
folder=OUT/('snapshot-'+uuid.uuid4().hex);folder.mkdir(exist_ok=False)
started=time.monotonic();target=folder/'odor-causal-v2'
subprocess.run(['cp','-cR',str(source),str(target)],check=True)
files={}
for p in source.rglob('*'):
 if p.is_file():
  rel=Path('odor-causal-v2')/p.relative_to(source);a=file_sha(p);assert a==file_sha(folder/rel);files[str(rel)]=a
for base in ('flygarden','scripts','static'):
 for p in (ROOT/base).rglob('*'):
  if not p.is_file() or '__pycache__' in p.parts:continue
  rel=p.relative_to(ROOT);dst=folder/'current-source'/rel;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dst);h=file_sha(p);assert h==file_sha(dst);files[str(Path('current-source')/rel)]=h
for name in ('requirements.lock','package.json','package-lock.json','reports/provenance.json','reports/neuron-mapping.json'):
 p=ROOT/name;dst=folder/'current-source'/name;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dst);h=file_sha(p);assert h==file_sha(dst);files['current-source/'+name]=h
atomic_json(receipt,{'status':'passed','snapshot':str(folder),'files':files,'file_count':len(files),'wall_seconds':time.monotonic()-started,'scope':'Verified CoW independent evidence copy; original trials, sources and checkpoints untouched'})
print('Evidence preserved and hash verified:',len(files),'files')
