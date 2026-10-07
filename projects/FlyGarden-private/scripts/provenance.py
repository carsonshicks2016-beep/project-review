import hashlib,json,subprocess,importlib.metadata as m,sys
from pathlib import Path
root=Path(__file__).resolve().parents[1]
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
 return h.hexdigest()
files=['vendor/fly-brain/data/2025_Completeness_783.csv','vendor/fly-brain/data/2025_Connectivity_783.parquet','data/annotations.tsv']
versions={p:m.version(p) for p in ('brian2','flygym','mujoco','numpy','pandas','fastapi','uvicorn','networkx')}
report=dict(source_revisions={p:subprocess.check_output(['git','-C',str(root/'vendor'/p),'rev-parse','HEAD'],text=True).strip() for p in ('fly-brain','flygym')},annotations_revision='a83b2776d60d5764cef36b927f5f9679c16c47a2',hashes={p:sha(root/p) for p in files},versions=versions,licenses={'fly-brain':'GPL-2.0-or-later upstream; original Brian2 materials MIT; retain notices','flygym':'Apache-2.0, pinned 2.1 source','annotations':'See upstream flywire_annotations README and source publication'},alterations=['Remove undefined w from released neuron reset','Engineered ORN, LPLC2 and DNp09 stimulation','Gamma KC to MBON01 and MBON11 engineered depression; remaining weights fixed','Supplied HybridTurningController leg coordination','No edge threshold or reduced network','Visual adapter currently geometric egocentric proxy, not retina fidelity'])
(root/'reports/provenance.json').write_text(json.dumps(report,indent=2))
(root/'requirements.lock').write_text(subprocess.check_output(['uv','pip','freeze','--python',str(root/'.venv-next/bin/python')],text=True))
print(json.dumps(versions))
