"""Immutable local checkpoints; never deserialize user-supplied files."""
from pathlib import Path
import json,pickle,uuid,hashlib,time,os
ROOT=Path(__file__).resolve().parents[1]
class Store:
    def __init__(self,root=None):self.root=Path(root or ROOT/'data/checkpoints');self.root.mkdir(parents=True,exist_ok=True)
    def create(self,kind,payload,brain=None,parent=None):
        identity=uuid.uuid4().hex;folder=self.root/identity;folder.mkdir()
        try:
            raw=pickle.dumps(payload,protocol=5);(folder/'state.bin').write_bytes(raw)
            if brain is not None:brain.save(folder/'brain.bin')
            provenance=json.loads((ROOT/'reports/provenance.json').read_text()) if (ROOT/'reports/provenance.json').exists() else {}
            metadata=dict(schema_version=payload.get('version',1),provenance=provenance,id=identity,kind=kind,parent=parent,created=time.time(),sha256=hashlib.sha256(raw).hexdigest(),brain=brain is not None,files={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in folder.iterdir()})
            (folder/'metadata.json').write_text(json.dumps(metadata,indent=2))
            return metadata
        except Exception:
            import shutil;shutil.rmtree(folder);raise
    def load(self,identity):
        if len(identity)!=32 or any(c not in '0123456789abcdef' for c in identity):raise ValueError('Invalid checkpoint identifier')
        folder=self.root/identity;metadata=json.loads((folder/'metadata.json').read_text())
        for name,expected in metadata['files'].items():
            if hashlib.sha256((folder/name).read_bytes()).hexdigest()!=expected:raise ValueError('Checkpoint integrity check failed')
        return pickle.loads((folder/'state.bin').read_bytes()),metadata,folder
    def list(self):
        records=[]
        for p in self.root.glob('*/metadata.json'):
            try:
                r=json.loads(p.read_text())
                if r['id']!=p.parent.name or r['kind'] not in ('scene','learned') or not isinstance(r['files'],dict):continue
                r['created']=float(r['created']);records.append(r)
            except (ValueError,KeyError,TypeError,OSError):continue
        return sorted(records,key=lambda r:r['created'],reverse=True)
