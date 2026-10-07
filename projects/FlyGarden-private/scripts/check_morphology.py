import sys,json,time,hashlib
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from flygarden.morphology import catalog,skeleton,coverage,SOURCE
from flygarden.storage import ROOT
import numpy as np
rows=catalog();sample=[]
for cell in ('ORN_DM1','ORN_DM2','DNp09','MBON01','MBON11','LPLC2','APL'):
    sample.extend([r for r in rows if r['cell_type']==cell][:2])
sample.extend([r for r in rows if r['cell_class']=='Kenyon_Cell'][:2])
t=time.perf_counter();out=[]
for row in sample:
    try:
        s=skeleton(row['id']);v=np.asarray(s.get('vertices',[])).reshape(-1,3)
        out.append({**row,**{k:s[k] for k in ('status','bytes','sha256','vertices','edges') if k in s and k not in ('vertices','edges')},'nodes':len(v),'bounds_um':[v.min(axis=0).tolist(),v.max(axis=0).tolist()] if len(v) else None})
        print(row['cell_type'],row['id'],s['status'],flush=True)
    except Exception as exc:out.append({**row,'status':'failed','error':str(exc)})
bytes_=sum(s.get('bytes',0) for s in out);success=sum(s['status']=='available' for s in out)
r=dict(status='completed',source=SOURCE,dataset=783,samples=out,wall_seconds=time.perf_counter()-t,coverage=coverage(),estimated_raw_full_cache_bytes=round(bytes_/max(1,success)*len(rows)),estimate_scope='Small targeted anatomical sample, not a representative full-network size estimate; no bulk download',alignment='Native FlyWire nanometer coordinates converted to micrometers uniformly; no mixed-template registration',license='Standalone skeleton redistribution license not established; local cache only',documentation='https://fafbseg-py.readthedocs.io/en/latest/source/generated/fafbseg.flywire.get_skeletons.html')
(ROOT/'reports/recording-delivery/morphology-feasibility.json').write_text(json.dumps(r,indent=2))
