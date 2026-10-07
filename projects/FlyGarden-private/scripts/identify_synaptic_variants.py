"""Keep altered model identity distinct from unchanged decoder identity."""
import sys,json,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from flygarden.recording import atomic_json
from flygarden.continuous_candidate import file_sha
OUT=ROOT/'reports/brain-integration/recovery/synaptic-scale-calibration-v1'
s=json.loads((OUT/'diagnostic-protocol.json').read_text());variants=[]
for scale in (.7,1.,1.3):
 files=list((OUT/'diagnostics').glob(f'*-scale{scale:g}-*/manifest.json'));assert len(files)==8
 ms=[json.loads(p.read_text()) for p in files];assert all(m['status']=='complete' for m in ms)
 weights={m['initial_effective_weight_sha256'] for m in ms};assert len(weights)==1;assert all(m['final_effective_weight_sha256'] in weights for m in ms)
 identity={'base':'Released v783 full network; original equations and fixed source-pinned supported interface','global_synaptic_scale':scale,'weight_per_signed_synapse_mV':.275*scale,'effective_weight_sha256':next(iter(weights)),'sources':s['sources'],'decoder':'continuous candidate bilateral DNa02 difference, unchanged','learning':False,'neurons':138639,'connection_records':15091983,'topology_and_signs_preserved':True,'original_parameter_equivalence':scale==1.}
 digest=hashlib.sha256(json.dumps(identity,sort_keys=True,separators=(',',':')).encode()).hexdigest();identity['model_variant_id']=f'full-v783-wscale-{scale:g}-{digest[:12]}'
 identity['run_manifests']={str(p.relative_to(ROOT)):file_sha(p) for p in files};variants.append(identity)
atomic_json(OUT/'model-variants.json',{'status':'identified','variants':variants,'scope':'Separate model parameter identities; decoder identity alone is insufficient to identify a run.'})
print('Three model parameter identities retained')
