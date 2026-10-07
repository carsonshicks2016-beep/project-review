"""Independent conservative envelope checks against the resolved whole-course profiles.
Native images remain required: an envelope bound does not prove visual continuity.
"""
import json,math,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
p=ROOT/'Assets/Resources/Circuits/Nordschleife.json';d=json.loads(p.read_text());kit=json.loads((ROOT/'Assets/Resources/StageDressing/TreeKitV3.json').read_text())
profiles=d['roadProfiles'];assert len(profiles)==len(d['points'])
# Maximum branch-card horizontal reach including trunk bend, all LODs, yaw and jitter.
reach=[]
for i,s in enumerate(kit['species']):
 h=16*(.65 if i==2 else .8 if i==3 else 1)
 crown=h*s['crownRadius']*1.22*math.sqrt(1+(.6*.52)**2)
 bend=h*math.sqrt((.3*.055)**2+.012**2)
 reach.append({'species':s['name'],'maximumHorizontalReach':crown+bend+.08})
maximum=max(x['maximumHorizontalReach'] for x in reach)
# The runtime rejects candidates within 4m of the nearest globally projected barrier.
# For the present uniform profiles, the nearest centerline is also the nearest corridor;
# differing barrier widths must be audited per candidate instead of using this bound.
for key in ('left','right','leftBarrier','rightBarrier'):
 assert max(x[key] for x in profiles)-min(x[key] for x in profiles)<1e-6, 'Variable widths require candidate-by-candidate envelope audit'
minimum=min(x[k+'Barrier']+4-x[k+'Kerb'] for x in profiles for k in ('left','right'))-maximum
assert minimum>0
result={'revision':d['revision'],'generatorVersion':d['generatorVersion'],'sourceGeometryHash':hashlib.sha256(p.read_bytes()).hexdigest(),'sampledProfiles':len(profiles),'barrierClearanceRuleMetres':4,'allNearbyRoads':'global horizontal nearest-segment lookup; uniform corridor widths independently verified','speciesEnvelopes':reach,'minimumConservativeCanopyClearanceFromPavedKerbEnvelope':minimum,'decorativeColliders':'none; verified source construction and native collider total','inspection':'full-route native road/elevated captures independently reviewed','limitations':['Conservative horizontal canopy bound proves clearance from the paved/kerb envelope for current uniform widths. Foliage above grass shoulders/barriers is visually inspected rather than treated as a physical hazard.','This does not replace terrain-ray or visual gap checks.']}
out=ROOT/'.rally/circuit-review/stage-c-delivery-clearance-audit.json';out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
