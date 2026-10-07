"""Map exact modeled sites using their distinct pre/post locations.

Only the already validated native-side atlas is accepted. A failed mirrored
atlas is never used as a controller or anatomical assignment fallback.
"""
from pathlib import Path
from collections import Counter
import hashlib
import json
import time
import warnings
import numpy as np
import pandas as pd
import rdata
import trimesh
from flygarden.compartment_mapping import unique_mesh_assignment
from flygarden.synapse_archive import directional_endpoints

ROOT=Path(__file__).resolve().parents[1]
OLD=ROOT/'reports/brain-integration/recovery/patchy-compartment-mapping-v1'
OUT=ROOT/'reports/brain-integration/recovery/patchy-compartment-mapping-v2'

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    start=time.monotonic()
    result=json.loads((OUT/'synapse-results.json').read_text())
    if not result['every_imported_pair_exact']:
        raise ValueError('Exact graph/site reconciliation required before mapping')
    native=json.loads((OLD/'spatial-results.json').read_text())
    if not native['alignment_gate_passed']:
        raise ValueError('Native atlas validation required')
    inputs=[OUT/'selected-synapses.parquet',OUT/'pair-count-audit.csv',OLD/'source/flywire_al.surf.rda',
            OLD/'spatial-results.json',OUT/'mirror-results.json',ROOT/'flygarden/compartment_mapping.py',
            ROOT/'flygarden/synapse_archive.py',Path(__file__).resolve()]
    protocol={'version':1,'source_hashes':{str(p.relative_to(ROOT)):sha(p) for p in inputs},
              'accepted_transform':'Identity only, validated native-side atlas',
              'mirror':'Excluded because pooled opposite-side registration gate failed',
              'coordinate_selection':'input uses post_pt_position; output uses pre_pt_position; nanometers',
              'site_filter':'Keep all sites on exact-reconciled imported pairs; preserve other source rows separately',
              'assignment':'Unique closed-mesh enclosure is an anatomical candidate only',
              'electrical_parameters_assigned':False,'no_glomerular_aliases':True,'no_controller_changes':True}
    (OUT/'mapping-protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    sites=pd.read_parquet(inputs[0])
    pairs=pd.read_csv(inputs[1],dtype={'pre_root_id':str,'post_root_id':str})
    allowed={(int(r.pre_root_id),int(r.post_root_id)) for r in pairs[pairs.imported_pair].itertuples()}
    keep=np.fromiter(((int(a),int(b)) in allowed for a,b in zip(sites.pre_pt_root_id,sites.post_pt_root_id)),bool,len(sites))
    modeled=sites[keep].copy()
    roots=json.loads((OLD/'neurons.json').read_text());lookup={int(r['root_id']):r for r in roots}
    endpoints=directional_endpoints(modeled,lookup)
    warnings.filterwarnings('ignore',message='Missing constructor for R class')
    data=rdata.read_rda(OLD/'source/flywire_al.surf.rda')['flywire_al.surf']
    v=data['Vertices'][['X','Y','Z']].to_numpy();meshes={}
    for name,frame in data['Regions'].items():
        faces=frame.to_numpy(dtype=np.int64)-1;used=np.unique(faces)
        remap=np.full(len(v),-1);remap[used]=np.arange(len(used))
        mesh=trimesh.Trimesh(v[used],remap[faces],process=True)
        if not mesh.is_watertight or not mesh.is_winding_consistent:
            raise ValueError('Invalid native mesh')
        meshes[str(name)]=mesh
    # Classify unique physical points once. Distinct pre/post endpoints remain distinct.
    points,inverse=np.unique(endpoints[['x','y','z']].to_numpy(),axis=0,return_inverse=True)
    membership=np.zeros((len(points),len(meshes)),bool)
    for j,(name,mesh) in enumerate(meshes.items()):
        idx=np.flatnonzero(np.all((points>=mesh.bounds[0])&(points<=mesh.bounds[1]),axis=1))
        for first in range(0,len(idx),500):
            part=idx[first:first+500];membership[part,j]=mesh.contains(points[part])
        (OUT/'mapping-progress.json').write_text(json.dumps({'region':name,'regions_done':j+1,'regions':len(meshes),'unique_endpoint_coordinates':len(points)})+'\n')
    labels=unique_mesh_assignment(membership,list(meshes))
    endpoints['mesh_candidate']=labels[inverse]
    endpoints['mesh_memberships']=membership.sum(axis=1)[inverse]
    endpoints['anatomical_candidate']=endpoints.mesh_memberships==1
    endpoints.to_parquet(OUT/'mapped-endpoints.parquet',index=False,compression='zstd')
    summary=[]
    for (rid,direction),a in endpoints.groupby(['root_id','direction']):
        annotation=lookup[int(rid)]
        summary.append({'root_id':rid,'cell_type':annotation['cell_type'],'side':annotation['side'],
                        'direction':direction,'imported_endpoint_sites':len(a),
                        'unique_atlas_candidates':int(a.anatomical_candidate.sum()),
                        'atlas_labels':dict(Counter(a.mesh_candidate)),
                        'source_neuropils':dict(Counter(a.source_neuropil)),
                        'physiological_compartment_assignment':False})
    (OUT/'endpoint-root-summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    final={'status':'mapping_complete_native_atlas_only','modeled_synapse_sites':len(modeled),
           'directional_endpoints':len(endpoints),'unique_endpoint_coordinates':len(points),
           'unique_enclosed_endpoints':int(endpoints.anatomical_candidate.sum()),
           'outside_endpoints':int((endpoints.mesh_memberships==0).sum()),
           'overlap_endpoints':int((endpoints.mesh_memberships>1).sum()),
           'unmodeled_partner_sites_preserved':int((~keep).sum()),
           'opposite_hemisphere_atlas_accepted':False,'physiological_compartments_created':0,
           'application_changed':False,'wall_seconds':time.monotonic()-start,
           'sources_reverified':all(sha(ROOT/p)==h for p,h in protocol['source_hashes'].items())}
    (OUT/'mapping-results.json').write_text(json.dumps(final,indent=2)+'\n')
    print(json.dumps(final,indent=2))

if __name__=='__main__':main()
