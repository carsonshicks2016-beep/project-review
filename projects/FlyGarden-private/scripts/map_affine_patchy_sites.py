"""Count-preserving partial anatomical mapping with explicit region eligibility."""
from pathlib import Path
from collections import Counter
import json
import time
import warnings
import numpy as np
import pandas as pd
import rdata
import trimesh
from flygarden.compartment_mapping import unique_mesh_assignment
from scripts.validate_local_glomerular_registration import sha,ROOT,V1,V2
from scripts.prepare_glomerular_affine_data import OUT

def main():
    start=time.monotonic();results=json.loads((OUT/'results.json').read_text())
    if not results['pooled_gate_passed'] or not results['DM1_gate_passed']:raise ValueError('Fresh pooled and DM1 gates required')
    source=[V2/'mapped-endpoints.parquet',V1/'source/flywire_al.surf.rda',V2/'mirrored-atlas-vertices.npz',
            OUT/'affine-atlas.npz',OUT/'per-glomerulus-results.json',ROOT/'flygarden/compartment_mapping.py',Path(__file__).resolve()]
    regions=json.loads((OUT/'per-glomerulus-results.json').read_text())
    eligible={r['glomerulus'] for r in regions if r['candidate']=='affine_candidate' and r['confidence_gate_passed']}
    protocol={'version':1,'source_hashes':{str(p.relative_to(ROOT)):sha(p) for p in source},
              'left':'Previously validated native atlas; original grouping candidates only',
              'right_eligible':sorted(eligible),'right_unvalidated_regions':'Remain in geometry membership tests but are not eligible grouping candidates',
              'ambiguity':'Require exactly one membership among all native and fitted meshes, including unvalidated neighbors',
              'input_coordinates':'Source post positions','output_coordinates':'Source pre positions','no_count_or_graph_change':True,
              'physiological_compartments_assigned':False}
    (OUT/'mapping-protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    warnings.filterwarnings('ignore',message='Missing constructor for R class')
    data=rdata.read_rda(V1/'source/flywire_al.surf.rda')['flywire_al.surf']
    native=data['Vertices'][['X','Y','Z']].to_numpy();mirrored=np.load(V2/'mirrored-atlas-vertices.npz')['vertices']
    affine=np.load(OUT/'affine-atlas.npz');meshes={}
    for name,df in data['Regions'].items():
        faces=df.to_numpy(np.int64)-1;used=np.unique(faces);remap=np.full(len(native),-1);remap[used]=np.arange(len(used))
        meshes['left:'+str(name)]=trimesh.Trimesh(native[used],remap[faces],process=True)
        if str(name) in affine:
            right=trimesh.Trimesh(mirrored[used],remap[faces][:,::-1],process=True)
            assert right.vertices.shape==affine[str(name)].shape
            right.vertices=affine[str(name)];meshes['right:'+str(name)]=right
    endpoints=pd.read_parquet(V2/'mapped-endpoints.parquet')
    points,inverse=np.unique(endpoints[['x','y','z']].to_numpy(),axis=0,return_inverse=True)
    memberships=np.zeros((len(points),len(meshes)),bool)
    for j,(name,mesh) in enumerate(meshes.items()):
        if not mesh.is_watertight or not mesh.is_winding_consistent:raise ValueError('Invalid mesh')
        ids=np.flatnonzero(np.all((points>=mesh.bounds[0])&(points<=mesh.bounds[1]),axis=1))
        for i in range(0,len(ids),500):
            sub=ids[i:i+500];memberships[sub,j]=mesh.contains(points[sub])
        (OUT/'mapping-progress.json').write_text(json.dumps({'regions_done':j+1,'regions':len(meshes),'region':name})+'\n')
    labels=unique_mesh_assignment(memberships,list(meshes))[inverse];counts=memberships.sum(axis=1)[inverse]
    accepted=np.array([str(label).startswith('left:') or str(label) in {'right:'+g for g in eligible} for label in labels])&(counts==1)
    endpoints['bilateral_mesh_candidate']=labels;endpoints['bilateral_memberships']=counts
    endpoints['region_evidence_eligible']=accepted
    endpoints['region_status']=np.where(accepted,'eligible_anatomical_candidate',
        np.where(counts==0,'unavailable_outside',np.where(counts>1,'unavailable_overlap','unavailable_region_evidence')))
    endpoints.to_parquet(OUT/'partial-bilateral-endpoints.parquet',index=False,compression='zstd')
    summaries=[]
    for (rid,direction),a in endpoints.groupby(['root_id','direction']):
        summaries.append({'root_id':rid,'direction':direction,'original_sites':len(a),'eligible_anatomical_sites':int(a.region_evidence_eligible.sum()),
                          'eligible_region_counts':dict(Counter(a.loc[a.region_evidence_eligible,'bilateral_mesh_candidate'])),
                          'unavailable_status_counts':dict(Counter(a.loc[~a.region_evidence_eligible,'region_status']))})
    previous=json.loads((V2/'endpoint-root-summary.json').read_text());expected={(r['root_id'],r['direction']):r['imported_endpoint_sites'] for r in previous}
    assert all(expected[(r['root_id'],r['direction'])]==r['original_sites'] for r in summaries)
    assert len(endpoints)==179560
    (OUT/'partial-root-summary.json').write_text(json.dumps(summaries,indent=2)+'\n')
    dm1=endpoints[endpoints.bilateral_mesh_candidate=='right:DM1']
    dm1.groupby(['root_id','direction']).size().rename('candidate_site_count').reset_index().to_csv(OUT/'right-DM1-counts.csv',index=False)
    final={'status':'count_preserving_partial_bilateral_mapping_complete','endpoints':len(endpoints),
           'all_68_root_direction_totals_preserved':True,'right_individually_eligible_regions':sorted(eligible),
           'eligible_candidate_endpoints':int(accepted.sum()),'right_DM1_unique_endpoints':len(dm1),
           'status_counts':dict(Counter(endpoints.region_status)),'all_fitted_neighbor_meshes_included':True,
           'physiological_compartments_created':0,'controller_changed':False,'wall_seconds':time.monotonic()-start,
           'sources_reverified':all(sha(ROOT/p)==h for p,h in protocol['source_hashes'].items())}
    (OUT/'mapping-results.json').write_text(json.dumps(final,indent=2)+'\n');print(json.dumps(final,indent=2))

if __name__=='__main__':main()
