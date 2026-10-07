"""Exact version-783 skeleton / synapse geometry audit of all 34 patchy roots."""
from pathlib import Path
import json
import resource
import shutil
import time
import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from flygarden.morphology import parse_skeleton
from flygarden.skeleton_site_audit import nearest_segments
from scripts.validate_local_glomerular_registration import sha,ROOT,V1,V2

OUT=ROOT/'reports/brain-integration/recovery/patchy-skeleton-site-audit-v1'

def main():
    started=time.monotonic();OUT.mkdir(parents=True,exist_ok=True)
    roots=json.loads((V1/'neurons.json').read_text());cache=ROOT/'data/morphology/v783'
    inputs=[V2/'mapped-endpoints.parquet',V1/'neurons.json',ROOT/'flygarden/morphology.py',
            ROOT/'flygarden/skeleton_site_audit.py',Path(__file__).resolve()]
    for r in roots:inputs.extend([cache/(r['root_id']+'.bin'),cache/(r['root_id']+'.json')])
    protocol={'version':1,'source_hashes':{str(p.relative_to(ROOT)):sha(p) for p in inputs},
              'materialization':783,'units':'nanometers','skeletons':'Exact-root native full-resolution cached Neuroglancer skeletons',
              'method':'Exact point-to-edge distances, center-radius bounded search over all edge-length bins',
              'no_site_filter_or_graph_change':True,'distance_thresholds_nm':[1000,10000],
              'threshold_meaning':'Diagnostic only; do not discard or reassign a source synapse',
              'electrical_coupling_or_compartment_claim':False,'radius_or_membrane_parameters_available':False}
    (OUT/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    if shutil.disk_usage(OUT).free<2*1024**3+512*1024**2:raise RuntimeError('Disk reserve')
    sites=pd.read_parquet(V2/'mapped-endpoints.parquet');summaries=[];mapped=[]
    for i,r in enumerate(roots):
        rid=r['root_id'];meta=json.loads((cache/(rid+'.json')).read_text());binary=cache/(rid+'.bin')
        if str(meta['id'])!=rid or meta['materialization']!=783 or meta['units']!='nanometers' or sha(binary)!=meta['sha256']:
            raise ValueError('Skeleton identity mismatch')
        vertices,edges=parse_skeleton(binary.read_bytes())
        graph=coo_matrix((np.ones(len(edges)),(edges[:,0],edges[:,1])),shape=(len(vertices),len(vertices)))
        components,_=connected_components(graph,directed=False)
        a=sites[sites.root_id==rid].copy()
        distance,index,fraction=nearest_segments(a[['x','y','z']].to_numpy(),vertices,edges)
        a['skeleton_edge_index']=index;a['edge_fraction']=fraction;a['distance_to_skeleton_nm']=distance;mapped.append(a)
        degrees=np.bincount(edges.ravel(),minlength=len(vertices))
        length=np.linalg.norm(vertices[edges[:,0]]-vertices[edges[:,1]],axis=1)
        summary={'root_id':rid,'cell_type':r['cell_type'],'soma_side':r['side'],'source':meta['source'],
                 'skeleton_sha256':meta['sha256'],'vertices':len(vertices),'edges':len(edges),'connected_components':int(components),
                 'cycle_rank':int(len(edges)-len(vertices)+components),'branch_vertices':int((degrees>2).sum()),
                 'isolated_vertices':int((degrees==0).sum()),'edge_length_p99_nm':float(np.quantile(length,.99)),
                 'max_edge_length_nm':float(length.max()),'sites':len(a),
                 'median_distance_nm':float(np.median(distance)),'p95_distance_nm':float(np.quantile(distance,.95)),
                 'p99_distance_nm':float(np.quantile(distance,.99)),'max_distance_nm':float(distance.max()),
                 'within_1um_fraction':float((distance<=1000).mean()),'beyond_10um_sites':int((distance>10000).sum())}
        summaries.append(summary)
        (OUT/'progress.json').write_text(json.dumps({'status':'auditing','roots_done':i+1,'roots':len(roots),'root_id':rid})+'\n')
    result=pd.concat(mapped,ignore_index=True);assert len(result)==len(sites)
    result.to_parquet(OUT/'skeleton-mapped-endpoints.parquet',index=False,compression='zstd')
    (OUT/'root-summary.json').write_text(json.dumps(summaries,indent=2)+'\n')
    distances=result.distance_to_skeleton_nm.to_numpy()
    final={'status':'audit_complete','roots':len(roots),'endpoints_preserved':len(result),
           'median_distance_nm':float(np.median(distances)),'p95_distance_nm':float(np.quantile(distances,.95)),
           'p99_distance_nm':float(np.quantile(distances,.99)),'max_distance_nm':float(distances.max()),
           'within_1um_fraction':float((distances<=1000).mean()),'beyond_10um_endpoints':int((distances>10000).sum()),
           'multi_component_skeletons':sum(r['connected_components']>1 for r in summaries),
           'nonzero_cycle_rank_skeletons':sum(r['cycle_rank']!=0 for r in summaries),
           'application_changed':False,'physiological_compartments_created':0,
           'wall_seconds':time.monotonic()-started,'peak_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
           'sources_reverified':all(sha(ROOT/p)==h for p,h in protocol['source_hashes'].items())}
    (OUT/'results.json').write_text(json.dumps(final,indent=2)+'\n')
    (OUT/'progress.json').write_text(json.dumps({'status':'complete'})+'\n')
    print(json.dumps(final,indent=2))

if __name__=='__main__':main()
