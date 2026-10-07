"""Spatial audit of covered sites. Unique mesh labels remain anatomical candidates."""
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

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'reports/brain-integration/recovery/patchy-compartment-mapping-v1'


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def classify(points, meshes):
    names=list(meshes); membership=np.zeros((len(points),len(names)),bool)
    for j,name in enumerate(names):
        mesh=meshes[name]
        if not mesh.is_watertight or not mesh.is_winding_consistent:continue
        bbox=np.all((points>=mesh.bounds[0])&(points<=mesh.bounds[1]),axis=1)
        ids=np.flatnonzero(bbox)
        for first in range(0,len(ids),500):
            part=ids[first:first+500];membership[part,j]=mesh.contains(points[part])
        (OUT/'spatial-progress.json').write_text(json.dumps({'status':'classifying','region':name,'region_index':j+1,'regions':len(names),'points':len(points)})+'\n')
    return unique_mesh_assignment(membership,names),membership.sum(axis=1)


def main():
    start=time.monotonic()
    paths=[OUT/'source/flywire_al.surf.rda',OUT/'selected-synapses.csv',OUT/'anchor-synapses.csv',Path(__file__).resolve(),ROOT/'flygarden/compartment_mapping.py']
    protocol={'version':1,'source_hashes':{str(p.relative_to(ROOT)):sha(p) for p in paths},
              'transform':'identity only; no fitted scale, translation, mirroring or aliasing',
              'mesh_source_commit':'9d31eb21e3ff5127152debc978f0ee5c8751b750',
              'coordinate_location':'Source synapse coordinate point, not distinct pre/post release and reception locations',
              'criteria':{'all_meshes_closed':True,'all_meshes_consistent_winding':True,
                          'unique_membership_required':True,
                          'dominant_side_anchor_unique_coverage_min':.70,
                          'dominant_side_anchor_correct_label_fraction_min':.80},
              'no_controller_assignment':True,
              'uncovered_edges':'Keep every imported count; leave missing positions unavailable.'}
    (OUT/'spatial-protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    warnings.filterwarnings('ignore',message='Missing constructor for R class')
    data=rdata.read_rda(OUT/'source/flywire_al.surf.rda')['flywire_al.surf']
    v=data['Vertices'][['X','Y','Z']].to_numpy();meshes={}
    for name,frame in data['Regions'].items():
        faces=frame.to_numpy(dtype=np.int64)-1
        assert faces.min()>=0 and faces.max()<len(v)
        used=np.unique(faces);remap=np.full(len(v),-1);remap[used]=np.arange(len(used))
        mesh=trimesh.Trimesh(vertices=v[used],faces=remap[faces],process=True)
        if not mesh.is_watertight or not mesh.is_winding_consistent:
            raise ValueError('Invalid glomerular mesh')
        meshes[str(name)]=mesh
    anchors=pd.read_csv(OUT/'anchor-synapses.csv',dtype={'pre_root_id':str,'post_root_id':str})
    label,multiplicity=classify(anchors[['x','y','z']].to_numpy(),meshes)
    anchors['mesh_candidate']=label;anchors['mesh_memberships']=multiplicity
    anchors['known_label_matches_mesh']=anchors.glomerulus==anchors.mesh_candidate
    anchors.to_csv(OUT/'anchor-spatial-audit.csv',index=False)
    side_stats={}
    for side,a in anchors.groupby('PN_side'):
        unique=a.mesh_memberships==1
        side_stats[side]={'sampled_coordinates':len(a),'unique_enclosed':int(unique.sum()),
                          'unique_fraction':float(unique.mean()),
                          'correct_unique_fraction':float(a.loc[unique,'known_label_matches_mesh'].mean()) if unique.any() else 0.,
                          'correct_sample_fraction':float(a.known_label_matches_mesh.mean())}
    dominant=max(side_stats,key=lambda s:side_stats[s]['correct_sample_fraction'])
    stats=side_stats[dominant]
    alignment=stats['unique_fraction']>=.70 and stats['correct_unique_fraction']>=.80
    # Per-glomerulus results expose failures rather than hiding them in a pooled gate.
    per_glom=[]
    for (glom,side),a in anchors.groupby(['glomerulus','PN_side']):
        u=a.mesh_memberships==1
        per_glom.append({'glomerulus':glom,'PN_side':side,'samples':len(a),
                         'atlas_name_present':glom in meshes,'unique_fraction':float(u.mean()),
                         'correct_sample_fraction':float(a.known_label_matches_mesh.mean())})
    (OUT/'anchor-by-glomerulus.json').write_text(json.dumps(per_glom,indent=2)+'\n')
    selected=pd.read_csv(OUT/'selected-synapses.csv',dtype={'pre_root_id':str,'post_root_id':str})
    label,multiplicity=classify(selected[['x','y','z']].to_numpy(),meshes)
    selected['mesh_candidate']=label;selected['mesh_memberships']=multiplicity
    selected['assignment_status']=np.where(multiplicity==1,'candidate_unique_enclosure','unavailable') if alignment else 'unavailable_alignment_failed'
    selected.to_csv(OUT/'spatial-synapses.csv',index=False)
    neuron=pd.read_json(OUT/'neurons.json',dtype={'root_id':str})
    lookup={str(r['root_id']):r for r in neuron.to_dict('records')}
    summaries=[]
    for rid,a in lookup.items():
        for direction,field in [('input','post_root_id'),('output','pre_root_id')]:
            syn=selected[selected[field]==rid]
            counts=Counter(syn.mesh_candidate)
            summaries.append({'root_id':rid,'cell_type':a['cell_type'],'side':a['side'],'direction':direction,
                              'covered_sites':len(syn),'mesh_labels':dict(counts),
                              'candidate_unique_sites':int((syn.mesh_memberships==1).sum()),
                              'not_a_physiological_compartment_assignment':True})
    (OUT/'spatial-root-summary.json').write_text(json.dumps(summaries,indent=2)+'\n')
    result={'version':1,'status':'spatial_audit_complete','regions':len(meshes),
            'all_meshes_closed_consistent':True,'anchor_side_statistics':side_stats,
            'atlas_hemisphere_inferred_from_anchors':dominant,'alignment_gate_passed':alignment,
            'selected_sites':len(selected),'unique_enclosed_sites':int((multiplicity==1).sum()),
            'outside_sites':int((multiplicity==0).sum()),'overlapping_sites':int((multiplicity>1).sum()),
            'application_changed':False,'controller_compartments_created':0,
            'wall_seconds':time.monotonic()-start,'versions':{'rdata':rdata.__version__,'trimesh':trimesh.__version__},
            'sources_reverified':all(sha(ROOT/p)==h for p,h in protocol['source_hashes'].items())}
    assert result['sources_reverified']
    (OUT/'spatial-results.json').write_text(json.dumps(result,indent=2)+'\n')
    (OUT/'spatial-progress.json').write_text(json.dumps({'status':'complete'})+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
