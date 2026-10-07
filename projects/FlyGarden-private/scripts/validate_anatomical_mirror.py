"""Frozen-source opposite-hemisphere atlas validation; no model changes."""
from pathlib import Path
import hashlib
import json
import time
import warnings
import numpy as np
import pandas as pd
import rdata
import trimesh
from flygarden.anatomical_mirror import AnatomicalMirror
from flygarden.compartment_mapping import unique_mesh_assignment

ROOT = Path(__file__).resolve().parents[1]
OLD = ROOT/'reports/brain-integration/recovery/patchy-compartment-mapping-v1'
OUT = ROOT/'reports/brain-integration/recovery/patchy-compartment-mapping-v2'

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def classify(points, meshes, phase):
    names = list(meshes)
    membership = np.zeros((len(points), len(names)), bool)
    for j, name in enumerate(names):
        mesh = meshes[name]
        if not mesh.is_watertight or not mesh.is_winding_consistent:
            raise ValueError('Invalid mesh: '+name)
        ids = np.flatnonzero(np.all((points >= mesh.bounds[0]) & (points <= mesh.bounds[1]), axis=1))
        for first in range(0, len(ids), 500):
            part = ids[first:first+500]
            membership[part, j] = mesh.contains(points[part])
        (OUT/'mirror-progress.json').write_text(json.dumps({'phase':phase, 'region':name, 'regions_done':j+1, 'regions':len(names)})+'\n')
    return unique_mesh_assignment(membership, names), membership.sum(axis=1)

def main():
    started = time.monotonic()
    inputs = [OUT/'source/FLYWIRE_mirror_landmarks.csv', OUT/'source/template_brain_meta.json',
              OLD/'source/flywire_al.surf.rda', OLD/'anchor-synapses.csv',
              ROOT/'flygarden/anatomical_mirror.py', Path(__file__).resolve()]
    protocol = {'version':1, 'source_hashes':{str(p.relative_to(ROOT)):sha(p) for p in inputs},
                'flybrains_revision':'273333c8d8bf5adeebebd274e554621462e388bd',
                'navis_revision':'cb9a5915b6b3587cb81154f4f77ffc62fe12b03a',
                'registration':'native x -> bbox x_min+x_max-x -> published TPS',
                'landmark_fit':'published source/target only; no sensory-anchor fit',
                'units':'nanometers', 'criteria':{'right_unique_fraction_min':.70, 'right_correct_unique_fraction_min':.80},
                'anchors':'v1 fixed ORN-to-cognate-uniglomerular-PN reservoir, stratified by PN soma side',
                'limitations':['Soma side is not a direct branch label', 'Mirror is an approximate registration, not new traced anatomy',
                               'No physiological or electrical compartment claim']}
    (OUT/'mirror-protocol.json').write_text(json.dumps(protocol, indent=2)+'\n')
    lm = pd.read_csv(inputs[0])
    meta = next(x for x in json.loads(inputs[1].read_text()) if x['label']=='FLYWIRE')
    mirror = AnatomicalMirror(lm[['x_flip','y_flip','z_flip']], lm[['x_mirr','y_mirr','z_mirr']], sum(meta['boundingbox'][:2]))
    error = np.linalg.norm(mirror.warp_flipped(lm[['x_flip','y_flip','z_flip']]) - lm[['x_mirr','y_mirr','z_mirr']].to_numpy(), axis=1)
    if error.max() > .01:
        raise ValueError('Published landmark interpolation exceeds 0.01 nm')
    warnings.filterwarnings('ignore', message='Missing constructor for R class')
    data = rdata.read_rda(inputs[2])['flywire_al.surf']
    vertices = data['Vertices'][['X','Y','Z']].to_numpy()
    transformed = mirror.transform(vertices)
    meshes = {}; inventory = []
    for name, frame in data['Regions'].items():
        faces = frame.to_numpy(dtype=np.int64)-1
        used = np.unique(faces); remap = np.full(len(vertices), -1); remap[used] = np.arange(len(used))
        mesh = trimesh.Trimesh(vertices=transformed[used], faces=remap[faces][:, ::-1], process=True)
        if not mesh.is_watertight or not mesh.is_winding_consistent or not np.isfinite(mesh.vertices).all():
            raise ValueError('Invalid warped mesh: '+name)
        meshes[str(name)] = mesh
        inventory.append({'name':str(name), 'watertight':bool(mesh.is_watertight), 'consistent_winding':bool(mesh.is_winding_consistent),
                          'vertices':len(mesh.vertices), 'faces':len(mesh.faces), 'bounds':mesh.bounds.tolist(), 'volume_nm3':float(mesh.volume)})
    anchors = pd.read_csv(inputs[3], dtype={'pre_root_id':str, 'post_root_id':str})
    label, multiplicity = classify(anchors[['x','y','z']].to_numpy(), meshes, 'anchors')
    anchors['mirror_candidate'] = label; anchors['mirror_memberships'] = multiplicity
    anchors['label_match'] = anchors.glomerulus == anchors.mirror_candidate
    anchors.to_csv(OUT/'mirror-anchor-audit.csv', index=False)
    stats = {}; per_glom = []
    for side, a in anchors.groupby('PN_side'):
        unique = a.mirror_memberships == 1
        stats[side] = {'samples':len(a), 'unique_fraction':float(unique.mean()),
                       'correct_unique_fraction':float(a.loc[unique,'label_match'].mean()) if unique.any() else 0.,
                       'correct_sample_fraction':float(a.label_match.mean())}
    for (glom, side), a in anchors.groupby(['glomerulus','PN_side']):
        unique = a.mirror_memberships == 1
        per_glom.append({'glomerulus':glom, 'PN_side':side, 'samples':len(a), 'unique_fraction':float(unique.mean()),
                         'correct_unique_fraction':float(a.loc[unique,'label_match'].mean()) if unique.any() else 0.,
                         'atlas_name_present':glom in meshes})
    right = stats.get('right', {})
    passed = right.get('unique_fraction', 0)>=.70 and right.get('correct_unique_fraction', 0)>=.80
    np.savez_compressed(OUT/'mirrored-atlas-vertices.npz', vertices=transformed)
    (OUT/'mirror-mesh-inventory.json').write_text(json.dumps(inventory, indent=2)+'\n')
    (OUT/'mirror-anchor-by-glomerulus.json').write_text(json.dumps(per_glom, indent=2)+'\n')
    result = {'status':'validation_complete', 'alignment_gate_passed':passed, 'anchor_side_statistics':stats,
              'landmarks':len(lm), 'landmark_max_error_nm':float(error.max()), 'regions':len(meshes),
              'all_meshes_closed_consistent':True, 'self_intersections_tested':False,
              'physiological_compartments_created':0, 'application_changed':False,
              'sources_reverified':all(sha(ROOT/p)==h for p,h in protocol['source_hashes'].items()),
              'wall_seconds':time.monotonic()-started}
    (OUT/'mirror-results.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))

if __name__ == '__main__':
    main()
