"""Freeze a calibration-only shape correction, then open a fresh source-class probe."""
from pathlib import Path
import json
import resource
import time
import warnings
import numpy as np
import pandas as pd
import rdata
import trimesh
from flygarden.anatomical_mirror import AnatomicalMirror
from flygarden.glomerular_affine import balanced_moments,covariance_map,apply_affine
from flygarden.compartment_mapping import unique_mesh_assignment
from scripts.validate_local_glomerular_registration import sha,ROOT,V1,V2
from scripts.prepare_glomerular_affine_data import OUT

def classify(points,meshes,phase):
    names=list(meshes);membership=np.zeros((len(points),len(names)),bool)
    for j,(name,mesh) in enumerate(meshes.items()):
        ids=np.flatnonzero(np.all((points>=mesh.bounds[0])&(points<=mesh.bounds[1]),axis=1))
        for k in range(0,len(ids),500):
            part=ids[k:k+500];membership[part,j]=mesh.contains(points[part])
        (OUT/'progress.json').write_text(json.dumps({'status':'classifying','candidate':phase,'region':name,'regions_done':j+1,'regions':len(names)})+'\n')
    return unique_mesh_assignment(membership,names),membership.sum(axis=1)

def evaluate(frame,name,seed):
    a=frame.copy();a['unique']=a[name+'_memberships']==1;a['correct']=a[name+'_label']==a.glomerulus
    values=a.groupby('ORN_root_id').agg(total=('correct','size'),unique=('unique','sum'),correct=('correct','sum')).to_numpy(float)
    rng=np.random.default_rng(seed);boot=[]
    if len(values):
        for _ in range(2000):
            n,u,c=values[rng.integers(len(values),size=len(values))].sum(axis=0);boot.append([u/n,c/u if u else 0.])
    ci=np.quantile(boot,[.025,.975],axis=0).T.tolist() if boot else [[0,0],[0,0]]
    stats={'sites':len(a),'receiving_ORNs':len(values),'unique_fraction':float(a.unique.mean()) if len(a) else 0.,
           'correct_unique_fraction':float(a.loc[a.unique,'correct'].mean()) if a.unique.any() else 0.,
           'correct_all_fraction':float(a.correct.mean()) if len(a) else 0.,
           'outside_sites':int((a[name+'_memberships']==0).sum()),'overlapping_sites':int((a[name+'_memberships']>1).sum()),
           'bootstrap_95_intervals':{'unique_fraction':ci[0],'correct_unique_fraction':ci[1]}}
    stats['adequate_probe']=len(a)>=50 and len(values)>=5
    stats['point_gate_passed']=stats['adequate_probe'] and stats['unique_fraction']>=.70 and stats['correct_unique_fraction']>=.80
    stats['confidence_gate_passed']=stats['adequate_probe'] and ci[0][0]>=.70 and ci[1][0]>=.80
    return stats

def main():
    started=time.monotonic()
    inputs=[OUT/'calibration.parquet',OUT/'sealed_probe.parquet',V1/'source/flywire_al.surf.rda',
            V2/'source/FLYWIRE_mirror_landmarks.csv',V2/'source/template_brain_meta.json',V2/'mirrored-atlas-vertices.npz',
            ROOT/'flygarden/glomerular_affine.py',ROOT/'flygarden/anatomical_mirror.py',Path(__file__).resolve()]
    protocol={'version':1,'source_hashes':{str(p.relative_to(ROOT)):sha(p) for p in inputs},
              'candidate':'Published mirrored mesh plus positive-definite neuron-balanced covariance transport and center correction',
              'calibration_only':True,'means_and_covariances':'Each presynaptic calibration ORN contributes equal total weight; source AL_L/AL_R sites only',
              'min_calibration_roots_per_side':5,'min_calibration_sites_per_side':50,
              'scale_bounds':[.5,2.],'extreme_or_degenerate_fit':'Reject; never clamp, regularize, substitute identity or tune on probe',
              'translation_bound':'Source mirrored mesh bounding-box diagonal; reject larger center shifts',
              'fresh_validation':'Sealed PN -> cognate held-out ORN sites; neither these source synapses nor these receiving ORNs entered fitting',
              'previous_tested_cohorts':'Historical only; not evaluated for acceptance here',
              'gates':{'unique_min':.70,'correct_among_unique_min':.80,'both_neuron_cluster_95_lower_bounds_required':True,
                       'min_probe_sites':50,'min_receiving_ORNs':5,'DM1_requires_its_own_confidence_gate':True},
              'bootstrap':{'iterations':2000,'pooled_seed':17005,'per_glomerulus_seed':17006,'unit':'receiving ORN root'},
              'coordinate_units':'nanometers','not_manually_traced_right_boundaries':True,'electrical_parameters_assigned':False,
              'application_changed':False,'no_probe_retuning':True}
    (OUT/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    cal=pd.read_parquet(OUT/'calibration.parquet').astype({'x':float,'y':float,'z':float})
    lm=pd.read_csv(V2/'source/FLYWIRE_mirror_landmarks.csv');meta=next(x for x in json.loads((V2/'source/template_brain_meta.json').read_text()) if x['label']=='FLYWIRE')
    mirror=AnatomicalMirror(lm[['x_flip','y_flip','z_flip']],lm[['x_mirr','y_mirr','z_mirr']],sum(meta['boundingbox'][:2]))
    left_mask=cal.side=='left';cal.loc[left_mask,['x','y','z']]=mirror.transform(cal.loc[left_mask,['x','y','z']].to_numpy())
    warnings.filterwarnings('ignore',message='Missing constructor for R class')
    data=rdata.read_rda(V1/'source/flywire_al.surf.rda')['flywire_al.surf'];v=np.load(V2/'mirrored-atlas-vertices.npz')['vertices']
    baseline={};candidate={};fits=[]
    for name,faces_frame in data['Regions'].items():
        faces=faces_frame.to_numpy(np.int64)-1;used=np.unique(faces);remap=np.full(len(v),-1);remap[used]=np.arange(len(used))
        mesh=trimesh.Trimesh(v[used],remap[faces][:,::-1],process=True);baseline[str(name)]=mesh
        left=cal[(cal.glomerulus==name)&(cal.side=='left')];right=cal[(cal.glomerulus==name)&(cal.side=='right')]
        row={'glomerulus':str(name),'left_calibration_sites':len(left),'right_calibration_sites':len(right),
             'left_calibration_ORNs':int(left.ORN_root_id.nunique()),'right_calibration_ORNs':int(right.ORN_root_id.nunique()),'available':False}
        try:
            if min(len(left),len(right))<50 or min(left.ORN_root_id.nunique(),right.ORN_root_id.nunique())<5:
                raise ValueError('Sparse calibration group')
            source_center,source_cov=balanced_moments(left[['x','y','z']],left.ORN_root_id)
            target_center,target_cov=balanced_moments(right[['x','y','z']],right.ORN_root_id)
            if np.linalg.norm(target_center-source_center)>np.linalg.norm(np.diff(mesh.bounds,axis=0)):
                raise ValueError('Center shift exceeds frozen geometric bound')
            matrix=covariance_map(source_cov,target_cov)
            transformed=mesh.copy();transformed.vertices=apply_affine(mesh.vertices,source_center,target_center,matrix)
            if not transformed.is_watertight or not transformed.is_winding_consistent or not np.isfinite(transformed.vertices).all():
                raise ValueError('Invalid affine geometry')
            candidate[str(name)]=transformed
            row.update({'available':True,'source_center_nm':source_center.tolist(),'target_center_nm':target_center.tolist(),
                        'matrix':matrix.tolist(),'scales':np.linalg.eigvalsh(matrix).tolist(),'volume_ratio':float(np.linalg.det(matrix)),
                        'source_covariance_nm2':source_cov.tolist(),'target_covariance_nm2':target_cov.tolist()})
        except ValueError as error:row['unavailable_reason']=str(error)
        fits.append(row)
    (OUT/'frozen-fit.json').write_text(json.dumps({'version':1,'fits':fits,'calibration_ORNs':sorted(set(cal.ORN_root_id)),
       'sealed_probe_sha256':sha(OUT/'sealed_probe.parquet'),'probe_not_opened_yet':True},indent=2)+'\n')
    np.savez_compressed(OUT/'affine-atlas.npz',**{name:m.vertices for name,m in candidate.items()})
    # Only now open positions and compute the fresh cohort's outcomes.
    probe=pd.read_parquet(OUT/'sealed_probe.parquet')
    assert not set(cal.ORN_root_id)&set(probe.ORN_root_id)
    for candidate_name,meshes in [('published_mirror',baseline),('affine_candidate',candidate)]:
        label,membership=classify(probe[['x','y','z']].to_numpy(),meshes,candidate_name)
        probe[candidate_name+'_label']=label;probe[candidate_name+'_memberships']=membership
    probe.to_parquet(OUT/'evaluated-fresh-probe.parquet',index=False,compression='zstd')
    pooled={name:evaluate(probe,name,17005) for name in ['published_mirror','affine_candidate']}
    per=[]
    for glom,a in probe.groupby('glomerulus'):
        for name in ['published_mirror','affine_candidate']:
            s=evaluate(a,name,17006);s.update({'glomerulus':glom,'candidate':name});per.append(s)
    dm1=next((s for s in per if s['glomerulus']=='DM1' and s['candidate']=='affine_candidate'),None)
    result={'version':1,'status':'fresh_shape_registration_audit_complete','available_regions':len(candidate),
            'calibration_sites':len(cal),'fresh_probe_sites':len(probe),'pooled':pooled,'DM1':dm1,
            'pooled_gate_passed':pooled['affine_candidate']['confidence_gate_passed'],
            'DM1_gate_passed':bool(dm1 and dm1['confidence_gate_passed']),
            'per_glomerulus_confidence_gates_passed':sum(s['confidence_gate_passed'] for s in per if s['candidate']=='affine_candidate'),
            'manual_boundary_validation':False,'electrical_compartments_created':0,'application_changed':False,
            'wall_seconds':time.monotonic()-started,'peak_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            'sources_reverified':all(sha(ROOT/p)==h for p,h in protocol['source_hashes'].items())}
    (OUT/'per-glomerulus-results.json').write_text(json.dumps(per,indent=2)+'\n')
    (OUT/'results.json').write_text(json.dumps(result,indent=2)+'\n')
    (OUT/'progress.json').write_text(json.dumps({'status':'complete'})+'\n');print(json.dumps(result,indent=2))

if __name__=='__main__':main()
