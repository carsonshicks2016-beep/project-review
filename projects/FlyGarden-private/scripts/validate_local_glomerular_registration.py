"""Prospective grouped validation of a translation-only local atlas candidate."""
from pathlib import Path
import hashlib
import json
import resource
import time
import warnings
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.ipc as ipc
import pyarrow.parquet as pq
import rdata
import trimesh
from flygarden.anatomical_mirror import AnatomicalMirror
from flygarden.compartment_mapping import partner_glomerulus,unique_mesh_assignment
from flygarden.local_atlas_registration import role,translation

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'reports/brain-integration/recovery'
V1=BASE/'patchy-compartment-mapping-v1'
V2=BASE/'patchy-compartment-mapping-v2'
OUT=BASE/'local-glomerular-registration-v1'

def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
    return h.hexdigest()

def classify(points,meshes):
    names=list(meshes);membership=np.zeros((len(points),len(names)),bool)
    for j,(name,mesh) in enumerate(meshes.items()):
        indices=np.flatnonzero(np.all((points>=mesh.bounds[0])&(points<=mesh.bounds[1]),axis=1))
        for i in range(0,len(indices),500):
            subset=indices[i:i+500];membership[subset,j]=mesh.contains(points[subset])
        (OUT/'progress.json').write_text(json.dumps({'status':'classifying','region':name,'regions_done':j+1,'regions':len(names)})+'\n')
    return unique_mesh_assignment(membership,names),membership.sum(axis=1)

def stats(frame):
    if frame.empty:return {'samples':0,'roots':0,'unique_fraction':0.,'correct_unique_fraction':0.,'correct_sample_fraction':0.}
    unique=frame.memberships==1
    return {'samples':len(frame),'roots':int(frame.pre_root_id.nunique()),'unique_fraction':float(unique.mean()),
            'correct_unique_fraction':float(frame.loc[unique,'correct'].mean()) if unique.any() else 0.,
            'correct_sample_fraction':float(frame.correct.mean())}

def main():
    start=time.monotonic();OUT.mkdir(parents=True,exist_ok=True)
    identity=json.loads((V2/'archive-identity.json').read_text())
    archive=V2/'source/flywire_synapses_783.feather'
    if not identity['published_checksum_verified'] or sha(archive)!=identity['sha256']:
        raise ValueError('Pinned archive identity mismatch')
    inputs=[ROOT/'data/annotations.tsv',V1/'anchor-synapses.csv',V1/'source/flywire_al.surf.rda',
            V2/'source/FLYWIRE_mirror_landmarks.csv',V2/'source/template_brain_meta.json',
            V2/'mirrored-atlas-vertices.npz',ROOT/'flygarden/compartment_mapping.py',
            ROOT/'flygarden/anatomical_mirror.py',ROOT/'flygarden/local_atlas_registration.py',Path(__file__).resolve()]
    protocol={'version':1,'source_hashes':{str(p.relative_to(ROOT)):sha(p) for p in inputs},'archive_identity':identity,
              'candidate':'Published mirrored glomerular meshes plus one translation vector per glomerulus. No scale/rotation/shape fit.',
              'split':'SHA256 salt flygarden-local-registration-v1-783 + exact ORN root; 3/5 calibration, 2/5 probe; all sites of a root share one role',
              'probe_exclusions':'Exclude every pre/post root pair represented in the previously inspected v1 right-side anchor reservoir',
              'anchor_definition':'Explicit cognate ORN -> annotated uniglomerular PN, exact labels only; PN soma side; postsynaptic location in native nm',
              'fit':'Right neuron-balanced median minus published mirror of left neuron-balanced median; calibration roots only',
              'translation_limit':'Reject shift longer than the source mirrored mesh bounding-box diagonal; never clamp it',
              'min_calibration_roots_per_side':5,'min_calibration_sites_per_side':50,'min_probe_roots_per_glomerulus':5,'min_probe_sites_per_glomerulus':50,
              'gates':{'unique_enclosure_min':.70,'correct_among_unique_min':.80,'root_cluster_bootstrap_95_lower_bounds_must_pass':True},
              'bootstrap':{'seed':17003,'iterations':2000,'unit':'ORN root'},
              'no_controller_change':True,'electrical_compartments_created':0,'not_independent_manual_boundary_validation':True,
              'no_threshold_or_transform_retuning_after_probe':True}
    (OUT/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    annotations=pd.read_csv(inputs[0],sep='\t',dtype=str).fillna('')
    ann={int(r['root_id']):r for r in annotations.to_dict('records')}
    labels={rid:partner_glomerulus(r) for rid,r in ann.items()}
    orn={rid:label[0] for rid,label in labels.items() if label[1]=='ORN_annotation'}
    pn={rid:label[0] for rid,label in labels.items() if label[1]=='uniglomerular_PN_annotation' and ann[rid]['side'] in ('left','right')}
    orn_ids=np.array(list(orn),dtype=np.int64);pn_ids=np.array(list(pn),dtype=np.int64)
    prior=pd.read_csv(inputs[1],dtype={'pre_root_id':str,'post_root_id':str})
    prior=prior[prior.PN_side=='right']
    old_pairs={(int(a),int(b)) for a,b in zip(prior.pre_root_id,prior.post_root_id)}
    writer=None;offset=0;selected=0
    tmp=OUT/'anchor-sites.parquet.partial'
    with pa.OSFile(str(archive),'rb') as f:
        reader=ipc.open_file(f)
        for i in range(reader.num_record_batches):
            batch=reader.get_batch(i)
            pre=batch.column(batch.schema.get_field_index('pre_pt_root_id')).to_numpy()
            post=batch.column(batch.schema.get_field_index('post_pt_root_id')).to_numpy()
            ids=np.flatnonzero(np.isin(pre,orn_ids)&np.isin(post,pn_ids))
            ids=np.array([j for j in ids if orn[int(pre[j])]==pn[int(post[j])]],dtype=np.int64)
            if len(ids):
                columns=['id','pre_pt_root_id','post_pt_root_id','post_pt_position_x','post_pt_position_y','post_pt_position_z']
                t=pa.Table.from_batches([batch]).select(columns).take(pa.array(ids));df=t.to_pandas()
                frame=pd.DataFrame({'synapse_id':df.id.astype(str),'pre_root_id':df.pre_pt_root_id.astype(str),
                    'post_root_id':df.post_pt_root_id.astype(str),'glomerulus':[orn[int(r)] for r in df.pre_pt_root_id],
                    'PN_side':[ann[int(r)]['side'] for r in df.post_pt_root_id],
                    'role':[role(r) for r in df.pre_pt_root_id],
                    'previously_tested_pair':[(int(a),int(b)) in old_pairs for a,b in zip(df.pre_pt_root_id,df.post_pt_root_id)],
                    'x':df.post_pt_position_x,'y':df.post_pt_position_y,'z':df.post_pt_position_z,'source_row':ids+offset})
                table=pa.Table.from_pandas(frame,preserve_index=False)
                if writer is None:writer=pq.ParquetWriter(tmp,table.schema,compression='zstd')
                writer.write_table(table);selected+=len(frame)
            offset+=batch.num_rows
            if i%50==0:(OUT/'progress.json').write_text(json.dumps({'status':'extracting_anchors','batches_done':i+1,'batches':reader.num_record_batches,'selected':selected})+'\n')
    if writer is None:raise ValueError('No annotated anchors')
    writer.close();tmp.replace(OUT/'anchor-sites.parquet')
    anchors=pd.read_parquet(OUT/'anchor-sites.parquet')
    if anchors.synapse_id.duplicated().any():raise ValueError('Duplicate anchor IDs')
    cal=anchors[anchors.role=='calibration']
    probe=anchors[(anchors.role=='probe')&(anchors.PN_side=='right')&(~anchors.previously_tested_pair)].copy()
    assert not set(cal.pre_root_id)&set(probe.pre_root_id)
    assert not any((int(a),int(b)) in old_pairs for a,b in zip(probe.pre_root_id,probe.post_root_id))
    lm=pd.read_csv(V2/'source/FLYWIRE_mirror_landmarks.csv')
    meta=next(x for x in json.loads((V2/'source/template_brain_meta.json').read_text()) if x['label']=='FLYWIRE')
    mirror=AnatomicalMirror(lm[['x_flip','y_flip','z_flip']],lm[['x_mirr','y_mirr','z_mirr']],sum(meta['boundingbox'][:2]))
    warnings.filterwarnings('ignore',message='Missing constructor for R class')
    data=rdata.read_rda(V1/'source/flywire_al.surf.rda')['flywire_al.surf']
    vertices=np.load(V2/'mirrored-atlas-vertices.npz')['vertices']
    base_meshes={};candidate={};fit=[]
    for name,faces_df in data['Regions'].items():
        faces=faces_df.to_numpy(dtype=np.int64)-1;used=np.unique(faces);remap=np.full(len(vertices),-1);remap[used]=np.arange(len(used))
        mesh=trimesh.Trimesh(vertices[used],remap[faces][:,::-1],process=True);base_meshes[str(name)]=mesh
        left=cal[(cal.glomerulus==name)&(cal.PN_side=='left')];right=cal[(cal.glomerulus==name)&(cal.PN_side=='right')]
        shift=translation(left,right,mirror)
        valid=shift is not None and np.linalg.norm(shift)<=np.linalg.norm(np.diff(mesh.bounds,axis=0))
        row={'glomerulus':str(name),'left_calibration_roots':int(left.pre_root_id.nunique()),'right_calibration_roots':int(right.pre_root_id.nunique()),
             'left_calibration_sites':len(left),'right_calibration_sites':len(right),'translation_nm':None if shift is None else shift.tolist(),
             'candidate_available':bool(valid)}
        fit.append(row)
        if valid:
            moved=mesh.copy();moved.vertices+=shift
            if not moved.is_watertight or not moved.is_winding_consistent:raise ValueError('Invalid translated mesh')
            candidate[str(name)]=moved
    # Freeze every fitted parameter before reading probe outcomes.
    (OUT/'frozen-fit.json').write_text(json.dumps({'version':1,'fits':fit,'calibration_roots':sorted(set(cal.pre_root_id)),
        'probe_roots':sorted(set(probe.pre_root_id)),'no_root_overlap':True,'previous_test_pair_exclusions':len(old_pairs)},indent=2)+'\n')
    np.savez_compressed(OUT/'translated-atlas.npz',**{name:mesh.vertices for name,mesh in candidate.items()})
    for candidate_name,meshes in [('published_mirror',base_meshes),('local_translation',candidate)]:
        label,count=classify(probe[['x','y','z']].to_numpy(),meshes)
        probe[candidate_name+'_label']=label;probe[candidate_name+'_memberships']=count
    probe.to_parquet(OUT/'held-out-probe.parquet',index=False,compression='zstd')
    evaluations={};per_glom=[]
    for candidate_name in ['published_mirror','local_translation']:
        a=probe.copy();a['memberships']=a[candidate_name+'_memberships'];a['correct']=a[candidate_name+'_label']==a.glomerulus
        evaluations[candidate_name]=stats(a)
        for glom,b in a.groupby('glomerulus'):
            s=stats(b);s['glomerulus']=glom;s['candidate']=candidate_name
            s['adequate_probe']=s['samples']>=50 and s['roots']>=5
            s['point_gate_passed']=s['adequate_probe'] and s['unique_fraction']>=.70 and s['correct_unique_fraction']>=.80
            per_glom.append(s)
    a=probe.copy();a['memberships']=a.local_translation_memberships;a['correct']=a.local_translation_label==a.glomerulus
    grouped=a.groupby('pre_root_id').agg(total=('correct','size'),unique=('memberships',lambda x:int((x==1).sum())),correct=('correct','sum'))
    values=grouped.to_numpy(dtype=float);rng=np.random.default_rng(17003);boot=[]
    if len(values):
        for _ in range(2000):
            sums=values[rng.integers(len(values),size=len(values))].sum(axis=0)
            boot.append([sums[1]/sums[0],sums[2]/sums[1] if sums[1] else 0.])
    intervals=np.quantile(boot,[.025,.975],axis=0).T.tolist() if boot else [[0,0],[0,0]]
    accepted=intervals[0][0]>=.70 and intervals[1][0]>=.80 and len(values)>=5
    result={'version':1,'status':'prospective_registration_audit_complete','archive_rows_scanned':offset,
            'anchor_sites':len(anchors),'calibration_sites':len(cal),'new_probe_sites':len(probe),'new_probe_roots':len(values),
            'previously_tested_pair_exclusions':len(old_pairs),'available_translated_regions':len(candidate),
            'evaluations':evaluations,'root_cluster_bootstrap_95_intervals':{'unique_fraction':intervals[0],'correct_unique_fraction':intervals[1]},
            'pooled_registration_gate_passed':accepted,'per_glomerulus_point_gates_passed':sum(r['point_gate_passed'] for r in per_glom if r['candidate']=='local_translation'),
            'manual_boundary_validation':False,'electrical_compartments_created':0,'application_changed':False,
            'wall_seconds':time.monotonic()-start,'peak_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            'sources_reverified':all(sha(ROOT/p)==h for p,h in protocol['source_hashes'].items()) and sha(archive)==identity['sha256']}
    (OUT/'per-glomerulus-results.json').write_text(json.dumps(per_glom,indent=2)+'\n')
    (OUT/'results.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
