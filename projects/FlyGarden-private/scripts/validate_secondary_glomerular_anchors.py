"""Independent source-class probe of the frozen local registration.

Uses only held-out ORN -> cognate ORN synapses in source neuropil AL_R.
Does not refit the existing candidate or change any controller.
"""
from pathlib import Path
import json
import resource
import shutil
import time
import warnings
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.ipc as ipc
import rdata
import trimesh
from flygarden.compartment_mapping import partner_glomerulus
from flygarden.local_atlas_registration import role
from scripts.validate_local_glomerular_registration import sha,stats,classify,ROOT,V1,V2,OUT

def main():
    start=time.monotonic()
    inputs=[ROOT/'data/annotations.tsv',OUT/'translated-atlas.npz',OUT/'frozen-fit.json',
            V1/'source/flywire_al.surf.rda',V2/'mirrored-atlas-vertices.npz',
            ROOT/'flygarden/local_atlas_registration.py',ROOT/'scripts/validate_local_glomerular_registration.py',Path(__file__).resolve()]
    identity=json.loads((V2/'archive-identity.json').read_text())
    protocol={'version':1,'source_hashes':{str(p.relative_to(ROOT)):sha(p) for p in inputs},'archive_identity':identity,
              'probe':'ORN -> cognate ORN, both exact roots in the frozen probe group; no autapses; source neuropil AL_R',
              'source_class_independent':'No ORN -> ORN site was used in the fit or primary probe',
              'coordinate':'post_pt_position in native nanometers',
              'candidate_fit_repeated':False,'min_glomerulus_sites':50,'min_glomerulus_roots':5,
              'gates':{'unique_fraction_min':.70,'correct_unique_fraction_min':.80},
              'bootstrap':{'unit':'presynaptic ORN root','iterations':2000,'seed':17004},
              'limits':'Grouped anatomical consistency test, not independently traced boundaries or physiology',
              'controller_changed':False}
    (OUT/'secondary-protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    if shutil.disk_usage(OUT).free<2*1024**3+512*1024**2:raise RuntimeError('Disk reserve')
    annotations=pd.read_csv(ROOT/'data/annotations.tsv',sep='\t',dtype=str).fillna('')
    orn={int(r['root_id']):partner_glomerulus(r)[0] for r in annotations.to_dict('records')
         if partner_glomerulus(r)[1]=='ORN_annotation' and role(r['root_id'])=='probe'}
    ids=np.array(list(orn),dtype=np.int64);parts=[];offset=0
    with pa.OSFile(str(V2/'source/flywire_synapses_783.feather'),'rb') as f:
        reader=ipc.open_file(f)
        for i in range(reader.num_record_batches):
            b=reader.get_batch(i)
            pre=b.column(b.schema.get_field_index('pre_pt_root_id')).to_numpy();post=b.column(b.schema.get_field_index('post_pt_root_id')).to_numpy()
            indices=np.flatnonzero(np.isin(pre,ids)&np.isin(post,ids)&(pre!=post))
            indices=np.array([j for j in indices if orn[int(pre[j])]==orn[int(post[j])]],dtype=np.int64)
            if len(indices):
                t=pa.Table.from_batches([b]).select(['id','pre_pt_root_id','post_pt_root_id','neuropil','post_pt_position_x','post_pt_position_y','post_pt_position_z']).take(pa.array(indices))
                a=t.to_pandas();a['source_row']=indices+offset;a=a[a.neuropil=='AL_R'].copy()
                if len(a):
                    a['pre_root_id']=a.pre_pt_root_id.astype(str);a['post_root_id']=a.post_pt_root_id.astype(str)
                    a['glomerulus']=[orn[int(x)] for x in a.pre_pt_root_id]
                    a['synapse_id']=a.id.astype(str)
                    a.rename(columns={'post_pt_position_x':'x','post_pt_position_y':'y','post_pt_position_z':'z'},inplace=True)
                    parts.append(a[['synapse_id','pre_root_id','post_root_id','glomerulus','x','y','z','source_row']])
            offset+=b.num_rows
            if i%50==0:(OUT/'secondary-progress.json').write_text(json.dumps({'status':'extracting','batches_done':i+1,'batches':reader.num_record_batches,'probe_sites':sum(len(a) for a in parts)})+'\n')
    if not parts:raise ValueError('No independent ORN-to-ORN anchors')
    probe=pd.concat(parts,ignore_index=True)
    fit=json.loads((OUT/'frozen-fit.json').read_text());cal=set(fit['calibration_roots'])
    assert not cal&set(probe.pre_root_id) and not cal&set(probe.post_root_id)
    primary=pd.read_parquet(OUT/'anchor-sites.parquet',columns=['synapse_id'])
    assert not set(primary.synapse_id)&set(probe.synapse_id)
    assert not probe.synapse_id.duplicated().any()
    probe.to_parquet(OUT/'secondary-source-probe.parquet',index=False,compression='zstd')
    warnings.filterwarnings('ignore',message='Missing constructor for R class')
    data=rdata.read_rda(V1/'source/flywire_al.surf.rda')['flywire_al.surf']
    v=np.load(V2/'mirrored-atlas-vertices.npz')['vertices'];translated=np.load(OUT/'translated-atlas.npz')
    base={};candidate={}
    for name,frame in data['Regions'].items():
        faces=frame.to_numpy(dtype=np.int64)-1;used=np.unique(faces);remap=np.full(len(v),-1);remap[used]=np.arange(len(used))
        m=trimesh.Trimesh(v[used],remap[faces][:,::-1],process=True);base[str(name)]=m
        if str(name) in translated:
            c=m.copy();assert c.vertices.shape==translated[str(name)].shape
            c.vertices=translated[str(name)];candidate[str(name)]=c
    evaluations={};per_glom=[]
    for name,meshes in [('published_mirror',base),('local_translation',candidate)]:
        labels,count=classify(probe[['x','y','z']].to_numpy(),meshes)
        probe[name+'_label']=labels;probe[name+'_memberships']=count
        a=probe.copy();a['memberships']=count;a['correct']=labels==a.glomerulus
        evaluations[name]=stats(a)
        for glom,b in a.groupby('glomerulus'):
            s=stats(b);s.update({'glomerulus':glom,'candidate':name,'adequate_probe':len(b)>=50 and b.pre_root_id.nunique()>=5})
            s['point_gate_passed']=s['adequate_probe'] and s['unique_fraction']>=.70 and s['correct_unique_fraction']>=.80
            per_glom.append(s)
    probe.to_parquet(OUT/'secondary-held-out-probe.parquet',index=False,compression='zstd')
    a=probe.copy();a['memberships']=a.local_translation_memberships;a['correct']=a.local_translation_label==a.glomerulus
    groups=a.groupby('pre_root_id').agg(total=('correct','size'),unique=('memberships',lambda s:int((s==1).sum())),correct=('correct','sum'))
    values=groups.to_numpy(dtype=float);rng=np.random.default_rng(17004);boot=[]
    for _ in range(2000):
        total,unique,correct=values[rng.integers(len(values),size=len(values))].sum(axis=0)
        boot.append([unique/total,correct/unique if unique else 0.])
    intervals=np.quantile(boot,[.025,.975],axis=0).T.tolist()
    result={'version':1,'status':'secondary_probe_complete','archive_rows_scanned':offset,
            'probe_sites':len(probe),'probe_roots':len(values),'evaluations':evaluations,
            'root_cluster_bootstrap_95_intervals':{'unique_fraction':intervals[0],'correct_unique_fraction':intervals[1]},
            'pooled_gate_passed':intervals[0][0]>=.70 and intervals[1][0]>=.80,
            'per_glomerulus_point_gates_passed':sum(s['point_gate_passed'] for s in per_glom if s['candidate']=='local_translation'),
            'no_calibration_neuron_or_source_site_overlap':True,'no_candidate_refit':True,'application_changed':False,
            'wall_seconds':time.monotonic()-start,'peak_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            'sources_reverified':all(sha(ROOT/p)==h for p,h in protocol['source_hashes'].items()) and sha(V2/'source/flywire_synapses_783.feather')==identity['sha256']}
    (OUT/'secondary-per-glomerulus-results.json').write_text(json.dumps(per_glom,indent=2)+'\n')
    (OUT/'secondary-results.json').write_text(json.dumps(result,indent=2)+'\n')
    (OUT/'secondary-progress.json').write_text(json.dumps({'status':'complete'})+'\n')
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
