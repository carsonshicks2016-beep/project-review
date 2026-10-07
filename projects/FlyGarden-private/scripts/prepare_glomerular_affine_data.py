"""Prepare calibration sites and seal a previously uninspected source-class probe."""
from pathlib import Path
import json
import shutil
import time
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.ipc as ipc
import pyarrow.parquet as pq
from flygarden.compartment_mapping import partner_glomerulus
from flygarden.local_atlas_registration import role
from scripts.validate_local_glomerular_registration import sha,ROOT,V1,V2

OUT=ROOT/'reports/brain-integration/recovery/glomerular-affine-registration-v1'

def main():
    start=time.monotonic();OUT.mkdir(parents=True,exist_ok=True)
    archive=V2/'source/flywire_synapses_783.feather';identity=json.loads((V2/'archive-identity.json').read_text())
    if sha(archive)!=identity['sha256']:raise ValueError('Archive identity mismatch')
    inputs=[ROOT/'data/annotations.tsv',ROOT/'flygarden/compartment_mapping.py',ROOT/'flygarden/local_atlas_registration.py',Path(__file__).resolve()]
    protocol={'version':1,'archive_identity':identity,'source_hashes':{str(p.relative_to(ROOT)):sha(p) for p in inputs},
              'calibration':'Cognate ORN(calibration role) -> uniglomerular PN; source AL_L or AL_R agrees with PN soma side',
              'fresh_probe':'Uniglomerular PN -> cognate ORN(probe role), source AL_R. Postsynaptic position in native nm.',
              'split':'Unchanged exact ORN-root SHA256 role from local registration v1',
              'freshness':'Only counts were inspected for feasibility; no earlier stage fitted or classified these PN -> ORN source synapses',
              'shared_probe_neurons':'Probe cohorts may share ORNs; source-class/synapse separation is not independent animals',
              'no_controller_change':True}
    (OUT/'data-protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    if shutil.disk_usage(OUT).free<2*1024**3+512*1024**2:raise RuntimeError('Disk reserve')
    annotations=pd.read_csv(ROOT/'data/annotations.tsv',sep='\t',dtype=str).fillna('')
    ann={int(r['root_id']):r for r in annotations.to_dict('records')};labels={rid:partner_glomerulus(a) for rid,a in ann.items()}
    orn={rid:v[0] for rid,v in labels.items() if v[1]=='ORN_annotation'}
    pn={rid:v[0] for rid,v in labels.items() if v[1]=='uniglomerular_PN_annotation'}
    cal_orn=np.array([r for r in orn if role(r)=='calibration'],dtype=np.int64)
    probe_orn=np.array([r for r in orn if role(r)=='probe'],dtype=np.int64);pn_ids=np.array(list(pn),dtype=np.int64)
    writers={};counts={'calibration':0,'sealed_probe':0};offset=0
    with pa.OSFile(str(archive),'rb') as source:
        reader=ipc.open_file(source)
        for i in range(reader.num_record_batches):
            if shutil.disk_usage(OUT).free<2*1024**3+128*1024**2:raise RuntimeError('Disk reserve; partial data preserved')
            batch=reader.get_batch(i);pre=batch.column(batch.schema.get_field_index('pre_pt_root_id')).to_numpy();post=batch.column(batch.schema.get_field_index('post_pt_root_id')).to_numpy()
            groups={'calibration':np.flatnonzero(np.isin(pre,cal_orn)&np.isin(post,pn_ids)),
                    'sealed_probe':np.flatnonzero(np.isin(pre,pn_ids)&np.isin(post,probe_orn))}
            for cohort,indices in groups.items():
                if not len(indices):continue
                columns=['id','pre_pt_root_id','post_pt_root_id','neuropil','post_pt_position_x','post_pt_position_y','post_pt_position_z']
                df=pa.Table.from_batches([batch]).select(columns).take(pa.array(indices)).to_pandas()
                selected=[];gloms=[];sides=[];root_groups=[]
                for k,(a,b,roi) in enumerate(zip(df.pre_pt_root_id,df.post_pt_root_id,df.neuropil)):
                    a,b=int(a),int(b)
                    if cohort=='calibration':
                        side=ann[b]['side'];valid=orn[a]==pn[b] and roi=={'left':'AL_L','right':'AL_R'}.get(side)
                        label=orn[a];group=str(a)
                    else:
                        side='right';valid=pn[a]==orn[b] and roi=='AL_R';label=orn[b];group=str(b)
                    if valid:selected.append(k);gloms.append(label);sides.append(side);root_groups.append(group)
                if not selected:continue
                a=df.iloc[selected]
                frame=pd.DataFrame({'synapse_id':a.id.astype(str).to_numpy(),'pre_root_id':a.pre_pt_root_id.astype(str).to_numpy(),
                     'post_root_id':a.post_pt_root_id.astype(str).to_numpy(),'ORN_root_id':root_groups,'glomerulus':gloms,'side':sides,
                     'x':a.post_pt_position_x.to_numpy(),'y':a.post_pt_position_y.to_numpy(),'z':a.post_pt_position_z.to_numpy(),
                     'source_row':indices[np.array(selected)]+offset})
                table=pa.Table.from_pandas(frame,preserve_index=False)
                if cohort not in writers:writers[cohort]=pq.ParquetWriter(OUT/(cohort+'.parquet.partial'),table.schema,compression='zstd')
                writers[cohort].write_table(table);counts[cohort]+=len(frame)
            offset+=batch.num_rows
            if i%50==0:(OUT/'data-progress.json').write_text(json.dumps({'status':'extracting','batches_done':i+1,'batches':reader.num_record_batches,'counts':counts})+'\n')
    for cohort,writer in writers.items():writer.close();(OUT/(cohort+'.parquet.partial')).replace(OUT/(cohort+'.parquet'))
    cal=pd.read_parquet(OUT/'calibration.parquet');probe=pd.read_parquet(OUT/'sealed_probe.parquet',columns=['synapse_id','ORN_root_id'])
    assert not set(cal.ORN_root_id)&set(probe.ORN_root_id)
    assert not set(cal.synapse_id)&set(probe.synapse_id)
    for previous in ['anchor-sites.parquet','secondary-source-probe.parquet']:
        ids=pd.read_parquet(OUT.parent/'local-glomerular-registration-v1'/previous,columns=['synapse_id'])
        assert not set(probe.synapse_id)&set(ids.synapse_id)
    result={'version':1,'rows_scanned':offset,'counts':counts,'no_calibration_ORN_overlap':True,'no_prior_source_synapse_overlap':True,
            'probe_positions_not_classified':True,'sealed_probe_sha256':sha(OUT/'sealed_probe.parquet'),
            'calibration_sha256':sha(OUT/'calibration.parquet'),'wall_seconds':time.monotonic()-start,
            'sources_reverified':all(sha(ROOT/p)==h for p,h in protocol['source_hashes'].items())}
    (OUT/'data-results.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))

if __name__=='__main__':main()
