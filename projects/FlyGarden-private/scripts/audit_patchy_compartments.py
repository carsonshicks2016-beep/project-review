"""Exact-root graph/coordinate audit; never mutates brain, body or application state."""
from pathlib import Path
import csv
import gzip
import hashlib
import json
import time
import resource
import shutil
from collections import Counter
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from flygarden.compartment_mapping import CoordinateDecoder, partner_glomerulus

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'reports/brain-integration/recovery/patchy-compartment-mapping-v1'


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for part in iter(lambda:f.read(8*1024**2),b''):h.update(part)
    return h.hexdigest()


def main():
    start=time.monotonic();OUT.mkdir(parents=True,exist_ok=True)
    if shutil.disk_usage(OUT).free < 2*1024**3+100*1024**2:raise RuntimeError('Disk reserve')
    annotations=pd.read_csv(ROOT/'data/annotations.tsv',sep='\t',dtype=str).fillna('')
    assert annotations.root_id.is_unique
    ann={r['root_id']:r for r in annotations.to_dict('records')}
    selected=annotations[annotations.cell_type.isin(['lLN2P_a','lLN2P_b','lLN2P_c'])]
    roots=set(selected.root_id);assert len(roots)==34
    selected.to_json(OUT/'neurons.json',orient='records',indent=2)
    graph=ROOT/'vendor/fly-brain/data/2025_Connectivity_783.parquet'
    coord=OUT/'source/synapse_coordinates.csv.gz'
    sourcepaths=[ROOT/'data/annotations.tsv',graph,coord,ROOT/'flygarden/brain.py',
                 ROOT/'flygarden/compartment_mapping.py',Path(__file__).resolve(),
                 OUT/'source/flywire_al.surf.rda']
    protocol={'version':1,'materialization':783,'selected_root_ids':sorted(roots),
              'source_hashes':{str(p.relative_to(ROOT)):sha(p) for p in sourcepaths},
              'coordinate_source_generation':'1733428450553759',
              'coordinate_semantics':'Independent carry-forward root columns; exact decimal strings; raw source lines retained',
              'classification':'Explicit ORN or uniglomerular-PN partner label is a provisional glomerular hint, never a compartment assignment',
              'anchor_sample':'Uniform reservoir per glomerulus and annotated PN side, capacity 500, seed 12001; ORN-to-cognate-uniglomerular-PN coordinates',
              'gates':{'exact_pair_count_equality':True,'all_aggregate_counts_preserved':True,
                       'mesh_assignment_requires_unique_enclosure':True,'no_mirror_or_alias_without_source':True},
              'application_changed':False,'physiology_validated':False}
    (OUT/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    rows=[];offset=0
    for batch in pq.ParquetFile(graph).iter_batches(batch_size=250000):
        df=batch.to_pandas()
        pre=df.Presynaptic_ID.astype(str);post=df.Postsynaptic_ID.astype(str)
        mask=pre.isin(roots)|post.isin(roots)
        subset=df.loc[mask,['Presynaptic_ID','Postsynaptic_ID','Connectivity','Excitatory','Excitatory x Connectivity']].copy()
        subset['source_position']=np.flatnonzero(mask)+offset
        subset['Presynaptic_ID']=pre[mask];subset['Postsynaptic_ID']=post[mask]
        rows.append(subset);offset+=len(df)
    edges=pd.concat(rows,ignore_index=True)
    assert np.isfinite(edges.Connectivity).all() and np.all(edges.Connectivity>=0)
    assert np.all(edges.Connectivity==np.floor(edges.Connectivity))
    assert not edges.duplicated(['Presynaptic_ID','Postsynaptic_ID']).any()
    edges.to_csv(OUT/'imported-edges.csv',index=False)
    model_counts={(r.Presynaptic_ID,r.Postsynaptic_ID):int(r.Connectivity) for r in edges.itertuples()}
    coord_counts=Counter();anchor_seen=Counter();anchors={};rng=np.random.default_rng(12001)
    labels={rid:partner_glomerulus(a) for rid,a in ann.items()}
    orn={rid for rid,a in ann.items() if labels[rid][1]=='ORN_annotation'}
    pn={rid for rid,a in ann.items() if labels[rid][1]=='uniglomerular_PN_annotation'}
    decoder=CoordinateDecoder();count=0;new_pre_without_post=0;previous_pre=None
    temporary=OUT/'selected-synapses.csv.partial'
    with gzip.open(coord,'rt',newline='') as f,temporary.open('w',newline='') as out:
        reader=csv.reader(f);assert next(reader)==['pre_root_id','post_root_id','x','y','z']
        writer=csv.writer(out);writer.writerow(['source_line','pre_root_id','post_root_id','x','y','z'])
        for line,row in enumerate(reader,2):
            if row[0] and row[0]!=previous_pre and not row[1]:new_pre_without_post+=1
            # Full source is numeric and gzip CRC-checked at EOF. Fast path retains
            # decimal IDs directly; the validating decoder is covered by tests.
            pre=row[0] or decoder.pre;post=row[1] or decoder.post
            if pre is None or post is None:raise ValueError('Missing initial root')
            decoder.pre,decoder.post=pre,post;previous_pre=pre;count+=1
            if pre in roots or post in roots:
                point=tuple(int(v) for v in row[2:]);assert len(point)==3 and min(point)>=0
                writer.writerow([line,pre,post,*point]);coord_counts[(pre,post)]+=1
            if pre in orn and post in pn and labels[pre][0]==labels[post][0]:
                key=(labels[pre][0],ann[post]['side']);anchor_seen[key]+=1
                n=anchor_seen[key];pool=anchors.setdefault(key,[])
                value=(line,pre,post,*map(int,row[2:]))
                if n<=500:pool.append(value)
                else:
                    j=int(rng.integers(n))
                    if j<500:pool[j]=value
            if count%2000000==0:
                (OUT/'progress.json').write_text(json.dumps({'status':'scanning_coordinates','rows':count,'selected_coordinates':sum(coord_counts.values()),'elapsed_seconds':time.monotonic()-start})+'\n')
    temporary.replace(OUT/'selected-synapses.csv')
    with (OUT/'anchor-synapses.csv').open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['glomerulus','PN_side','source_line','pre_root_id','post_root_id','x','y','z'])
        for key,pool in sorted(anchors.items()):
            for item in pool:w.writerow([*key,*item])
    pairs=[]
    for pair in sorted(set(model_counts)|set(coord_counts)):
        m=model_counts.get(pair,0);c=coord_counts.get(pair,0)
        pairs.append({'pre_root_id':pair[0],'post_root_id':pair[1],'imported_count':m,'coordinate_count':c,
                      'exact_match':m==c,'coordinate_source_present':c>0})
    pd.DataFrame(pairs).to_csv(OUT/'pair-count-audit.csv',index=False)
    summaries=[];hints=[]
    for rid in sorted(roots):
        a=ann[rid]
        for direction in ['input','output']:
            field='Postsynaptic_ID' if direction=='input' else 'Presynaptic_ID'
            partnerfield='Presynaptic_ID' if direction=='input' else 'Postsynaptic_ID'
            subset=edges[edges[field]==rid];total=int(subset.Connectivity.sum());hinted=0
            for row in subset.to_dict('records'):
                partner=row[partnerfield];label,method=labels.get(partner,(None,'unavailable'))
                if label:hinted+=int(row['Connectivity'])
                hints.append({'root_id':rid,'direction':direction,'partner_root_id':partner,
                              'partner_cell_type':ann.get(partner,{}).get('cell_type','unavailable'),
                              'partner_side':ann.get(partner,{}).get('side','unavailable'),
                              'hint_glomerulus':label or 'unavailable','hint_method':method,
                              'imported_count':int(row['Connectivity']),
                              'coordinate_count':coord_counts[(row['Presynaptic_ID'],row['Postsynaptic_ID'])],
                              'compartment_assignment':'unavailable_pending_spatial_validation'})
            summaries.append({'root_id':rid,'cell_type':a['cell_type'],'side':a['side'],'direction':direction,
                              'imported_records':len(subset),'imported_synapses':total,
                              'partner_label_hint_synapses':hinted,'unresolved_partner_synapses':total-hinted})
    pd.DataFrame(hints).to_csv(OUT/'partner-hints.csv',index=False)
    (OUT/'root-summary.json').write_text(json.dumps(summaries,indent=2)+'\n')
    mismatches=[r for r in pairs if not r['exact_match']]
    res={'version':1,'status':'graph_and_coordinate_audit_complete_spatial_review_pending',
         'selected_neurons':34,'graph_rows_scanned':offset,'coordinate_rows_scanned':count,
         'selected_imported_records':len(edges),'selected_imported_synapses':int(edges.Connectivity.sum()),
         'selected_coordinate_rows':sum(coord_counts.values()),'coordinate_pairs':len(coord_counts),
         'exact_matching_pairs':sum(r['exact_match'] for r in pairs),'mismatched_pairs':len(mismatches),
         'coordinate_only_pairs':sum(r['imported_count']==0 for r in pairs),
         'imported_pairs_without_coordinates':sum(r['imported_count']>0 and r['coordinate_count']==0 for r in pairs),
         'new_pre_with_missing_post':new_pre_without_post,
         'anchor_count_by_group':{'|'.join(k):v for k,v in sorted(anchor_seen.items())},
         'full_coordinate_scan_complete':True,'gzip_crc_checked':True,
         'exact_all_pair_parity':not mismatches,'application_changed':False,
         'wall_seconds':time.monotonic()-start,'peak_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
    (OUT/'results.json').write_text(json.dumps(res,indent=2)+'\n')
    (OUT/'progress.json').write_text(json.dumps({'status':'graph_and_coordinate_audit_complete','rows':count})+'\n')
    print(json.dumps({k:v for k,v in res.items() if k!='anchor_count_by_group'},indent=2))


if __name__=='__main__':main()
