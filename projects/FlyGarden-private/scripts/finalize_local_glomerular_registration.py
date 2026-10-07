"""Finish saved probe statistics after the metadata-name shadowing repair.

No fitting, classification, parameter changes or threshold changes occur here.
"""
from pathlib import Path
import hashlib
import json
import time
import numpy as np
import pandas as pd
from scripts.validate_local_glomerular_registration import stats,sha,ROOT,V2,OUT

def main():
    original=json.loads((OUT/'protocol-original.json').read_text())
    script=ROOT/'scripts/validate_local_glomerular_registration.py'
    checks={}
    for p,h in original['source_hashes'].items():
        source=OUT/'worker-original.py' if ROOT/p==script else ROOT/p
        checks[p]=sha(source)==h
    if not all(checks.values()):raise ValueError('Frozen source mismatch')
    files=['held-out-probe.parquet','frozen-fit.json','anchor-sites.parquet','translated-atlas.npz']
    amendment={'version':1,'reason':'Final report referenced a loop label instead of the archive identity dictionary',
               'parameter_or_threshold_changes':False,'fitting_repeated':False,'probe_classification_repeated':False,
               'original_sources_verified':checks,'corrected_worker_sha256':sha(script),
               'finalizer_sha256':sha(Path(__file__).resolve()),'saved_inputs_sha256':{p:sha(OUT/p) for p in files}}
    (OUT/'report-finalization-amendment.json').write_text(json.dumps(amendment,indent=2)+'\n')
    probe=pd.read_parquet(OUT/'held-out-probe.parquet');anchors=pd.read_parquet(OUT/'anchor-sites.parquet')
    fit=json.loads((OUT/'frozen-fit.json').read_text());evaluations={};per_glom=[]
    for name in ['published_mirror','local_translation']:
        a=probe.copy();a['memberships']=a[name+'_memberships'];a['correct']=a[name+'_label']==a.glomerulus
        evaluations[name]=stats(a)
        for glom,b in a.groupby('glomerulus'):
            s=stats(b);s.update({'glomerulus':glom,'candidate':name,'adequate_probe':len(b)>=50 and b.pre_root_id.nunique()>=5})
            s['point_gate_passed']=s['adequate_probe'] and s['unique_fraction']>=.70 and s['correct_unique_fraction']>=.80
            per_glom.append(s)
    a=probe.copy();a['memberships']=a.local_translation_memberships;a['correct']=a.local_translation_label==a.glomerulus
    grouped=a.groupby('pre_root_id').agg(total=('correct','size'),unique=('memberships',lambda x:int((x==1).sum())),correct=('correct','sum'))
    values=grouped.to_numpy(dtype=float);rng=np.random.default_rng(17003);boot=[]
    for _ in range(2000):
        sums=values[rng.integers(len(values),size=len(values))].sum(axis=0)
        boot.append([sums[1]/sums[0],sums[2]/sums[1] if sums[1] else 0.])
    intervals=np.quantile(boot,[.025,.975],axis=0).T.tolist()
    elapsed=time.time()-(OUT/'protocol.json').stat().st_mtime
    result={'version':1,'status':'prospective_registration_audit_complete','anchor_sites':len(anchors),
            'calibration_sites':int((anchors.role=='calibration').sum()),'new_probe_sites':len(probe),'new_probe_roots':len(values),
            'previously_tested_pair_exclusions':fit['previous_test_pair_exclusions'],
            'available_translated_regions':sum(r['candidate_available'] for r in fit['fits']),
            'evaluations':evaluations,'root_cluster_bootstrap_95_intervals':{'unique_fraction':intervals[0],'correct_unique_fraction':intervals[1]},
            'pooled_registration_gate_passed':intervals[0][0]>=.70 and intervals[1][0]>=.80 and len(values)>=5,
            'per_glomerulus_point_gates_passed':sum(r['point_gate_passed'] for r in per_glom if r['candidate']=='local_translation'),
            'manual_boundary_validation':False,'electrical_compartments_created':0,'application_changed':False,
            'elapsed_since_original_protocol_seconds':elapsed,'sources_reverified':True,
            'archive_identity_reverified':sha(V2/'source/flywire_synapses_783.feather')==original['archive_identity']['sha256'],
            'software_report_repair':'report-finalization-amendment.json'}
    if not result['archive_identity_reverified']:raise ValueError('Archive changed')
    (OUT/'per-glomerulus-results.json').write_text(json.dumps(per_glom,indent=2)+'\n')
    (OUT/'results.json').write_text(json.dumps(result,indent=2)+'\n')
    (OUT/'progress.json').write_text(json.dumps({'status':'complete'})+'\n')
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
