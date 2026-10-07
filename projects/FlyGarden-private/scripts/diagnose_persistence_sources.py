"""Exact-source steering and external-input local persistence hypotheses."""
import sys,json,argparse,time,fcntl,subprocess,shutil,collections
from pathlib import Path
import numpy as np,pandas as pd
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from flygarden.recording import atomic_json,space_check
from flygarden.continuous_candidate import file_sha
from scripts.diagnose_recovery_interfaces import mappings
import scripts.diagnose_downstream_persistence as base
OUT=ROOT/'reports/brain-integration/recovery/persistence-sources-v1'
PRIOR=ROOT/'reports/brain-integration/recovery/downstream-persistence-v1'
SEEDS=[11501,11502]
CASES=['local_recurrence_reference','local_plus_alpn_to_alln_off','local_plus_all_external_positive_off','steering_reference','steering_ps013_off','steering_top4_off','steering_all_positive_off']
SOURCES=base.SOURCES+['scripts/diagnose_persistence_sources.py','scripts/report_downstream_persistence.py',str((PRIOR/'results.json').relative_to(ROOT)),str((PRIOR/'protocol.json').relative_to(ROOT))]


def ranked_roots(results,target,case):
    total=collections.Counter();meta={}
    for s in results['source_rankings']:
        if s['target']==target and s['case']==case and s.get('sign','positive')=='positive':
            for x in s.get('top_positive_sources',s.get('top_sources',[])):
                total[x['root_id']]+=x['accepted_mV_per_second'];meta[x['root_id']]=x
    return [{**meta[r],'pooled_rank_score':float(v)} for r,v in total.most_common(4)]


def prepare():
    OUT.mkdir(exist_ok=True);hashes={n:file_sha(ROOT/n) for n in SOURCES}
    if (OUT/'protocol.json').exists():
        p=json.loads((OUT/'protocol.json').read_text());assert p['sources']==hashes;return p
    old=json.loads((PRIOR/'results.json').read_text());assert old['status']=='complete'
    local=ranked_roots(old,'both DM1 lPNs','positive_alln_recurrence_off')
    steering=ranked_roots(old,'DNa02_left','positive_alln_to_pn_off')
    assert len(local)==len(steering)==4 and steering[0]['cell_type']=='PS013'
    ids,mapping=mappings();lookup={str(v):i for i,v in enumerate(ids)}
    ann=pd.read_csv(ROOT/'data/annotations.tsv',sep='\t',low_memory=False).fillna('').set_index('root_id').reindex(ids).fillna('')
    def pop(indices):return [{'index':int(i),'root_id':str(ids[i]),'side':str(ann.iloc[i]['side'])} for i in indices]
    alln=np.flatnonzero(ann.cell_class.eq('ALLN'));alpn=np.flatnonzero(ann.cell_class.eq('ALPN'));pn=np.flatnonzero(ann.cell_type.eq('DM1_lPN'))
    for x in local+steering:
        x['index']=lookup[x['root_id']];assert ann.iloc[x['index']]['cell_type']==x['cell_type']
    assert all(x['index'] in set(alln) for x in local)
    mapping['ALLN']=pop(alln);mapping['ALPN']=pop(alpn);mapping['residual_local']=pop([x['index'] for x in local])
    dna=[n['index'] for k in ('DNa02_left','DNa02_right') for n in mapping[k]];left=np.array([n['index'] for n in mapping['DNa02_left']])
    selected=sorted(set(pn.tolist()+dna+[x['index'] for x in local]))
    con=pd.read_parquet(ROOT/'vendor/fly-brain/data/2025_Connectivity_783.parquet');pre=con.Presynaptic_Index.to_numpy(dtype=int);post=con.Postsynaptic_Index.to_numpy(dtype=int);signed=con['Excitatory x Connectivity'].to_numpy();positive=signed>0
    pa=np.isin(pre,alln);qa=np.isin(post,alln);qp=np.isin(post,pn);pl=np.isin(post,left)
    rec=positive&pa&qa;feed=positive&pa&qp
    masks=[rec,rec|(positive&np.isin(pre,alpn)&qa),rec|(positive&~pa&qa),feed,feed|(positive&pl&(pre==steering[0]['index'])),feed|(positive&pl&np.isin(pre,[x['index'] for x in steering])),feed|(positive&pl)]
    lesions={}
    for case,mask in zip(CASES,masks):
        rows=np.flatnonzero(mask);lesions[case]={'rows':rows.tolist(),'aggregated_records':len(rows),'anatomical_synapses':int(con.iloc[rows].Connectivity.sum()),'source_indices':np.unique(pre[rows]).tolist(),'target_indices':np.unique(post[rows]).tolist()}
    edges=[]
    for row in np.flatnonzero(np.isin(post,selected)):
        i,j=pre[row],post[row];edges.append({'row':int(row),'pre':int(i),'post':int(j),'signed_synapses':float(signed[row]),'pre_root_id':str(ids[i]),'post_root_id':str(ids[j]),'pre_cell_type':str(ann.iloc[i]['cell_type']),'pre_cell_class':str(ann.iloc[i]['cell_class']),'pre_side':str(ann.iloc[i]['side'])})
    # Descriptive upstream ranking from retained raw tails. Not accepted delivery:
    # previous recordings did not contain gates for these four new targets.
    nominal=[]
    for seed in (11401,11402):
        folder=PRIOR/'trials'/f'positive_alln_recurrence_off-{seed}';m=json.loads((folder/'manifest.json').read_text());counts=np.zeros(len(ids),dtype=np.int64)
        for chunk in m['chunks'][140:]:
            file=folder/chunk['file'];assert file_sha(file)==chunk['sha256']
            with np.load(file) as z:counts+=z['counts']
        mask=positive&~pa&np.isin(post,[x['index'] for x in local]);rows=np.flatnonzero(mask);groups={}
        for row in rows:
            cls=str(ann.iloc[pre[row]]['cell_class']) or 'unavailable';groups[cls]=groups.get(cls,0)+float(counts[pre[row]]/.5*signed[row]*.275)
        nominal.append({'seed':seed,'nominal_positive_external_arrival_mV_per_second_by_class':dict(sorted(groups.items(),key=lambda x:-x[1])),'scope':'Presynaptic counts times signed weights; gate/delay corrections unavailable for these targets in prior data. Hypothesis only.'})
    p={'version':1,'sources':hashes,'seeds':SEEDS,'cases':CASES,'mapping':mapping,'selected':selected,'incoming_edges':edges,'lesions':lesions,'clock':.0001,'window':.025,'duration':4.,'intervention_time':.8,
       'selected_local_targets':local,'selected_steering_sources':steering,'prior_external_hypotheses':nominal,
       'selection':'Four highest pooled scores among prior published top12 positive-source lists; not claimed to be a complete pooled all-source ranking. Prior11401/11402 used for selection;11501/11502 are new diagnostic seeds, not held-out navigation validation.',
       'schedule':'Full original network;65Hz walking support;50Hz left odorA .3-.8s. At.8s apply exact diagnostic mask only. All other weights/neurons/state/decoder unchanged; learning disabled.',
       'decision':'Compare local branches with local_recurrence_reference; steering branches with steering_reference. Strong suppression requires final.5s target-group mean <=10% matched reference in BOTH seeds, provided reference>10Hz. Report individual local neurons and DNa02 separately. These cuts are diagnostic, never proposed controllers.',
       'instrumentation':'One steering_reference11501 observer-disabled replay must match spikes/counts/inputs/commands and selected v/g/gates exactly.',
       'scope':'Full-network modeled connection dependence, not biological cause. ALLN/ALPN are exact cell-class annotations. Steering roots have exact cell types but unavailable cell classes; no function or new class inferred.'}
    atomic_json(OUT/'protocol.json',p)
    for n in SOURCES:
        if n.startswith(('data/','vendor/','reports/')):continue
        dst=OUT/'source'/n;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/n,dst)
    return p


base.OUT=OUT;base.SEEDS=SEEDS;base.CASES=CASES;base.prepare=prepare

def run():
    with (ROOT/'.runtime/experiment.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);p=prepare();done=[]
        jobs=[(c,s,False) for c in CASES for s in SEEDS]+[('steering_reference',SEEDS[0],True)]
        for case,seed,off in jobs:
            name=f"{case}-{seed}{'-observer-off' if off else ''}";path=OUT/'trials'/name/'manifest.json'
            if path.exists():
                m=json.loads(path.read_text());assert m['status']=='complete' and m['sources']==p['sources']
            else:
                space_check(OUT,250*1024**2);atomic_json(OUT/'progress.json',{'status':'running','completed':done,'current':name,'planned':15})
                args=[sys.executable,__file__,'--case',case,'--seed',str(seed)]
                if off:args.append('--unobserved')
                subprocess.run(args,cwd=ROOT,check=True)
            done.append(name)
        atomic_json(OUT/'progress.json',{'status':'complete','completed':done,'planned':15})

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--prepare',action='store_true');parser.add_argument('--case',choices=CASES);parser.add_argument('--seed',type=int);parser.add_argument('--unobserved',action='store_true');a=parser.parse_args()
    if a.prepare:prepare()
    elif a.case:base.worker(a.case,a.seed,a.unobserved)
    else:run()
